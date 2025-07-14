import os
import torch
import torch.nn as nn
import numpy as np
import random
import operator
from models import ArchGVAE, GNN_Predictor
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from collect_201_dataset import conver_cell2graph, arch2list
from models import ICNN
import logging
import sys
from typing import Optional
from torch.utils.data import Dataset, DataLoader
import tqdm
from utils import AvgrageMeter

# ログ設定
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

datasets = "ImageNet"

# 設定
configs = {
    'result_path': "results/0715/",

    'INITIAL_F': 0.5,
    'INITIAL_CR': 0.5,
    'F_GAMMA': 0.1,
    'CR_GAMMA': 0.1,
    'F_LOWEST': 0.1,
    'F_UPPER': 0.9,
    'DIMENSION': 64,
    'POPULATION_SIZE': 200,
    'GENERATION': 50,

    'dataset': datasets,  # 'CIFAR10', 'CIFAR100', 'ImageNet'
    'gvae_path': 'gvae/gvae_64_{}.pth'.format(datasets),
    # 'predictor_path': 'semi_predictor/semi_predictor_{}.pth'.format(datasets),
    'predictor_path': 'icnn/icnn_64_{}.pth'.format(datasets),
    'latent_path': 'dataset/latent_representations_64dim_{}.pth'.format(datasets),
    'seed': 42,
    
    # ICNN微調整設定
    'icnn_lr': 1e-4,
    'icnn_betas': (0.0, 0.5),
    'icnn_epoch_num': 10,
    'icnn_batch_size': 32,
    'tune_interval': 10  # 何世代ごとにICNN微調整するか
}

latent_data = torch.load(configs['latent_path'],weights_only=False)

def get_border_vaules(latent_data):
    MAX_value = [-float("inf") for i in range(configs["DIMENSION"])]
    MIN_value = [float("inf") for i in range(configs["DIMENSION"])]

    for di in range(configs["DIMENSION"]):
        for latent in latent_data:
            MAX_value[di] = max(latent[di],MAX_value[di])
            MIN_value[di] = min(latent[di],MIN_value[di])
    
    return MAX_value, MIN_value
configs['MAX_VALUES'], configs['MIN_VALUES'] = get_border_vaules(latent_data)

os.makedirs(configs['result_path'], exist_ok=True)

class Individual:
    def __init__(self, gene: Optional[torch.tensor]=None, eval_func=None):
        self.gene = gene
        self.eval_func = eval_func
        self.fitness = self.evaluate()
        self.F = configs['INITIAL_F']
        self.CR = configs['INITIAL_CR']

    def evaluate(self):
        if self.eval_func is None:
            print("none function")
            return 0.0
        return float(self.eval_func(self.gene))

class DE:
    def __init__(self, eval_func, population_size=None, seed=42):
        if population_size is None:
            population_size = configs['POPULATION_SIZE']
        self.eval_func = eval_func
        self.population_size = population_size
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.population = []
        for i in range(population_size):
            gene = latent_data[i]
            individual = Individual(gene=gene, eval_func=eval_func)
            self.population.append(individual)
        
        logging.info(f"Initialize Size: {population_size}")

    def mutation_and_crossover(self):
        updated_count = 0
        
        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            np.random.shuffle(indices)
            
            if np.random.rand() < configs['F_GAMMA']:
                self.population[i].F = configs['F_LOWEST'] + np.random.rand() * (configs['F_UPPER'] - configs['F_LOWEST'])
            
            if np.random.rand() < configs['CR_GAMMA']:
                self.population[i].CR = np.random.rand()
            
            mutant = (self.population[indices[0]].gene + 
                     self.population[i].F * (self.population[indices[1]].gene - self.population[indices[2]].gene))
            
            trial = torch.zeros_like(self.population[i].gene)
            jrand = np.random.randint(configs['DIMENSION'])
            
            for j in range(configs['DIMENSION']):
                if np.random.rand() < self.population[i].CR or j == jrand:
                    if configs['MIN_VALUES'][j] <= mutant[j] <= configs['MAX_VALUES'][j]:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = self.population[i].gene[j]
                else:
                    trial[j] = self.population[i].gene[j]
            
            trial_individual = Individual(gene=trial, eval_func=self.eval_func)
            trial_individual.F = self.population[i].F
            trial_individual.CR = self.population[i].CR
            
            if trial_individual.fitness > self.population[i].fitness:
                self.population[i] = trial_individual
                updated_count += 1
        
        return updated_count

    def get_best_individual(self, top_n=1):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        return self.population[:top_n]

    def get_average_fitness(self):
        total_fitness = sum(ind.fitness for ind in self.population)
        return total_fitness / self.population_size

    def evolve(self, generations=None, icnn=None):
        if generations is None:
            generations = configs['GENERATION']
        logging.info("Starting DE...")
        
        for generation in range(generations):
            # ICNN微調整（指定間隔で実行）
            if icnn is not None and generation % configs['tune_interval'] == 0 and generation > 0:
                logging.info(f"ICNN微調整実行中... (Generation {generation+1})")
                icnn = tune_icnn(icnn, self.population)
                # 評価関数を更新
                eval_func = create_evaluation_function(None, icnn)
                for individual in self.population:
                    individual.eval_func = eval_func
                    individual.fitness = individual.evaluate()
            
            updated_count = self.mutation_and_crossover()
            avg_fitness = self.get_average_fitness()
            best_individual = self.get_best_individual(1)[0]
            
            logging.info(f"Generation {generation+1}: Updated={updated_count}, "
                        f"Average Fitness={avg_fitness:.6f}, Best Fitness={best_individual.fitness:.6f}")
        
        return self.get_best_individual(top_n=5)

# TODO：評価関数の構造を見直す
def create_evaluation_function(gvae, icnn):
    """GVAEとICNNを使用した評価関数を作成"""
    def evaluate(latent_vector):
        with torch.no_grad():
            z = latent_vector.unsqueeze(0).cuda()
            
            # ICNNで直接潜在表現から性能を予測
            pred_acc = (-icnn(z) + 1.0).squeeze()
            
            # crlso.pyと同じスケールに合わせる（0.01倍）
            pred_acc = pred_acc * 0.01
            
            return pred_acc.item()
    
    return evaluate

def convert_latent_to_architecture(gvae, latent_vector):
    """潜在表現をアーキテクチャ文字列に変換"""
    with torch.no_grad():
        z = latent_vector.unsqueeze(0).cuda()
        arch_tensor = gvae.get_tensor(z)
        arch_str = gvae.conver_tensor2arch(arch_tensor)
        return arch_str

class ICNN_Dataset(Dataset):
    def __init__(self, latent_vectors, fitness_values):
        super().__init__()
        self.latent_vectors = latent_vectors
        self.fitness_values = fitness_values

    def __getitem__(self, idx):
        return self.latent_vectors[idx], self.fitness_values[idx]

    def __len__(self):
        return len(self.latent_vectors)

def tune_icnn(icnn, population):
    """現在の集団を使ってICNNを微調整"""
    # 集団から潜在ベクトルと適応度を抽出
    latent_vectors = torch.stack([ind.gene for ind in population])
    fitness_values = torch.tensor([ind.fitness for ind in population], dtype=torch.float32)
    
    # 適応度を100倍してICNNの元スケールに戻す
    fitness_values = fitness_values / 0.01
    
    dataset = ICNN_Dataset(latent_vectors, fitness_values)
    dataloader = DataLoader(dataset, batch_size=configs['icnn_batch_size'], shuffle=True)
    
    optimizer = torch.optim.Adam(
        icnn.parameters(),
        lr=configs['icnn_lr'], 
        betas=configs['icnn_betas'], 
        weight_decay=1e-5
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=float(configs['icnn_epoch_num']), eta_min=1e-5
    )
    
    mse = nn.MSELoss(reduction='mean')
    
    for epoch in range(configs['icnn_epoch_num']):
        objs = AvgrageMeter()
        
        for step, (latents, fitness) in enumerate(dataloader):
            n = len(latents)
            latents = latents.cuda()
            fitness = fitness.cuda()
            
            # ICNNの出力を適応度と一致させる
            pred_fitness = (-icnn(latents) + 1.0).squeeze()
            
            loss = mse(fitness, pred_fitness)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            objs.update(loss.data.item(), n)
            
            # ICNN制約の適用
            icnn.constraint_weights()
        
        scheduler.step()
    
    logging.info('ICNN微調整完了, loss: %e', objs.avg)
    return icnn

def main():
    logging.info("Loading Model...")
    gvae = torch.load(configs['gvae_path'], weights_only=False).cuda()
    
    # Load ICNN from the saved state_dict
    icnn = ICNN(input_dim=configs['DIMENSION'], hidden_dim=256, output_dim=1).cuda()
    icnn.load_state_dict(torch.load(configs['predictor_path'], weights_only=True))
    
    gvae.eval()
    icnn.eval()
    
    eval_func = create_evaluation_function(gvae, icnn)
    
    de = DE(eval_func, population_size=configs['POPULATION_SIZE'], seed=configs['seed'])
    best_individuals = de.evolve(generations=configs['GENERATION'], icnn=icnn)
    
    logging.info("=== DE Results ===")
    for i, individual in enumerate(best_individuals):
        arch_str = convert_latent_to_architecture(gvae, individual.gene)
        logging.info(f"Rank {i+1}: Predicted={individual.fitness:.6f}, Architecture={arch_str}")

    best_latents = torch.stack([ind.gene for ind in best_individuals])
    torch.save(best_latents, f'{configs["result_path"]}de_best_latents_{configs["dataset"]}.pth')
    
    best_fitnesses = [ind.fitness for ind in best_individuals]
    torch.save(best_fitnesses, f'{configs["result_path"]}de_best_fitnesses_{configs["dataset"]}.pth')
    
    logging.info("Save the results")

if __name__ == "__main__":
    main()