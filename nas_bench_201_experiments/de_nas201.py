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
import logging
import sys
from typing import Optional

# ログ設定
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

datasets = "CIFAR10"

# 設定
configs = {
    'result_path': "results/0701/",

    'INITIAL_F': 0.5,
    'INITIAL_CR': 0.5,
    'F_GAMMA': 0.1,
    'CR_GAMMA': 0.1,
    'F_LOWEST': 0.1,
    'F_UPPER': 0.9,
    'DIMENSION': 64,
    'POPULATION_SIZE': 500,
    'GENERATION': 100,

    'dataset': datasets,  # 'CIFAR10', 'CIFAR100', 'ImageNet'
    'gvae_path': 'gvae/gvae_64_{}.pth'.format(datasets),
    'predictor_path': 'semi_predictor/semi_predictor_{}.pth'.format(datasets),
    'latent_path': 'dataset/latent_representations_64dim.pth',
    'seed': 42
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
            # TODO：gvaeのエンコードで取得した潜在表現を使用する
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

    def evolve(self, generations=None):
        if generations is None:
            generations = configs['GENERATION']
        logging.info("Starting DE...")
        
        for generation in range(generations):
            updated_count = self.mutation_and_crossover()
            avg_fitness = self.get_average_fitness()
            best_individual = self.get_best_individual(1)[0]
            
            logging.info(f"Generation {generation+1}: Updated={updated_count}, "
                        f"Average Fitness={avg_fitness:.6f}, Best Fitness={best_individual.fitness:.6f}")
        
        return self.get_best_individual(top_n=5)

# TODO：評価関数の構造を見直す
def create_evaluation_function(gvae, predictor):
    """GVAEとpredictorを使用した評価関数を作成"""
    def evaluate(latent_vector):
        with torch.no_grad():
            z = latent_vector.unsqueeze(0).cuda()
            
            # 潜在表現からアーキテクチャ文字列を生成
            arch_tensor = gvae.get_tensor(z)
            arch_str = gvae.conver_tensor2arch(arch_tensor)
            
            # アーキテクチャ文字列をリストに変換
            arch_list = arch2list(arch_str)
            
            # アーキテクチャリストからグラフ構造を作成
            edge_index, node_attr, edge_attr, cell_tensor = conver_cell2graph(arch_list)
            
            # PyTorch Geometricのデータ構造に変換
            graph_data = Data(
                x=node_attr,
                edge_index=edge_index,
                edge_attr=edge_attr,
                tensor=cell_tensor
            )
            graph_data = graph_data.cuda()
            
            # predictorで性能を予測
            pred_acc = predictor(graph_data)
            
            return pred_acc.item()
    
    return evaluate

def convert_latent_to_architecture(gvae, latent_vector):
    """潜在表現をアーキテクチャ文字列に変換"""
    with torch.no_grad():
        z = latent_vector.unsqueeze(0).cuda()
        arch_tensor = gvae.get_tensor(z)
        arch_str = gvae.conver_tensor2arch(arch_tensor)
        return arch_str

def main():
    logging.info("Loading Model...")
    gvae = torch.load(configs['gvae_path'], weights_only=False).cuda()
    predictor = torch.load(configs['predictor_path'], weights_only=False).cuda()
    
    gvae.eval()
    predictor.eval()
    
    eval_func = create_evaluation_function(gvae, predictor)
    
    de = DE(eval_func, population_size=configs['POPULATION_SIZE'], seed=configs['seed'])
    best_individuals = de.evolve(generations=configs['GENERATION'])
    
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