from collect_201_dataset import NAS_Bench_201_Dataset, random_sample_a_genotype, conver_cell2graph
from models import ArchGVAE, GNN_Predictor
import torch.nn as nn
import logging
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import AvgrageMeter, create_exp_dir
import tqdm
from train_gvae_semi_supervised import train_gvae
from nas_201_api import NASBench201API as API
from torch_geometric.data import Data
from copy import deepcopy
from nas_201_database import NASBench201DataBase
import random
import operator
from models import ICNN
from typing import Optional

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = log_format, datefmt = '%m/%d %I:%M:%S %p')

configs = {
    'nas_bench_201_dataset_path' : 'dataset/nas_201_dataset.pth',

    # which dataset to evaluate?
    'dataset' : 'ImageNet',
    # the maximum evaluation number
    'evaluate_num' : 200,  # 500 → 200 for faster execution

    # hyperparameters of fine-tunning the ICNN 
    'lr' : 1e-4,
    'betas' : (0.0, 0.5),
    'weight_decay' : 0.0,
    'epoch_num' : 20,      # 50 → 20 for faster execution
    'batch_size' : 64,     # 32 → 64 for faster execution
    'topk' : 5,

    'pretrained_gvae' : True, 
    'zdim' : 64,

    # DE parameters
    'INITIAL_F': 0.5,
    'INITIAL_CR': 0.5,
    'F_GAMMA': 0.1,
    'CR_GAMMA': 0.1,
    'F_LOWEST': 0.1,
    'F_UPPER': 0.9,
    'DIMENSION': 64,
    'POPULATION_SIZE': 50,  # smaller population for incremental learning
    'GENERATION': 5,        # 10 → 5 for faster execution
    'step_num' : 1,
    'eta' : 0.2,
    'delta_eta' : 0.2,
    'random_num' : 2000,
}

configs['gvae_path'] = 'gvae/gvae_{}_{}.pth'.format(configs['zdim'],configs['dataset'])

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
    def __init__(self, eval_func, labeled_set, population_size=None, seed=42):
        if population_size is None:
            population_size = configs['POPULATION_SIZE']
        self.eval_func = eval_func
        self.population_size = population_size
        self.labeled_set = labeled_set
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Get border values from labeled set
        latent_data = labeled_set[1]
        self.MAX_VALUES, self.MIN_VALUES = self.get_border_values(latent_data)
        
        # Initialize population from labeled set with noise for diversity
        self.population = []
        for i in range(population_size):
            if i < len(latent_data):
                # Use original top-k latents
                gene = latent_data[i]
            else:
                # For additional population members, add noise to existing genes
                base_idx = i % len(latent_data)
                gene = latent_data[base_idx] + 0.1 * torch.randn_like(latent_data[base_idx])
            individual = Individual(gene=gene, eval_func=eval_func)
            self.population.append(individual)
        
        logging.info(f"Initialize Size: {len(self.population)}")
    
    def get_border_values(self, latent_data):
        MAX_value = [-float("inf") for i in range(configs["DIMENSION"])]
        MIN_value = [float("inf") for i in range(configs["DIMENSION"])]

        for di in range(configs["DIMENSION"]):
            for latent in latent_data:
                MAX_value[di] = max(latent[di],MAX_value[di])
                MIN_value[di] = min(latent[di],MIN_value[di])
        
        return MAX_value, MIN_value

    def mutation_and_crossover(self):
        updated_count = 0
        
        for i in range(len(self.population)):
            indices = list(range(len(self.population)))
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
                    if self.MIN_VALUES[j] <= mutant[j] <= self.MAX_VALUES[j]:
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
        return total_fitness / len(self.population)

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
        
        return self.get_best_individual(top_n=configs['topk'])

def create_evaluation_function(gvae):
    """GVAEとICNNを使用した評価関数を作成"""
    def evaluate(latent_vector):
        with torch.no_grad():
            z = latent_vector.unsqueeze(0).cuda()
            
            # ICNNで直接潜在表現から性能を予測
            pred_acc = (-gvae.icnn(z) + 1.0).squeeze()
            
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
    def __init__(self, labeled_set):
        super().__init__()
        self.dataset = labeled_set

    def __getitem__(self, idx):
        return self.dataset[1][idx], self.dataset[2][idx]

    def __len__(self):
        return len(self.dataset[1])

class CRLSO:
    def __init__(self, configs = configs):
        logging.info("Initializing CRLSO...")
        self.database = NASBench201DataBase('data/nasbench201_with_edge_flops_and_params.json')
        logging.info("Database loaded")
        
        self.dataset = torch.load(configs['nas_bench_201_dataset_path'],weights_only=False)
        logging.info("Dataset loaded")
        
        self.configs = configs

        if configs['pretrained_gvae']:
            logging.info("Loading pretrained GVAE...")
        else:
            logging.info("Training new GVAE...")
            train_gvae()
            
        logging.info(f"Loading GVAE from: {configs['gvae_path']}")
        self.gvae = torch.load(configs['gvae_path'],weights_only=False).cuda()
        logging.info("GVAE loaded and moved to CUDA")
        
        self.labeled_set = self.gvae.labeled_set
        logging.info(f"Initial labeled_set size: {len(self.labeled_set[1])}")
        logging.info("CRLSO initialization complete")

    def main_loop(self, noise = True):
        iteration_count = 0
        while len(self.labeled_set[1]) < (self.configs['evaluate_num']):
            # Tune ICNN every 3 iterations for speed
            if iteration_count % 3 == 0:
                self.tune_icnn()
            iteration_count += 1

            # 部分データセットから上位5つの構造を取得
            values, indices = self.labeled_set[2].topk(self.configs['topk'])
            topk_latents = [self.labeled_set[1][indice] for indice in indices]
            
            # Create DE instance with only top-k latents (similar to crlso.py)
            topk_labeled_set = [
                [self.labeled_set[0][i] for i in indices],
                torch.stack(topk_latents),
                values
            ]
            
            eval_func = create_evaluation_function(self.gvae)
            de = DE(eval_func, topk_labeled_set, population_size=min(self.configs['POPULATION_SIZE'], len(topk_latents)*10))
            
            # Run DE for a few generations
            best_individuals = de.evolve(generations=self.configs['GENERATION'])
            
            # Add new architectures from DE results
            new_arch_count = 0
            for individual in best_individuals:
                latent = individual.gene
                
                with torch.no_grad():
                    arch_tensor = self.gvae.get_tensor(latent.unsqueeze(0).cuda())
                    arch_str = self.gvae.conver_tensor2arch(arch_tensor)

                if arch_str not in set(self.labeled_set[0]):
                    arch_index = self.dataset.str2index(arch_str)

                    if self.configs['dataset'] == 'CIFAR10':
                        acc = self.dataset.cifar10_acc[arch_index][0]
                    elif self.configs['dataset'] == 'CIFAR100':
                        acc = self.dataset.cifar100_acc[arch_index][0]
                    elif self.configs['dataset'] == 'ImageNet':
                        acc = self.dataset.imagenet_acc[arch_index][0]

                    acc = 0.01*acc

                    logging.info('Obtain an new architecture with acc:%f', acc)

                    self.labeled_set[0].append(arch_str)
                    self.labeled_set[1] = torch.cat(
                        [self.labeled_set[1], deepcopy(latent.detach().cpu()).unsqueeze(0)])
                    self.labeled_set[2] = torch.cat(
                        [self.labeled_set[2], torch.tensor([acc]).float()])
                    
                    new_arch_count += 1
                    
                    # Stop if we reach the evaluation limit
                    if len(self.labeled_set[1]) >= self.configs['evaluate_num']:
                        break
            
            # If no new architectures found, add random exploration
            if new_arch_count == 0:
                logging.info("No new architectures found. Adding random exploration...")
                # Add random noise to diversify search
                for _ in range(self.configs['topk']):
                    # Sample random latent from the latent space bounds
                    random_latent = torch.randn(self.configs['DIMENSION'])
                    
                    with torch.no_grad():
                        arch_tensor = self.gvae.get_tensor(random_latent.unsqueeze(0).cuda())
                        arch_str = self.gvae.conver_tensor2arch(arch_tensor)
                    
                    if arch_str not in set(self.labeled_set[0]):
                        arch_index = self.dataset.str2index(arch_str)

                        if self.configs['dataset'] == 'CIFAR10':
                            acc = self.dataset.cifar10_acc[arch_index][0]
                        elif self.configs['dataset'] == 'CIFAR100':
                            acc = self.dataset.cifar100_acc[arch_index][0]
                        elif self.configs['dataset'] == 'ImageNet':
                            acc = self.dataset.imagenet_acc[arch_index][0]

                        acc = 0.01*acc

                        logging.info('Obtain random architecture with acc:%f', acc)

                        self.labeled_set[0].append(arch_str)
                        self.labeled_set[1] = torch.cat(
                            [self.labeled_set[1], random_latent.unsqueeze(0)])
                        self.labeled_set[2] = torch.cat(
                            [self.labeled_set[2], torch.tensor([acc]).float()])
                        
                        break  # Add only one random architecture per iteration
            
            logging.info(f"Current labeled_set size: {len(self.labeled_set[1])}/{self.configs['evaluate_num']}")

    def tune_icnn(self, noise = True):
        mse = nn.MSELoss(reduction = 'mean')

        dataset = ICNN_Dataset(self.labeled_set)
        dataloader = DataLoader(
            dataset, batch_size = self.configs['batch_size'], shuffle = True
        )

        optimizer = torch.optim.Adam(
            self.gvae.icnn.parameters(),
            lr = self.configs['lr'], betas = self.configs['betas'], weight_decay = 1e-5
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max = float(self.configs['epoch_num']), eta_min = 1e-5
        )

        for epoch in tqdm.tqdm(range(self.configs['epoch_num'])):
            objs = AvgrageMeter()
            mse = nn.MSELoss(reduction = 'mean')
            for step, (latents, acc) in enumerate(dataloader):
                n = len(latents)

                if not noise:
                    latents = latents.cuda()
                    acc = acc.cuda()
                else:
                    # add some noise to explore the search space
                    latents = latents.cuda() + 0.05*torch.randn_like(latents.cuda())
                    acc = acc.cuda() + 0.01*torch.randn_like(acc.cuda())

                pred_acc = (-self.gvae.icnn(latents) + 1.0).squeeze()

                loss = mse(acc, pred_acc)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                objs.update(loss.data.item(), n)

                self.gvae.icnn.constraint_weights()

            scheduler.step()

        logging.info('Finetune the icnn, loss_pred:%e', objs.avg)

    def obtain_topk_performance(self, topk = 1):
        values, indices = self.labeled_set[2].topk(self.configs['topk'])
        arch_str = self.labeled_set[0][indices[topk]]
        arch_info = self.database.query_by_str(arch_str)
        return arch_info

if __name__ == '__main__':
    lso = CRLSO()
    lso.main_loop()
    lso.obtain_topk_performance(3)