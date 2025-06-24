import torch
import torch.nn as nn
import numpy as np
import random
import operator
from models import ArchGVAE, GNN_Predictor
from torch_geometric.loader import DataLoader
import logging
import sys

# ログ設定
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

# 設定
configs = {
    # 差分進化パラメータ
    'INITIAL_F': 0.5,
    "INITIAL_CR": 0.5,
    "F_GAMMA": 0.1,
    "CR_GAMMA": 0.1,
    "F_LOWEST": 0.1,
    "F_UPPER" : 0.9,
    "DIMENSION":64,  # 潜在表現の次元数
    "POPULATION_SIZE": 50,
    "GENERATION": 100,

    # 潜在表現の探索範囲
    "MIN_VALUE":  -3.0,
    "MAX_VALUE":  3.0,
    'dataset': 'ImageNet',  # 'CIFAR10', 'CIFAR100', 'ImageNet'
    'gvae_path': 'gvae/gvae_64_ImageNet.pth',
    'predictor_path': 'semi_predictor/semi_predictor_ImageNet.pth',
    'latent_path': 'latent_representations_64dim.pth',
    'seed': 42
}

class Individual:
    def __init__(self, gene=None, eval_func=None):
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
        
        # シード設定
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 初期個体群の生成
        self.population = []
        for _ in range(population_size):
            # 潜在表現をランダムに生成
            gene = torch.FloatTensor(configs['DIMENSION']).uniform_(configs['MIN_VALUE'], configs['MAX_VALUE'])
            individual = Individual(gene=gene, eval_func=eval_func)
            self.population.append(individual)
        
        logging.info(f"初期個体群を生成しました。サイズ: {population_size}")

    def mutation_and_crossover(self):
        updated_count = 0
        
        for i in range(self.population_size):
            # 突然変異のためのインデックス選択
            indices = list(range(self.population_size))
            indices.remove(i)
            np.random.shuffle(indices)
            
            # jDE アルゴリズム：パラメータ F と CR の更新
            if np.random.rand() < configs['F_GAMMA']:
                self.population[i].F = configs['F_LOWEST'] + np.random.rand() * (configs['F_UPPER'] - configs['F_LOWEST'])
            
            if np.random.rand() < configs['CR_GAMMA']:
                self.population[i].CR = np.random.rand()
            
            # 突然変異ベクトルの生成
            mutant = (self.population[indices[0]].gene + 
                     self.population[i].F * (self.population[indices[1]].gene - self.population[indices[2]].gene))
            
            # 交叉
            trial = torch.zeros_like(self.population[i].gene)
            jrand = np.random.randint(configs['DIMENSION'])
            
            for j in range(configs['DIMENSION']):
                if np.random.rand() < self.population[i].CR or j == jrand:
                    # 境界制約
                    if configs['MIN_VALUE'] <= mutant[j] <= configs['MAX_VALUE']:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = self.population[i].gene[j]
                else:
                    trial[j] = self.population[i].gene[j]
            
            # 試行個体の評価
            trial_individual = Individual(gene=trial, eval_func=self.eval_func)
            trial_individual.F = self.population[i].F
            trial_individual.CR = self.population[i].CR
            
            # 選択
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
        logging.info("差分進化を開始します...")
        
        for generation in range(generations):
            updated_count = self.mutation_and_crossover()
            avg_fitness = self.get_average_fitness()
            best_individual = self.get_best_individual(1)[0]
            
            logging.info(f"世代 {generation+1}: 更新個体数={updated_count}, "
                        f"平均適応度={avg_fitness:.6f}, 最良適応度={best_individual.fitness:.6f}")
        
        return self.get_best_individual(top_n=5)

def create_evaluation_function(gvae, predictor):
    """GVAEとpredictorを使用した評価関数を作成"""
    def evaluate(latent_vector):
        with torch.no_grad():
            # 潜在表現をテンソルに変換
            z = latent_vector.unsqueeze(0).cuda()
            
            # GVAEで潜在表現をアーキテクチャテンソルに変換
            arch_tensor = gvae.get_tensor(z)
            
            # predictorで性能を予測
            pred_acc = predictor(arch_tensor)
            
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
    # モデルの読み込み
    logging.info("モデルを読み込んでいます...")
    gvae = torch.load(configs['gvae_path'], weights_only=False).cuda()
    predictor = torch.load(configs['predictor_path'], weights_only=False).cuda()
    
    gvae.eval()
    predictor.eval()
    
    # 評価関数の作成
    eval_func = create_evaluation_function(gvae, predictor)
    
    # 差分進化の実行
    de = DE(eval_func, population_size=configs['POPULATION_SIZE'], seed=configs['seed'])
    best_individuals = de.evolve(generations=configs['GENERATION'])
    
    # 結果の表示
    logging.info("=== 最適化結果 ===")
    for i, individual in enumerate(best_individuals):
        arch_str = convert_latent_to_architecture(gvae, individual.gene)
        logging.info(f"第{i+1}位: 予測性能={individual.fitness:.6f}, アーキテクチャ={arch_str}")
    
    # 結果の保存
    best_latents = torch.stack([ind.gene for ind in best_individuals])
    torch.save(best_latents, f'de_best_latents_{configs["dataset"]}.pth')
    
    best_fitnesses = [ind.fitness for ind in best_individuals]
    torch.save(best_fitnesses, f'de_best_fitnesses_{configs["dataset"]}.pth')
    
    logging.info("結果を保存しました。")

if __name__ == "__main__":
    main()