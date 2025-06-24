"""
アーキテクチャ変換用のユーティリティ関数
NASBench201DataBaseに依存しない軽量版
"""
import torch
import torch.nn as nn
from torch_geometric.data import Data

PRIMITIVES_201 = ['avg_pool_3x3', 'nor_conv_1x1', 'skip_connect', 'nor_conv_3x3', 'none']

def arch2list(arch_str):
    """アーキテクチャ文字列をリストに変換"""
    node_strs = arch_str.split('+')
    genotypes = []
    for i, node_str in enumerate(node_strs):
        inputs = list(filter(lambda x: x != '', node_str.split('|')))
        for xinput in inputs: 
            assert len(xinput.split('~')) == 2, 'invalid input length : {:}'.format(xinput)
        inputs = (xi.split('~') for xi in inputs)
        input_infos = tuple((op, int(IDX)) for (op, IDX) in inputs)
        genotypes.append(input_infos)
    return genotypes

def conver_cell2tensor(arch_list):
    """アーキテクチャリストをテンソルに変換"""
    node_attr = nn.functional.one_hot(torch.tensor([0, 1, 2, 3]), 4)
    edge_attr = nn.functional.one_hot(torch.tensor([0, 1, 2, 3, 4]), 5)
    edge_list = []
    for node_index in range(3):
        node_geno = arch_list[node_index]
        for _, (primitive, input_node) in enumerate(node_geno):
            edge_list.append((input_node, node_index + 1, primitive))   
            
    def edge2architecture(edge_list):
        edge_tensor_list = []
        for edge_index, edge in enumerate(edge_list):
            edge_tensor = torch.cat(
                [node_attr[edge[0]], node_attr[edge[1]],
                edge_attr[PRIMITIVES_201.index(edge[2])]]
                ).unsqueeze(0).float()
            edge_tensor_list.append(edge_tensor)        
        architecture = torch.cat(edge_tensor_list, dim=0)
        return architecture
    
    architecture = edge2architecture(edge_list)
    return architecture  

def conver_cell2graph(arch_list):
    """アーキテクチャリストをグラフ構造に変換"""
    node_attr = nn.functional.one_hot(torch.tensor([0, 1, 2, 3]), 4).float()
    
    source_nodes = torch.LongTensor([0,0,1,0,1,2]).unsqueeze(0)
    target_nodes = torch.LongTensor([1,2,2,3,3,3]).unsqueeze(0)
    
    edge_index = torch.cat([
        torch.cat([source_nodes, target_nodes], dim=1),
        torch.cat([target_nodes, source_nodes], dim=1)], dim=0)
    
    edge_primitives = []
    
    for node_index in range(3):
        node_geno = arch_list[node_index]
        for _, (primitive, input_node) in enumerate(node_geno):
            edge_primitives.append(PRIMITIVES_201.index(primitive))
        
    edge_primitives = torch.LongTensor(edge_primitives)
    cell_tensor = conver_cell2tensor(arch_list).unsqueeze(0)

    edge_attr = nn.functional.one_hot(edge_primitives, len(PRIMITIVES_201)).float()
    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    
    return edge_index, node_attr, edge_attr, cell_tensor