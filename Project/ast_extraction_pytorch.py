import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from torch_geometric.data import Data
from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score
from torch_geometric.data import Dataset

import clang.cindex
import os
import numpy as np
import pandas as pd


def save_ast_pytorch(node):
   
    node.children = list(node.get_children())

    for child in node.children:
        counter = save_ast_pytorch(child)


def numbering_ast_nodes_pytorch(node, counter=1):
  
    node.identifier = counter
    counter += 1

    node.children = list(node.get_children())
    for child in node.children:
        counter = numbering_ast_nodes_pytorch(child, counter)

    return counter


def generate_edgelist_pytorch(ast_root):
  
    edges = [[],[]]

    def walk_tree_and_add_edges_pytorch(node):
        for child in node.children:
            # edges.append([node.identifier, child.identifier])
            # walk_tree_and_add_edges_pytorch(child)
            edg_0 = (node.identifier)-1
            edg_1 = (child.identifier)-1
            # edges[0].append(node.identifier)
            # edges[1].append(child.identifier)
            edges[0].append(edg_0)
            edges[1].append(edg_1)
            walk_tree_and_add_edges_pytorch(child)

    walk_tree_and_add_edges_pytorch(ast_root)
    return  torch.tensor(edges, dtype=torch.long)

def generate_features_pytorch(ast_root):
  
    features = []

    def walk_tree_and_set_features_pytorch(node):
        out_degree = len(node.children)
        #in_degree = 1
        #degree = out_degree + in_degree
        degree = out_degree 
        node_id = node.identifier
        features.append([node_id, degree])

        for child in node.children:
            walk_tree_and_set_features_pytorch(child)

    walk_tree_and_set_features_pytorch(ast_root)

    features_array = np.asarray(features)
    # nodes_tensor = torch.from_numpy(features_array).float()
    nodes_tensor = torch.tensor(features_array, dtype=torch.float)
    # nodes_tensor = torch.LongTensor(features).unsqueeze(1)
    return nodes_tensor


def clang_process_pytorch(testcase, **kwargs):
 
    parse_list = [
        (testcase.filename, testcase.code)
        
    ]

    # source_file= get_source_file(testcase)

    # Parsing the source code and extracting AST using clang
    index = clang.cindex.Index.create()
    translation_unit = index.parse(
        path=testcase.filename,
        unsaved_files=parse_list,
    )
    ast_root = translation_unit.cursor

    save_ast_pytorch(ast_root)
    numbering_ast_nodes_pytorch(ast_root)

    graphs_embedding = generate_edgelist_pytorch(ast_root)

    nodes_embedding = generate_features_pytorch(ast_root)
   

    y = torch.tensor([testcase.vulnerable], dtype=torch.int64)



    # delete clang objects
    del translation_unit
    del ast_root
    del index

    return Data(x=nodes_embedding, edge_index=graphs_embedding, y=y)