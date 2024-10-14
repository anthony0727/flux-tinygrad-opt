
import sys
sys.path.insert(0, '/Users/anthony/tinygrad')


import numpy as np
import gymnasium as gym

import torch as th
import torch_geometric as pyg
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.data import Batch, Data
from transformers import AutoTokenizer, AutoModel


import tinygrad as td

from torch import nn
from torch_geometric.nn.conv import GATv2Conv

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Load the model

class GraphUOpsEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv1 = GATv2Conv(input_dim, hidden_dim)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim)
    
    def forward(self, g_lst):
        data_list = [Data(x=g.nodes, edge_index=g.edge_links) for g in g_lst] # contiguity
        batch = Batch.from_data_list(data_list)

        l = []
        for idx in range(batch.num_graphs):
            g = batch[idx]
            x = self.conv1(g.x, g.edge_index)
            x = self.conv2(x  , g.edge_index)
            x = self.conv3(x  , g.edge_index)
            x = x.mean(dim=0).squeeze()
            l.append(x)
        
        return th.stack(l).squeeze()
    
class SrcEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # x = x.mean(dim=0)

        return x
    
class TabularEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # x = x.mean(dim=0)

        return x

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == "src":
                extractors[key] = SrcEncoder(subspace.shape[0], 128)
                
            elif key == "graph":
                # Run through a simple MLP
                extractors[key] = GraphUOpsEncoder(subspace.node_space.shape[0], 128)
                
            elif key == "tabular":
                # Run through a simple MLP
                extractors[key] = TabularEncoder(subspace.shape[0], 128)
                

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = 384

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            if key == "graph":
                x = extractor(observations[key].nodes, observations[key].edge_links)
            else:
                x = extractor(observations[key])
            encoded_tensor_list.append(x)
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        # print([i.shape for i in encoded_tensor_list])
        return th.cat(encoded_tensor_list)