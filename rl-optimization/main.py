import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import random
import numpy as np
from env import CompilerOptEnv
from tinygrad import Device
from tinygrad.device import Compiled
from tinygrad import Device
from pprint import pprint
from tinygrad_utils import get_sched_resnet
from gymnasium import spaces
from typing import Any, List, Optional, Union
import torch as th
from collections import deque
from argparse import Namespace
from gymnasium.wrappers import TimeLimit
from models import GraphUOpsEncoder, SrcEncoder, TabularEncoder
from torch_geometric.data import Data
#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 32

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class PPO(nn.Module):
    def __init__(self, n_action):
        super(PPO, self).__init__()
        self.n_action = n_action
        self.data = []
        
        self.graphuops_enc = GraphUOpsEncoder(768, 256)
        self.src_enc = SrcEncoder(384, 256)
        self.tab_enc = TabularEncoder(10, 256)

        # self.fc1   = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256*3,n_action)
        self.fc_v  = nn.Linear(256*3,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def _encode(self, x):
        graph_emb = self.graphuops_enc(x['graph'])
        src_emb = self.src_enc(x['src'])
        tab_emb = self.tab_enc(x['tabular'])
        if graph_emb.dim() == 1:
            graph_emb = graph_emb.unsqueeze(0)
        if src_emb.dim() == 1:
            src_emb = src_emb.unsqueeze(0)
        if tab_emb.dim() == 1:
            tab_emb = tab_emb.unsqueeze(0)
        x = torch.cat([graph_emb, src_emb, tab_emb], dim=1)

        return x
    
    def pi(self, x, softmax_dim = 1):
        x = F.relu(self._encode(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob.squeeze()
    
    def v(self, x):
        x = F.relu(self._encode(x))
        v = self.fc_v(x)
        return v
    

    def _s_lst_stack(self, s_lst):
        g_lst = [x['graph'] for x in s_lst]
        # edge_links = [x['graph'].edge_links for x in s_lst]
        # graph = [s['graph'].nodes.to(DEVICE), edge_index=s['graph'].edge_links.to(DEVICE)) for s in s_lst]
        src = torch.stack([x['src'] for x in s_lst]).to(DEVICE)
        tabular = torch.stack([x['tabular'] for x in s_lst]).to(DEVICE)
        return {
            # 'graph': Namespace(**{
            #     'nodes': nodes,
            #     'edge_links': edge_links
            # }),
            'graph': g_lst,
            'src': src,
            'tabular': tabular
        }

    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = self._s_lst_stack(s_lst), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), self._s_lst_stack(s_prime_lst), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r.to(DEVICE) + gamma * self.v(s_prime) * done_mask.to(DEVICE)
            delta = td_target - self.v(s)
            delta = delta.cpu().detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(DEVICE)

            pi = self.pi(s, softmax_dim=1)
            if pi.shape[0] != a.shape[0]:
                pi = pi.unsqueeze(0)

            pi_a = pi.gather(1,a.to(DEVICE))
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a.to(DEVICE)))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            print('loss', loss.item())
            self.optimizer.step()
    

    # env = gym.make('CartPole-v1') 

score = 0.0
print_interval = 1

env = CompilerOptEnv(
    get_sched_resnet()
)
# env = TimeLimit(env, max_episode_steps=T_horizon)
model = PPO(n_action=env.action_space.n).to(DEVICE)
for n_epi in range(10000):
    s, _ = env.reset()
    done = False
    while not done:
        for t in range(env.action_space.n):
            prob = model.pi(model._s_lst_stack([s]))
            m = Categorical(prob)
            a = m.sample().item()
            s_prime, r, done, truncated, info = env.step(a)
            print(f'epi: {n_epi} kern_idx: {env.curr_idx} step: {t} rew: {r} score: {score} {info}')
            model.put_data((s, a, r/int(sys.maxsize), s_prime, prob.squeeze()[a].item(), done))
            s = s_prime

            score += r
            if done or truncated:
                obs, info = env.reset()
                done = False

        model.train_net()

    # if n_epi%print_interval==0 and n_epi!=0:
    print("# of episode :{}, avg score : {:.1f}".format(n_epi, score))
    score = 0.0