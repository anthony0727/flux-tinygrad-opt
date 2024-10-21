import sys
sys.path.insert(0, '/Users/anthony/tinygrad')

from typing import List

import numpy as np
import gymnasium as gym
from gymnasium import spaces # Graph, Box, Sequence, MultiDiscrete, Dict
import networkx as nx

import torch as th
import torch_geometric as pyg
from torch_geometric.utils.convert import from_networkx
# from transformers import T5Tokenizer, T5PreTrainedModel
from sentence_transformers import SentenceTransformer

from tinygrad import Tensor, Device, nn
from tinygrad.codegen.kernel import Kernel
from tinygrad.ops import UOps, Uop
from tinygrad.device import Compiled
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.search import bufs_from_lin, time_linearizer
from tinygrad.engine.search import actions as tactions
from tinygrad.helpers import getenv
from tinygrad.engine.search import _ensure_buffer_alloc
from extra.models.resnet import ResNet50
from tinygrad import Tensor, Device, dtypes, nn
from tinygrad.ops import UOps, sym_infer
from tinygrad.viz.serve import uops_colors
from tinygrad.helpers import word_wrap


def graph_uops(uops:List[UOp]):
  G = nx.DiGraph()
  for u in uops:
      if u.op in {UOps.ENDRANGE, UOps.ENDIF}: continue
      G.add_node(
          uops.index(u), label=f"{str(u.op)[5:]}{(' '+word_wrap(str(u.arg).replace(':', ''))) if u.arg is not None else ''}\n{str(u.dtype)}",
          style="filled", fillcolor=uops_colors.get(u.op, "#ffffff")
      )
      for v in u.src: G.add_edge(uops.index(v), uops.index(u))

  return G


class CompilerOptEnv(gym.Env):
    def __init__(self, sched, device=None):
        self.sched = [x for x in sched if x.ast.op is UOps.SINK]
        self.num_kernels = len(self.sched)
        self.device: Compiled = Device[Device.DEFAULT] or device
        self.reset_flag = False


        # _graph_space.shape = (384*2,)
        self.observation_space = spaces.Dict({
            'src': spaces.Box(low=-1e6, high=1e6, shape=(384,)),
            # 'graph_node': spaces.Sequence(spaces.Box(low=-1e6, high=1e6, shape=(384*2,))),
            # 'graph_edge': spaces.Sequence(spaces.Box(low=-1e6, high=1e6, shape=(2,))),
            'graph': spaces.Graph(
                node_space=spaces.Box(low=-1e6, high=1e6, shape=(384*2,)),
                edge_space=None
            ),
            'tabular': spaces.Box(low=-1e6, high=1e6, shape=(10,))
        })
        
        self.actions = tactions
        self.action_space = gym.spaces.Discrete(len(self.actions))

        self.curr_idx = 0
        self.cnt = 0
        self.prev_flops = 0
        # self.episode_length = len(self.sched) * len(self.actions)
        # self.initial_flops = 0
        
        print(f"optimizing for {self.device}")

        self.text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to('cuda')

    def encode_src(self, src):
        return self.text_encoder.encode(src, convert_to_tensor=True).float()
    
    def encode_graph(self, graph):
        # 일단 torch로
        pyg_data = from_networkx(graph)
        
        inst_lst, dtype_lst = [], []
        for i in pyg_data.label:
            a, b = i.split('\n')
            inst_lst.append(a)
            dtype_lst.append(b)
        
        # this requires no grad
        inst_enc = self.text_encoder.encode(inst_lst, convert_to_tensor=True).float()
        dtype_enc = self.text_encoder.encode(dtype_lst, convert_to_tensor=True).float()
        pyg_data.x = th.cat(
            [inst_enc, dtype_enc], # if changed, change graph space too
            axis=1
        )
        
        return spaces.graph.GraphInstance(
            nodes=pyg_data.x.to('cuda'),
            edges=None,
            edge_links=pyg_data.edge_index.to('cuda')
        )

    def encode_hardware(
            self,
            mem_estimate:int,
            global_size:List[int],
            local_size:List[int],
            vars:List[int],
            globals:List[int]
        ):
        def _normalize(l):
            if len(l) > 0:
                return [x / 1024. for x in l]
            
        mem_estimate = mem_estimate / 8e10 # 80GB
        global_size = _normalize(global_size)
        local_size = _normalize(local_size)
        # vars = _normalize(vars)
        globals = _normalize(globals)
        
        return th.from_numpy(np.array([mem_estimate] + global_size + local_size + globals, dtype=np.float32)).to('cuda')  #+ vars

    def featurize(self, prg):
        # 'tabular': prg.mem_estimate + prg.global_size + prg.local_size + prg.vars + prg.globals,
            # 'mem_estimate': prg.mem_estimate,
            # 'global_size': prg.global_size,
            # 'local_size': prg.local_size,
            # 'outs': prg.outs,
            # 'vars': prg.vars,
            # 'globals': prg.globals,
            # mem_estimate=244096396, global_size=[12544, 64, 64], local_size=[1, 1, 1], vars=[], globals=[0, 1, 2], outs=[0]
        # }
        return {
            'src': self.encode_src(prg.src),
            'graph': self.encode_graph(graph_uops(prg.uops)),
            'tabular': self.encode_hardware(
                prg.mem_estimate, 
                prg.global_size, prg.local_size, 
                prg.vars, prg.globals
            )
            # mem_estimate=244096396, global_size=[12544, 64, 64], local_size=[1, 1, 1], vars=[], globals=[0, 1, 2], outs=[0]
        }

    def count_flops(self, prg):
        tm = time_linearizer(self.lin, self.rawbufs, allow_test_size=False, cnt=10, disable_cache=True)
        # ops = (prg:=self.lin.to_program()).op_estimate
        ops = prg.op_estimate
        flops = sym_infer(ops, {k:k.min for k in self.lin.ast.variables()})/tm

        return flops
      

    def step(self, action):
        if not self.reset_flag:
            raise Exception("reset before stepping")
        
        info = {}
        if self.cnt >= len(tactions): # budget as much as num actions
            self.cnt = 0
            self.curr_idx += 1
            obs, info = self.reset(self.curr_idx) # add info

            return obs, 0, False, False, info
        
        term = False
        try:
            self.lin.apply_opt(self.actions[action])
        except Exception as e:
            info['error'] = str(e)
            return self.featurize(self.lin.to_program()), 0, term, False, info

        prg = self.lin.to_program()
        try:
            curr_flops = self.count_flops(prg)
        except Exception as e:
            info['error'] = '[count_flops failed]' + str(e)
            curr_flops = 0

        delta_flops = curr_flops - self.prev_flops
        self.prev_flops = curr_flops

        
        if self.curr_idx == len(self.sched):
            term = True
            self.curr_idx = 0

        self.cnt += 1

        return self.featurize(prg), delta_flops, term, False, {}


    def reset(self, ker_idx=0, seed=42, **kwargs):
        self.si = self.sched[ker_idx]
        self.lin = Kernel(self.si.ast, opts=self.device.renderer)
        # self.var_vals = {k: (k.max + k.min) // 2 for k in self.lin.ast.variables()}
        self.rawbufs = _ensure_buffer_alloc(bufs_from_lin(self.lin))

        self.reset_flag = True

        return self.featurize(self.lin.to_program()), {}


    def render(self):
        print(self.lin.ast)


# sched = get_sched_resnet()
# env = CompilerOptEnv(sched)
# obs, info = env.reset()
# prg=env.lin.to_program()
# graph = graph_uops(prg.uops)