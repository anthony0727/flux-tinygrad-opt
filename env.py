import sys

from tinygrad.shape.symbolic import sym_infer
sys.path.insert(0, '/Users/anthony/tinygrad')

import gymnasium as gym
import networkx as nx
from tinygrad import Tensor, Device, nn
from tinygrad.codegen.kernel import Kernel
from tinygrad.ops import UOps
from tinygrad.device import Compiled
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.search import bufs_from_lin, time_linearizer
from tinygrad.engine.search import actions as tactions
from tinygrad.helpers import getenv
from tinygrad.engine.search import _ensure_buffer_alloc, get_kernel_actions
from extra.models.resnet import ResNet50
from tinygrad import Tensor, Device, dtypes, nn
from tqdm import tqdm

def get_sched_dummy():
    m = nn.Linear(10, 10)
    out = m(Tensor.empty(10, 10).to('cuda'))
    
    return create_schedule([out.lazydata])

def get_sched_resnet():
  mdl = ResNet50()
  optim = (nn.optim.LARS if getenv("LARS") else nn.optim.SGD)(nn.state.get_parameters(mdl))
  BS = getenv("BS", 64)

  # run model twice to get only what changes, these are the kernels of the model
  for _ in range(2):
    out = mdl(Tensor.empty(BS, 3, 224, 224))
    targets = [out.lazydata]
    if getenv("BACKWARD"):
      optim.zero_grad()
      out.sparse_categorical_crossentropy(Tensor.empty(BS, dtype=dtypes.int)).backward()
      targets += [x.lazydata for x in optim.schedule_step()]
    sched = create_schedule(targets)
    # print(f"schedule length {len(sched)}")
  return sched

class CompilerOptEnv(gym.Env):
    def __init__(self, sched, device=None):
        self.sched = [x for x in sched if x.ast.op is UOps.SINK]
        self.num_kernels = len(self.sched)
        self.device: Compiled = Device[Device.DEFAULT] or device
        self.reset_flag = False
        
        self.actions = tactions
        self.action_space = gym.spaces.Discrete(len(self.actions))

        self.curr_idx = 0
        self.cnt = 0
        self.prev_flops = 0
        # self.episode_length = len(self.sched) * len(self.actions)
        # self.initial_flops = 0
        
        print(f"optimizing for {self.device}")

    def count_flops(self):
        tm = time_linearizer(self.lin, self.rawbufs, allow_test_size=False, cnt=10, disable_cache=True)
        ops = (prg:=self.lin.to_program()).op_estimate
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
            return self.lin.ast, -10, term, False, info

        curr_flops = self.count_flops()
        delta_flops = curr_flops - self.prev_flops
        self.prev_flops = curr_flops

        
        if self.curr_idx == len(self.sched):
            term = True
            self.curr_idx = 0

        self.cnt += 1

        return self.lin.ast, delta_flops, term, False, {}


    def reset(self, ker_idx=0):
        self.si = self.sched[ker_idx]
        self.lin = Kernel(self.si.ast, opts=self.device.renderer)
        # self.var_vals = {k: (k.max + k.min) // 2 for k in self.lin.ast.variables()}
        self.rawbufs = _ensure_buffer_alloc(
            bufs_from_lin(self.lin)
        )

        self.reset_flag = True

        return self.lin.ast, {}


    def render(self):
        print(self.lin.ast)


sched = get_sched_resnet()

env = CompilerOptEnv(sched)
env.reset()
fails = 0
sucesses = 0
n = 1

# for i in range(n):
    # print(i)
for i in tqdm(range(100)):
    obs, rew, done, trunc, info = env.step(env.action_space.sample())
    print(obs, rew, info)
    