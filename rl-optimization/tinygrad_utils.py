import sys
sys.path.insert(0, '/Users/anthony/tinygrad')


from tinygrad import Tensor, nn
from tinygrad.engine.schedule import create_schedule
from tinygrad.helpers import getenv
from extra.models.resnet import ResNet50
from tinygrad import Tensor, dtypes, nn
from tinygrad.dtype import DTYPES_DICT
dtype_strlist = list(DTYPES_DICT.values())
dtype_strlist += [f'PtrDType({x})' for x in dtype_strlist]


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


