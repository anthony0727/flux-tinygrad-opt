"""PREREQUISITES
apt install graphviz git-lfs
pip install pillow pydot networkx sentencepiece torch transformers diffusers accelerate
pip install git+https://github.com/black-forest-labs/flux.git

You might also have to add path for tinygrad/extras and tinygrad/examples folders. e.g.:
# sys.path.append('/home/anthony/tinygrad')
"""

from typing import List, Tuple
from argparse import Namespace
from tqdm import tqdm
from collections import defaultdict
from pprint import pprint

import torch

from tinygrad import Tensor, Device, nn
from tinygrad.codegen.kernel import Kernel
from tinygrad.ops import UOps
from tinygrad.device import Compiled
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.search import time_linearizer, bufs_from_lin, beam_search
from tinygrad.helpers import DEBUG, ansilen, getenv, colored
from tinygrad.shape.symbolic import sym_infer
from tinygrad import Tensor, nn
from tinygrad import dtypes

from extra.mcts_search import mcts_search
from examples.flux1 import (
    load_flow_model
)

# from diffusers import FluxPipeline
# model_id = "black-forest-labs/FLUX.1-schnell" #you can also use `black-forest-labs/FLUX.1-dev`
# pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
# pipe.transformer

def get_original_flow():
    from transformers import pipeline
    from flux.util import (
        configs,
        embed_watermark,
        load_ae,
        load_clip,
        load_flow_model,
        load_t5,
    )

    NSFW_THRESHOLD = 0.85

    def get_models(name: str, device: torch.device, offload: bool, is_schnell: bool):
        t5 = load_t5(device, max_length=256 if is_schnell else 512)
        clip = load_clip(device)
        model = load_flow_model(name, device="cpu" if offload else device)
        ae = load_ae(name, device="cpu" if offload else device)
        nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)
        return model, ae, t5, clip, nsfw_classifier

    return get_models("flux-schnell", torch.device("cuda"), False, True)[0]


def get_sched_dummy():
  out = nn.Linear(10, 10)(Tensor.empty(10, 10))
  return create_schedule([out.lazydata])


def get_sched_flux(inp, db_inp, sb_inp, t_vec, vec):
    args = Namespace(**{
        'name': 'flux-schnell', 'width': 512, 'height': 512, 'seed': None, 
        'prompt': 'a horse sized cat eating a bagel', 'num_steps': None, 
        'guidance': 3.5, 'offload': False, 'output_dir': 'output'
    })

    num_steps = 4 if args.name == "flux-schnell" else 50
    
    args.seed = Tensor._seed
    print(f"Generating with seed {args.seed}:\n{args.prompt}")

    # why is __getitem__ in the schedule? is it related to warp, block size?
    def _remove_getitem(sched):
        return [x for x in sched if 'getitem' not in str(x.metadata)]
  
    model = load_flow_model(args.name)
    def only_db(db_inp):
        db = model.double_blocks[0]
        img, txt = db(img=db_inp['img'], txt=db_inp['txt'], vec=db_inp['vec'], pe=db_inp['pe'])
        sched = create_schedule([img.lazydata, txt.lazydata])

        return sched
  
    def only_sb():
        db = model.single_blocks[0]
        img = db(img=sb_inp['img'], vec=sb_inp['vec'], pe=sb_inp['pe'])
        sched = create_schedule([img.lazydata])

        return sched

    def only_flow():
        # def __call__(self, img:Tensor, img_ids:Tensor, txt:Tensor, txt_ids:Tensor, timesteps:Tensor, y:Tensor, guidance:Tensor | None = None) -> Tensor:
        img = model(
            img=inp['img'], img_ids=inp['img_ids'], 
            txt=inp['txt'], txt_ids=inp['txt_ids'],
            timesteps=t_vec, 
            y=vec, 
            guidance=args.guidance
        )
        sched = create_schedule([img.lazydata])
        # sched = _remove_getitem(sched)

        return sched

    sched = only_flow()
#   print(f"pruned schedule length {len(sched)}")

    return sched
# if getenv("HALF", 1):
#   dtypes.default_float = dtypes.half

BEAM = 20
MCTS = 150
def opt(sched, methods: List[str] = ['RAW'], device: Compiled = Device[Device.DEFAULT]):
    res_tm, res_gflop = defaultdict(float), defaultdict(float)
    if getenv("BACKWARD"): Tensor.training = True
    print(f"optimizing for {device}")

    sched = globals()[f"get_sched_{getenv('MODEL', 'resnet')}"]()
  # sched = get_sched_flux_flow_model()
    sched = [x for x in sched if x.ast.op is UOps.SINK]

    # focus on one kernel
    if getenv("KERNEL", -1) >= 0: sched = sched[getenv("KERNEL", -1):getenv("KERNEL", -1)+1]

    # work with the schedule
    total_tm = 0
    running_gflops = 0
    usage = {}
    for i,si in enumerate(sched):
        if DEBUG >= 3: print(si.ast)

        rawbufs = bufs_from_lin(Kernel(si.ast))

        # "linearize" the op into uops in different ways
        lins: List[Tuple[Kernel, str]] = []

        # raw
        if 'RAW' in methods:
            lin = Kernel(si.ast, opts=device.renderer)
            # lin.hand_coded_optimizations()
            lins.append((lin, "RAW"))

        # always try hand coded opt
        if 'HC' in methods:
            lin = Kernel(si.ast, opts=device.renderer)
            lin.hand_coded_optimizations()
            lins.append((lin, "HC"))

        # maybe try tensor cores
        if 'TC' in methods:
            lin = Kernel(si.ast, opts=device.renderer)
            if lin.apply_tensor_cores():
                lins.append((lin, "TC"))
            else:
                lins.append((lin, "TC")) # if not applicable, just raw code

        # try a beam search
        if "BEAM" in methods:
            lin = Kernel(si.ast, opts=device.renderer)
            lin = beam_search(lin, rawbufs, BEAM, bool(getenv("BEAM_ESTIMATE", 1)))
            lins.append((lin, "BEAM"))

        # try MCTS
        if "MCTS" in methods:
            lin = Kernel(si.ast, opts=device.renderer)
            lin = mcts_search(lin, rawbufs, MCTS)
            lins.append((lin, "MCTS"))

        # benchmark the programs
        choices = []
        for lin, nm in lins:
            tm = time_linearizer(lin, rawbufs, allow_test_size=False, cnt=10, disable_cache=True)
            ops = (prg:=lin.to_program()).op_estimate
            gflops = sym_infer(ops, {k:k.min for k in lin.ast.variables()})*1e-9/tm
            res_tm[nm] += (tm * 1000)
            res_gflop[nm] += (gflops * tm)
            choices.append((tm, gflops, lin, prg, nm))

        pprint(res_tm)
        pprint(res_gflop)
        sorted_choices = sorted(choices, key=lambda x: x[0])
        if DEBUG >= 1: # print all kernels
            for tm, gflops, lin, prg, nm in choices:
                print(f"                 kernel {i:2d} {lin.name+' '*(37-ansilen(lin.name))} {str(prg.global_size):18s} {str(prg.local_size):12s} takes {tm*1000:7.2f} ms, {gflops:6.0f} GFLOPS -- {colored(nm, 'green') if lin is sorted_choices[0][2] else nm}")

        tm, gflops, lin, prg, nm = sorted_choices[0]
        if getenv("SRC"):
            print(si.ast)
            print(lin.applied_opts)
            print(lin.to_program().src)
        total_tm += tm
        running_gflops += gflops * tm
        if (key := str([str(m) for m in si.metadata] if si.metadata is not None else None)) not in usage: usage[key] = (0, 0)
        usage[key] = (usage[key][0] + tm, usage[key][1] + 1)
        print(f"*** {total_tm*1000:7.2f} ms : kernel {i:2d} {lin.name+' '*(37-ansilen(lin.name))} {str(prg.global_size):18s} {str(prg.local_size):12s} takes {tm*1000:7.2f} ms, {gflops:6.0f} GFLOPS {[str(m) for m in si.metadata] if si.metadata is not None else ''}")
    print(f"******* total {total_tm*1000:.2f} ms, {running_gflops/(total_tm+1e-8):6.0f} GFLOPS")
    print("usage:")
    for k in sorted(usage, key=lambda x: -usage[x][0])[:10]:
        print(f"{usage[k][0]*1000:.2f} ms: {k} ({usage[k][1]} times)")
        print('total time', res_tm)
        print('total gflop', res_gflop)
