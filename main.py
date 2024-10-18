#!/usr/bin/env python
# coding: utf-8

import torch
from utils import get_sched_flux
from contextlib import contextmanager
from tinygrad import dtypes
from tinygrad.tensor import Tensor

@contextmanager
def cuda_timer(label="Timer"):
    # Create CUDA events
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)

    # Start the timer
    s.record()

    try:
        # Execute the block of code
        yield
    finally:
        # End the timer
        e.record()

        # Synchronize and calculate the elapsed time
        torch.cuda.synchronize()
        elapsed_time = s.elapsed_time(e)

        # Print the result in milliseconds
        print(f'{label}: {elapsed_time} ms')
        
def _t(shp, dtype=dtypes.bfloat16): return Tensor.empty(shp, dtype=dtype)

def _parse_dtype(dtype): return str(dtype).split('.')[1]

def to_torch(tensor): 
    arr = tensor.numpy()
    return torch.from_numpy(arr).to(
        getattr(torch, _parse_dtype(tensor.dtype))
    ).to('cuda')

def dict_mapper(_dict, func):
    return {k: func(v) for k, v in _dict.items()}


if __name__ == "__main__":
    BS = 2 # batch size
    GUIDANCE = 3.5

    inp = dict(
        img=_t((BS, 1024, 64)),
        img_ids=_t((BS, 1024, 3)),
        txt=_t((BS, 256, 4096)),
        txt_ids=_t((BS, 256, 3)),
        vec=_t((BS, 768)),
    )
    db_inp = dict(
        img=_t((BS, 1024, 3072)),
        txt=_t((BS, 256, 3072)),
        vec=_t((BS, 3072)),
        pe=_t((BS, 1, 1280, 64, 2, 2)),
    )
    
    sb_inp = dict(
        img=_t((BS, 1280, 3072)),
        vec=_t((BS, 3072)),
        pe=_t((BS, 1, 1280, 64, 2, 2)),
    ) 
    # timesteps = get_schedule(
    #   num_steps, 
    #   inp["img"].shape[1], 
    #   shift=(args.name != "flux-schnell")
    # )
    timesteps_inp = [1.0, 0.75, 0.5, 0.25, 0.0]
    vec = _t((1, 768))
    t_vec = _t((1,))

    """
    compiled_original_model = torch.compile(get_original_flow())
    inp_pt = dict_mapper(inp, to_torch)
    def _f():
        compiled_original_model(
            img=inp_pt['img'], img_ids=inp_pt['img_ids'], 
            txt=inp_pt['txt'], txt_ids=inp_pt['txt_ids'],
            timesteps=to_torch(t_vec), 
            y=to_torch(vec),
        )

    # warmup
    with cuda_timer('warmup'):
        for _ in tqdm(range(10)): 
            _f()

    with cuda_timer('inference'):
        _f()
    """

    sched = get_sched_flux(inp, db_inp, sb_inp, t_vec, vec)
    opt(sched, ['RAW', 'HC', 'TC', 'BEAM', 'MCTS'])