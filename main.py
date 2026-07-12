from utils import get_sched_flux, opt
from tinygrad import dtypes
from tinygrad.tensor import Tensor
        
def _t(shp, dtype=dtypes.bfloat16): return Tensor.empty(shp, dtype=dtype)

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

    sched = get_sched_flux(inp, db_inp, sb_inp, t_vec, vec)
    opt(sched, ['RAW', 'HC', 'TC', 'BEAM', 'MCTS'])
