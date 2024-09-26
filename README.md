# flux-tinygrad-opt

## Objective
Try optimizing Blackforest's Flux.1 with tinygrad's optimization.

Flux model class is based on PR [#6334](https://github.com/tinygrad/tinygrad/pull/6334), which can seamlessly load weights of official one, downloaded from huggingface.

Optim. methods include 
* RAW: The raw code.
* HC: Hand-coded; usual optim. such as dead code elimination, loop unrolling, etc. 
* TC: Seems like NVIDIA's tensor cores optim. Only applied when applicable, else raw code.
---
* BEAM: Beam search the kernel.
* MCTS: MCTS the kernel.

[TODO-1] Reverse-engineer optimized IR back to high-level tinygrad code and benchmark with "torch compiled" original Flux, measured by torch profiler or `torch.cuda.Event`.
Also, refer to [this](https://github.com/tinygrad/tinygrad/blob/master/docs/abstractions2.py).

## Result
### Benchmark
**Final result : total 592.40 ms,  60225 GFLOPS**

Cumulative inference times for propagating through all kernels. 

Best time among all methods is chosen for each kernel. e.g. BEAM(594.451) would be chosen for kernel_X for below table.

Flux's flow breaks down to 1423 tinygrad kernels.

<br>

**Another result**; When solely optimized with one method.
| Optimization Method | Time (ms) |
|---------------------|-----------|
| RAW                 | 201654.093|
| HC                  | 7989.641  |
| TC                  | 6317.517  |
| BEAM                | 594.451   |
| MCTS                | 825.108   |

(The raw result significantly differs from the values from Peter and Antonio, check.)


**Caveat**
Couldn't solve TODO-1, therefore, the times(ms) and GFLOPS are measured within tinygrad's method.
[TODO-2] Does tinygrad lookup hardware intrinsics? or actually measure the time? The [runtime dispatcher](https://github.com/tinygrad/tinygrad/blob/4fc5a34fe794036d929217df9939acf9337ae46d/tinygrad/engine/realize.py#L85) returns execution time when called?


## Run
(You should own resource enough for running Flux.1)
```
python main.py

You might also have to add path for tinygrad/extras and tinygrad/examples folders. e.g. in main.py:
import sys; sys.path.append('PATH/TO/tinygrad')
```

## Prerequisites
```
pip install -r requirements.txt
```
```
apt install graphviz git-lfs
```

### Example outputs

**An optimized kernel**
<details>
<summary>Block size-tuned kernel</summary>

```
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
extern "C" __global__ void __launch_bounds__(64) r_2048_4_16_16(float* data0, const float* data1) {
  __shared__ float temp1[64];
  int gidx0 = blockIdx.x; /* 2048 */
  int lidx0 = threadIdx.x; /* 4 */
  int lidx1 = threadIdx.y; /* 16 */
  int alu0 = (lidx0*16);
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 16; ridx0++) {
    float val0 = data1[(gidx0*1024)+(lidx0*256)+lidx1+(ridx0*16)];
    acc0 = (acc0+val0);
  }
  temp1[alu0+lidx1] = acc0;
  __syncthreads();
  if (((bool)(lidx1)!=1)) {
    float acc1 = 0.0f;
    for (int ridx1 = 0; ridx1 < 16; ridx1++) {
      float val1 = temp1[alu0+ridx1];
      acc1 = (acc1+val1);
    }
    data0[(gidx0*4)+lidx0] = acc1;
  }
}

#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
extern "C" __global__ void __launch_bounds__(128) r_16_256_2_64_4(float* data0, const float* data1, const float* data2) {
  __shared__ float temp1[128];
  int gidx0 = blockIdx.x; /* 256 */
  int gidx1 = blockIdx.y; /* 16 */
  int lidx0 = threadIdx.x; /* 2 */
  int lidx1 = threadIdx.y; /* 64 */
  int alu0 = (lidx0*64);
  float val0 = data2[(gidx1*2)+lidx0];
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float val1 = data1[(gidx1*131072)+(gidx0*256)+(lidx0*65536)+(lidx1*4)+ridx0];
    float alu1 = (val1+(val0*(-1.0f)));
    acc0 = (acc0+(alu1*alu1));
  }
  temp1[alu0+lidx1] = acc0;
  __syncthreads();
  if (((bool)(lidx1)!=1)) {
    float acc1 = 0.0f;
    for (int ridx1 = 0; ridx1 < 64; ridx1++) {
      float val2 = temp1[alu0+ridx1];
      acc1 = (acc1+val2);
    }
    data0[(gidx1*512)+gidx0+(lidx0*256)] = acc1;
  }
}
```
</details>

<br>

**Metrics by kernel**

                 kernel 818 r_2560_21504_3072                     [21504, 2560, 1]   [1, 1, 1]    takes 1862.52 ms,    182 GFLOPS -- RAW                                                                                                    
                 kernel 818 r_80_448_8_16_768_4_3_4               [448, 80, 1]       [8, 16, 1]   takes   76.35 ms,   4431 GFLOPS -- HC                                                                                                     
                 kernel 818 r_32_168_2_2_2_2_2_4_192_8_2_2_2_5_4  [168, 32, 1]       [16, 2, 4]   takes    8.21 ms,  41219 GFLOPS -- TC                                                                                                     
                 kernel 818 r_192_40_2_2_2_2_2_192_8_2_2_2_4_7_2  [40, 192, 1]       [8, 2, 2]    takes    5.16 ms,  65530 GFLOPS -- BEAM                                                                                                   
                 kernel 818 r_32_384_2_2_2_2_2_192_8_2_2_2_7_5    [384, 32, 1]       [8, 2, 2]    takes    5.77 ms,  58602 GFLOPS -- MCTS 
e.g. `r_80_448_8_16_768_4_3_4`, `r`: reduce, `80_448`: global size `8_16`: local size. [TODO-3] Clarify the output. Is this OpenCL context? 

`r` is the reduction operations like sum, dot product.

<details>
<summary>Related code</summary>

```python
@functools.cached_property
def name(self) -> str:
  # kernel name (before late upcast)
  name = ("r" if self.reduceop else ("C" if all(x.op in BUFFER_UOPS for x in self.ast.parents) else "E")) + \
               (f"{len(self.ast.src)}_" if len(self.ast.src) > 1 else "_") + \
               colored('_', 'BLACK').join([colored(str(x), c) for x,c in zip(self.full_shape, self.colors())])

  # name the function something unique
  Kernel.kernel_cnt[(function_name := to_function_name(name))] += 1
  suffix = f"{'n'+str(Kernel.kernel_cnt[function_name]-1)}" if Kernel.kernel_cnt[function_name] > 1 else ""
return name+colored(suffix, 'BLACK')
```
```python
print(f"                 kernel {i:2d} {lin.name+' '*(37-ansilen(lin.name))} {str(prg.global_size):18s} {str(prg.local_size):12s} takes {tm*1000:7.2f} ms, {gflops:6.0f} GFLOPS -- {colored(nm, 'green') if lin is sorted_choices[0][2] else nm}")
```
```python
 @property
  def global_dims(self) -> int: return self.first_reduce-self.local_dims

  # there's eight chunks of the shape
  # blue   -- global dims
  # cyan   -- local dims (warp ones first)
  #  *** self.first_reduce
  # green  -- reduce-local dims
  # white  -- reduce-late upcasted dim (self.upcast_in_mid_reduce_axes)
  # red    -- reduce loops
  #  *** self.upcasted
  # purple -- reduce upcasted
  # yellow -- normal upcasted dimensions
  def colors(self) -> List[str]:
    # first non local non reduce dims are global (blue)
    colors = ["blue"] * self.global_dims if not self.dont_use_locals else ["BLUE"] * self.global_dims
    # after global are local_dims; warp ones used in tensor cores must be closest to first_reduce (cyan)
    colors += ["cyan"] * self.local_dims
    # between first_reduce and first_reduce + group_for_reduces, they are either upcast mid reduce (white), or late upcasted (green)
    colors += ["white" if i in self.upcast_in_mid_reduce_axes else "green" for i in range(self.first_reduce, self.first_reduce + self.group_for_reduces)]  # noqa: E501
    # between first_reduce + group_for_reduces and upcasted, they are reduce (red)
    colors += ["red"] * (self.first_upcast - (self.first_reduce + self.group_for_reduces))
    # upcasted dimensions are reduce (magenta) or normal (yellow)
    colors += ["magenta" if self.full_shape[i] != self.sts[0].shape[i] else "yellow" for i in range(self.first_upcast, self.shape_len)]
    assert len(colors) == self.shape_len, "colors size mismatch"
    return colors
```
</details>

## Background

Supported ops

https://github.com/tinygrad/tinygrad/blob/d1bae42d3527e265b2772e39e563cdbaea34592e/tinygrad/ops.py
