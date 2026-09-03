# Phase 1 measurements: AMD Radeon 780M (RDNA3, gfx1103)

> Full sweep, 256 MiB buffers. The machine was lightly in use (14.7% CPU), so
> these are a floor, not a ceiling. `bench/nightly.ps1` refreshes them daily and
> writes to `bench/runs/`, which is gitignored: it records the host name and a
> process list alongside the numbers.

Machine: Ryzen 7 7840HS + Radeon 780M (12 CU, wave32/64), 32 GB DDR5-5600
dual-channel (~89.6 GB/s theoretical, shared with the CPU). Vulkan 1.4.315,
AMD Windows driver 0x800161. Async-compute queue family 1.

## 1. Memory bandwidth by memory kind

| kind | Vulkan memory type | copy GB/s | read GB/s |
|---|---|---|---|
| `shared` (HOST_VISIBLE\|COHERENT, zero-copy) | type 1, heap 1 | 54.72 | 68.70 |
| `cached` (HOST_VISIBLE\|COHERENT\|CACHED) | type 3, heap 1 | 55.37 | 67.18 |
| `device` (DEVICE_LOCAL, needs staging) | type 0, heap 0 | 68.18 | 79.75 |
| `bar` (DEVICE_LOCAL\|HOST_VISIBLE, 256 MiB) | type 2, heap 2 | 58.94 | 68.15 |

Peak read is 79.75 GB/s against 89.6 GB/s theoretical, i.e. **89% of peak**.

## 2. Staging upload: the cost zero-copy avoids

`host -> device-local` via `vkCmdCopyBuffer`: **23.16 GB/s**.

## 3. What 1 and 2 actually decide

The naive hypothesis was "unified memory means never copy". The measurement says
something more precise, and more useful:

- Host-visible memory runs at **86% of device-local read bandwidth** (68.70 vs 79.75).
- A staging upload costs **23.16 GB/s one-way** on top of that.

So the right policy is per-tensor, chosen by role:

| tensor role | touched by CPU | reused on GPU | best kind | why |
|---|---|---|---|---|
| input batch, labels | every step | once | `shared` | staging costs more than the 14% bandwidth gap |
| weights, optimizer state | only at init | every step | `device` | 17% faster, staging amortizes over the run |
| loss, gradient norms | read back each step | once | `cached` | write-combined memory is brutal for CPU reads |

This is what `Device.KINDS` already exposes, so the allocator can select by role.
**The framework should not have one global answer to "copy or not".**

## 4. Dispatch overhead

| | time |
|---|---|
| per dispatch, batched into one command buffer | **0.62 us** |
| per submit + fence wait | **93.61 us** |

That is a factor of **151**. A training step of ~50 kernels costs ~31 us of launch
overhead when recorded into one command buffer, versus ~4.7 ms if each is
submitted separately, which would dominate everything else at small model sizes.
Record-once-and-replay is not an optimization here. It is the difference between
viable and not. Section 13 confirms this on the real workload.

(With validation layers on, per-submit rises to ~165 us. Benchmark without them.)

## 5. Compute ceilings

| | measured | theoretical | ratio |
|---|---|---|---|
| fp32 vector FMA | 4.94 TFLOPS | ~8.2 (dual-issue) | 60% |
| f16 WMMA (cooperative matrix) | **13.75 TFLOPS** | ~16.6 | **83%** |

WMMA reaches 83% of theoretical and is **2.8x** fp32. The matrix cores are real,
reachable from Vulkan on a consumer AMD iGPU under Windows, and close to their
paper number.

The earlier `--quick` run reported 4.56 TFLOPS for the same kernel. The whole
3x difference was clock ramp: short kernels never reach boost. Worth remembering
before drawing conclusions from any brief GPU benchmark.

## 6. Ridge point

**~200 FLOP/byte.** Below that arithmetic intensity a kernel is memory-bound.

For scale: an unfused elementwise op is ~0.1 FLOP/byte. A fused bias+GELU chain
is maybe 2. Even a 64x64-tiled fp16 matmul reaches only ~32. **Essentially
nothing in a small training workload is compute-bound on this machine.**

That is the finding the rest of the project hangs on. The compiler's job is
minimizing bytes moved; tile sizes and instruction selection stay second-order
until byte traffic is already minimal.

---

# Phase 2 measurements: WMMA matmul

> From the same full sweep unless noted. Machine lightly in use.

## 7. Correctness first

All against a numpy f32 reference, relative error at 1e-6:

- `C = A @ B` across five shapes and tile configurations
- `C = A^T @ B` (this is `dW = X^T dY`)
- `C = A @ B^T` (this is `dX = dY W^T`)
- f32 accumulation proven **exact** summing K=1024 unit products, which pure f16
  accumulation could not do

Both transposes are layout flips on the cooperative-matrix load
(`gl_CooperativeMatrixLayoutColumnMajor`), not a materialised transpose. At a
ridge point of ~200 FLOP/byte, writing a transposed copy would cost more than
the matmul consuming it.

**The backward pass of a linear layer now runs on the matrix cores.** That is
the piece no existing Vulkan project has.

## 8. Matmul throughput, and the comparison that matters

| size | GPU TFLOPS | % of WMMA peak | CPU TFLOPS (numpy/OpenBLAS, 8 cores) | GPU speedup |
|---|---|---|---|---|
| 256 | 0.40 | 2.9% | 0.18 | 2.2x |
| 512 | 2.19 | 16.0% | 0.49 | 4.5x |
| 1024 | 3.11 | 22.7% | 0.38 | 8.2x |
| 2048 | 3.24 | 23.6% | 0.47 | 6.9x |
| 4096 | **3.63** | **26.4%** | 0.62 | 5.8x |

**The iGPU beats the CPU sharing its die and its DRAM by 4 to 8x.** Below ~256 it is not worth the trouble. That crossover is the
practical answer to "can you train on hardware people already own".

## 9. Negative result: LDS staging mostly does not help

The textbook CUDA optimisation is to stage tiles through shared memory. The
direct-from-global kernel re-reads B once per subgroup, so staging should cut B
traffic by a factor of `sg`. Measured at 512x512x512:

| variant | best config | TFLOPS |
|---|---|---|
| direct from global | sg4 wm4 wn4 | 1.445 |
| LDS staged | sg4 wm2 wn4 | 1.303 |

LDS was **slower** at matched effort. The reason is that all subgroups in a
workgroup read the same B columns at the same time, so the L2 already serves
the reuse that LDS was going to provide, and the staging adds two barriers and
an LDS round trip per K-block.

It only turns positive once tiles get wide (at 512 the best overall config after
widening the search was `sg8 wm2 wn4 lds1` at 2.19 TFLOPS). So the honest
statement is: **LDS staging is not the lever it is on discrete NVIDIA hardware,
and a tuner that assumed it would be would have picked wrong.**

## 10. What is actually the lever: tile width

Global traffic for a tiled matmul is `M*K*(N/BN) + K*N*(M/BM)`. Only BM and BN
reduce it. Widening the search space from `wn <= 4` to `wn <= 8` moved
512x512x512 from 1.44 to 2.19 TFLOPS (+52%) with no other change. At 2048 the
winner is the widest tile available, 256x128.

## 11. Occupancy pruning

`VK_KHR_pipeline_executable_properties` reports `numUsedVgprs`,
`ldsUsageSizeInBytes` and LDS budget per compiled pipeline. Combined with
`VK_AMD_shader_core_properties` (12 CUs, 1024 VGPRs/SIMD, max 16 waves/SIMD)
that gives resident waves per SIMD analytically, before running anything.

Currently prunes 2 of 84 configs at a conservative 4-waves-per-SIMD floor. The
mechanism works and costs one compile per config, but the threshold is tuned
too loosely to save much yet. Raising it is a tuning question for the full run,
not a design change.

One caveat worth recording: the driver reports `subgroupSize: 64` in pipeline
executable properties even when `requiredSubgroupSize=32` was granted and the
shader genuinely observes 32. That field cannot be trusted; the runtime check
in `test_runtime.py` can.

---

# Phase 3 measurements: autograd and a full training step

## 12. Gradient correctness

A 2-layer GELU MLP, checked three independent ways:

| check | result |
|---|---|
| loss vs numpy reference | 1.694581 vs 1.694635 |
| dW head vs numpy | 4.45e-04 |
| db head vs numpy | 1.56e-04 |
| dW fc0 vs numpy | 4.43e-04 |
| db fc0 vs numpy | 4.80e-04 |
| finite differences, 4 random entries | worst 2.1e-02 |
| AdamW overfitting one fixed batch | loss 2.3665 -> 0.0033 |

The residual 1e-4 is f16 activations feeding the dW matmuls, which is expected
and correct. A wrong derivative shows up orders of magnitude above this.

## 13. The launch-overhead result, confirmed on a real workload

Phase 1 predicted that a training step submitted kernel-by-kernel would be
dominated by launch overhead, and that recording the step once into a single
command buffer was the difference between viable and not. Measured on the full
MLP training step (1024 -> 256 -> 10, batch 128), which is **15 dispatches**:

| | GPU step | vs CPU | 200-step wall |
|---|---|---|---|
| one submit per kernel | 2.937 ms | 1.53x | 2.2 s |
| recorded once, replayed | **0.660 ms** | **4.72x** | **0.2 s** |

A 4.45x speedup from changing nothing but submission strategy. The arithmetic
checks out: 15 dispatches x ~127 us of per-submit cost is ~1.9 ms of predicted
overhead, and the measured step dropped by 2.28 ms.

Without this the whole project would have looked like a 1.5x curiosity. With it
the training step tracks the raw matmul speedup. **On small models the binding
constraint is not FLOPs or even bandwidth, it is how many times per step you
talk to the driver.**

Two things made the recording possible, and both are consequences of unified
memory rather than clever engineering:

- The batch and labels are written straight into mapped memory the GPU already
  sees, so a new batch needs no re-recording and no upload.
- Step-varying scalars (learning rate, Adam bias correction) live in a mapped
  hyperparameter buffer instead of push constants, which are baked in at record
  time. The CPU writes 6 floats per step.

## 14. MNIST, end to end

`python -m examples.mnist --epochs 4`, model 1024 -> 256 -> 10, batch 128:

| epoch | loss | test accuracy |
|---|---|---|
| 1 | 0.3100 | 95.52% |
| 2 | 0.1311 | 96.86% |
| 3 | 0.0879 | 97.28% |
| 4 | 0.0658 | **97.69%** |

1872 steps in **1.6 s wall**. Best step 0.471 ms, 271,589 samples/s, against
2.398 ms and 53,387 samples/s for the same model in numpy on all 8 CPU cores:
**5.09x**.

Clears the >=97% correctness gate. Everything above it, from the cooperative
matrix loads to the fused AdamW, is exercised by this run.

## 15. Fused kernels in the step

Seven elementwise kernels, each deliberately doing several logical ops in one
pass because every kernel boundary is a DRAM round trip at this ridge point:

| kernel | fuses |
|---|---|
| `bias_gelu` | bias add, GELU, f32->f16 narrowing, saving the pre-activation |
| `bias_gelu_bwd` | GELU derivative, gradient multiply, emit both f16 and f32 |
| `softmax_ce` | max, logsumexp, loss, softmax, gradient, class-padding mask |
| `adamw` | moment updates, bias correction, decoupled decay, weight update, f16 mirror refresh |

AdamW unfused would be four separate passes over every parameter, which at this
ridge point can cost as much as the forward pass.

---

# Phase 3b: a transformer

1.84 M parameter decoder-only model: d=192, 4 heads, 4 layers, seq 128,
batch 16, char vocab 96. **278 dispatches recorded into one command buffer.**

## 16. Correctness

Full forward and backward against an independent numpy implementation of the
same architecture:

| | relative error |
|---|---|
| loss | 2.2e-05 |
| all 20 gradient tensors, worst | 3.0e-03 |

That covers causal attention, LayerNorm, the token and position embeddings
(a scatter-add through `VK_EXT_shader_atomic_float`), residual branches, and
both attention transposes. Training the model on local text takes the loss
from 4.49 to 2.56 in 200 steps.

## 17. Where the time went, and the two access-pattern bugs

The first working version ran a step in 47.4 ms, only 1.15x the CPU, which was
suspicious given the MLP hit 5x. Profiling each kernel separately rather than
guessing:

| kernel | per call | calls/step | total |
|---|---|---|---|
| `attn_softmax` | 3.33 ms | 4 | 13.3 ms |
| `attn_softmax_bwd` | 4.39 ms | 4 | 17.6 ms |
| all 18 matmul shapes | | | ~13 ms |
| everything else | | | ~3 ms |

**The attention softmax was 31 ms of a 47 ms step**, six times the cost of every
matmul combined. The cause was not arithmetic. Giving each invocation one
attention row makes adjacent lanes read addresses `T*4` bytes apart, so every
single access is its own cache line. Restructured to one wave32 per row, with
the row strided across the lanes and the max and sum done as subgroup
reductions, the reads become contiguous.

LayerNorm had the identical bug for the identical reason (one thread per
192-element row) and got the identical fix.

| change | step time | vs CPU |
|---|---|---|
| first working version | 47.4 ms | 1.15x |
| subgroup-per-row attention softmax | 25.0 ms | 2.21x |
| subgroup-per-row LayerNorm | 22.6 ms | 2.40x |
| autotuned matmul tiles | **18.3 ms** | **3.14x** |

**2.6x faster, entirely from memory access patterns and tile selection.** No
change to the maths and no change to the number of dispatches.

## 18. The autotuner earns its keep on awkward shapes

The heuristic tile chooser prefers the widest tile that divides the shape. For
the weight-gradient matmuls that is wrong, because those are short and fat
(`dW = X^T dY` is 192 x 768 x 2048) and a wide tile leaves only 18 workgroups
for 12 CUs:

| shape | heuristic | autotuned | tile chosen |
|---|---|---|---|
| 192 x 768 x 2048 (`dW` qkv) | 0.78 TFLOPS | **3.26** | 64 x 32 |
| 192 x 192 x 2048 (`dW` proj) | 0.88 TFLOPS | 1.14 | 64 x 16 |
| 2048 x 768 x 192 (qkv fwd) | 2.71 TFLOPS | 2.60 | 128 x 128 |

A 4.2x gain on the worst shape, by picking a *smaller* tile that yields more
workgroups. The heuristic is fine for the square-ish forward matmuls and bad
for the transposed backward ones, which is a good argument for measuring
instead of reasoning about tile shape.

## 19. Honest remaining gap

18.3 ms per step is roughly 13 ms of matmul and 5 ms of everything else. The
batched attention matmuls still run at 0.7 to 1.0 TFLOPS because the autotuner
does not yet cover batched shapes, and the head dimension of 48 forces a 16
wide tile. Split-K for the tall-skinny weight-gradient matmuls is the other
obvious lever and is not implemented.

The CPU comparison here is deliberately unfair to the GPU: the numpy baseline
times only the dominant matmuls, with no LayerNorm, no softmax, and no
optimiser, so the real speedup is higher than the 3.14x quoted.


## 20. Model-size sweep: the speedup is flat

Full transformer training step across a 9x parameter range, autotuned:

| d_model | heads | params | GPU step | CPU step (matmuls only) | speedup |
|---|---|---|---|---|---|
| 128 | 2 | 0.83 M | 11.55 ms | 30.54 ms | 2.64x |
| 192 | 4 | 1.84 M | 19.23 ms | 54.76 ms | 2.85x |
| 256 | 4 | 3.24 M | 32.64 ms | 74.54 ms | 2.28x |
| 384 | 6 | 7.22 M | 49.33 ms | 132.15 ms | 2.68x |

Matmul alone reaches 8.2x. Transformer training sits at 2.3-2.9x and does not
improve with size. Both processors share one memory controller, and transformer
training at these sizes is memory-bound end to end, so the ratio converges to a
bandwidth ratio rather than a compute ratio.

**On an APU the iGPU's advantage over its own CPU is capped by the shared memory
bus, not by arithmetic.** Most of the 13.75 TFLOPS of matrix hardware is
unreachable for training, and more of it would not help.

## 21. Non-power-of-two tiles

A 192-wide model with 4 heads has head dim 48. With tile counts restricted to
powers of two the widest tile dividing 48 is 16. Adding `wn=3` to the search
space makes a 48-wide tile reachable:

| shape | before | after |
|---|---|---|
| 128x48x128 batched | 0.31 TFLOPS | 0.59 TFLOPS |

Search spaces built from powers of two quietly exclude the right answer for
real model dimensions. The other two attention shapes did not improve, and the
batched attention matmuls remain the weakest part of the stack.

## 22. Forward/backward split

Transformer step, d=192, autotuned: forward 5.79 ms over 59 dispatches,
backward plus optimiser 11.53 ms over the remaining 165. The 2:1 ratio is
expected, since backward runs two matmuls (dX and dW) per forward matmul.


## 23. Confirming the bandwidth claim

`bench/traffic.py` totals the bytes every dispatch moves and divides by step
time, bracketing effective traffic between compulsory (every buffer once) and
amplified (every matmul tile re-reads its operands).

| d_model | step | compulsory | amplified | ceiling |
|---|---|---|---|---|
| 128 | 11.60 ms | 51.4 GB/s | 78.0 GB/s | 79.75 |
| 192 | 18.31 ms | 51.3 GB/s | 85.4 GB/s | 79.75 |
| 256 | 33.21 ms | 37.0 GB/s | 70.5 GB/s | 79.75 |
| 384 | 48.66 ms | 39.1 GB/s | 85.8 GB/s | 79.75 |

The amplified figure sits on the measured DRAM ceiling, flat across a 9x
parameter range. The step is bandwidth-bound, and effective traffic is close to
the no-reuse bound.

Traffic breakdown at d=192 (1,491 MiB amplified per step):

| kernel | calls | MiB | share |
|---|---|---|---|
| matmul | 75 | 917.4 | 61.5% |
| bias_gelu_bwd | 4 | 84.0 | 5.6% |
| col_sum_chunk | 35 | 81.8 | 5.5% |
| bias_gelu | 4 | 60.0 | 4.0% |
| layernorm_bwd | 9 | 54.1 | 3.6% |

Matmul operand re-reads dominate, so tiling and blocking hold the remaining
headroom. Note that for a 1.84M parameter model the step moves 896 MiB even at
the compulsory bound, roughly 120x the parameter bytes: activations and
intermediates, not weights, are the traffic.


## 24. Can arithmetic replace the autotuner? No.

`bench/predict.py`: for every candidate config on a shape, compute predicted
DRAM traffic and measure actual throughput.

| shape | role | traffic-only | >=1/CU | >=4/CU | >=8/CU | rho | best |
|---|---|---|---|---|---|---|---|
| 2048x768x192 | qkv forward | 100% | 100% | 100% | 95% | +0.85 | 2.74T |
| 1024x1024x1024 | square | 94% | 94% | 92% | 94% | +0.66 | 3.22T |
| 2048x192x768 | dX proj | 53% | 53% | 88% | 90% | +0.68 | 2.01T |
| 2048x96x192 | head forward | 72% | 72% | 89% | 68% | +0.20 | 1.37T |
| 192x768x2048 | dW qkv | 49% | 49% | 41% | 38% | -0.11 | 4.20T |
| 192x192x2048 | dW proj | 23% | 23% | 18% | 96% | -0.29 | 2.22T |

Percentages are the fraction of the measured best achieved by picking the
minimum-traffic config among those with at least N workgroups.

Forward matmuls: the model picks the optimum, rho strongly positive.
Transposed weight-gradient matmuls: rho is *negative*, so traffic actively
misleads. Minimising bytes there favours a tile spanning all of M=192, leaving
12 workgroups for 12 CUs with no slack to hide latency.

No workgroup-count floor fixes it. `>=8/CU` takes dW proj from 23% to 96% while
dropping dW qkv to 38% and costing the forward shapes several percent.

Empirical autotuning stays necessary, and it pays off exactly on the shapes the
analytic model gets wrong, which are the two-thirds of matmul work that backward
represents.
