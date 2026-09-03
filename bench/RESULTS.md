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
| per dispatch, batched into one command buffer | **0.61 us** |
| per submit + fence wait | **91.89 us** |

That is a factor of **150**. A training step of ~50 kernels costs ~31 us of launch
overhead when recorded into one command buffer, versus ~4.7 ms if each is
submitted separately, which would dominate everything else at small model sizes.
Record-once-and-replay is not an optimization here. It is the difference between
viable and not. Section 13 confirms this on the real workload.

(With validation layers on, per-submit rises to ~165 us. Benchmark without them.)

## 5. Compute ceilings

| | measured | theoretical | ratio |
|---|---|---|---|
| fp32 vector FMA | 5.30 TFLOPS | ~8.2 (dual-issue) | 65% |
| f16 WMMA (cooperative matrix) | **15.33 TFLOPS** | ~16.6 | **92%** |

WMMA reaches 92% of theoretical and is **2.9x** fp32. The matrix cores are real,
reachable from Vulkan on a consumer AMD iGPU under Windows, and essentially at
their paper number.

These are post-warmup figures. Cold, the same kernels report 13.75 and 4.94,
and in `--quick` mode 4.56. Section 25 explains why and what it invalidated.

## 6. Ridge point

**~224 FLOP/byte.** Below that arithmetic intensity a kernel is memory-bound.

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
ridge point of ~224 FLOP/byte, writing a transposed copy would cost more than
the matmul consuming it.

**The backward pass of a linear layer now runs on the matrix cores.** That is
the piece no existing Vulkan project has.

## 8. Matmul throughput, and the comparison that matters

| size | GPU TFLOPS | % of WMMA peak | CPU TFLOPS (numpy/OpenBLAS, 8 cores) | GPU speedup |
|---|---|---|---|---|
| 256 | 0.51 | 3.4% | 0.19 | 2.7x |
| 512 | 2.22 | 14.5% | 0.45 | 4.9x |
| 1024 | 3.28 | 21.4% | 0.61 | 5.3x |
| 2048 | 3.43 | 22.3% | 0.70 | 4.9x |
| 4096 | **3.76** | **24.5%** | 0.76 | 5.0x |

**The iGPU beats the CPU sharing its die and its DRAM by about 5x**, steadily,
for anything at or above 512. Below ~256 it is not worth the trouble. That
crossover is the practical answer to "can you train on hardware people already
own". (An earlier cold-clock run of this table showed a spurious 8.2x at 1024,
caused by an unusually low CPU sample rather than a fast GPU one.)

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

Interleaved in one warmed run, median of 15 samples each, with the tuner using
round-robin measurement (section 26):

| shape | heuristic | autotuned | gain | tile chosen |
|---|---|---|---|---|
| 192 x 192 x 2048 (`dW` proj) | 0.34 TFLOPS | **1.86** | **5.5x** | 16 x 16 |
| 192 x 768 x 2048 (`dW` qkv) | 2.07 TFLOPS | 4.07 | 2.0x | 32 x 32 |
| 2048 x 192 x 768 (`dX` proj) | 2.64 TFLOPS | 3.21 | 1.2x | 128 x 64 |
| 2048 x 768 x 192 (qkv fwd) | 2.58 TFLOPS | 2.59 | 1.01x | 256 x 128 |
| 1024 x 1024 x 1024 (square) | 3.16 TFLOPS | 3.18 | 1.00x | 256 x 64 |

1.2x to 5.5x on the transposed and awkward shapes, exactly neutral on forward
ones, never a loss. The gain comes from picking a *smaller* tile that yields
more workgroups.

Two earlier versions of this table were wrong: 4.2x on `dW qkv` (clock ramp,
section 25) and a 0.94x loss on a forward shape (evaluation-order drift,
section 26).

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

Matmul alone reaches about 5x. Transformer training sits at 2.3-2.9x and does not
improve with size. Both processors share one memory controller, and transformer
training at these sizes is memory-bound end to end, so the ratio converges to a
bandwidth ratio rather than a compute ratio.

**On an APU the iGPU's advantage over its own CPU is capped by the shared memory
bus, not by arithmetic.** Most of the 15.33 TFLOPS of matrix hardware is
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
| 128 | 11.48 ms | 52.0 GB/s | 78.8 GB/s | 79.75 |
| 192 | 17.72 ms | 53.0 GB/s | 88.2 GB/s | 79.75 |
| 256 | 32.06 ms | 38.3 GB/s | 73.0 GB/s | 79.75 |
| 384 | 47.84 ms | 39.8 GB/s | 87.3 GB/s | 79.75 |

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
| 2048x768x192 | qkv forward | 100% | 100% | 100% | 96% | +0.76 | 2.51T |
| 1024x1024x1024 | square | 100% | 100% | 91% | 94% | +0.67 | 3.21T |
| 2048x192x768 | dX proj | 51% | 51% | 99% | 100% | +0.71 | 0.67T |
| 2048x96x192 | head forward | 69% | 69% | 80% | 59% | +0.41 | 0.73T |
| 192x768x2048 | dW qkv | 52% | 52% | 39% | 22% | +0.11 | 3.11T |
| 192x192x2048 | dW proj | 22% | 22% | 19% | 94% | -0.32 | 1.96T |

Percentages are the fraction of the measured best achieved by picking the
minimum-traffic config among those with at least N workgroups.

Forward matmuls: the model picks the optimum, rho strongly positive.
Transposed weight-gradient matmuls: rho collapses to near zero or negative, so
traffic carries little information and can actively mislead. Minimising bytes there favours a tile spanning all of M=192, leaving
12 workgroups for 12 CUs with no slack to hide latency.

No workgroup-count floor fixes it. `>=8/CU` takes dW proj from 23% to 96% while
dropping dW qkv to 38% and costing the forward shapes several percent.

Empirical autotuning stays necessary, and it pays off exactly on the shapes the
analytic model gets wrong, which are the two-thirds of matmul work that backward
represents.


## 25. Clock ramp invalidates any benchmark without a warmup

Same matmul, 40 consecutive measurements, nothing else changing:

```
1.14 1.23 1.19 1.25 1.59 | 1.55 1.91 1.99 1.84 2.21 | 2.26 2.37 2.38 2.49 2.54
2.62 2.58 2.75 2.79 2.78 | 2.95 2.58 2.51 2.74 3.03 | 2.84 2.96 3.08 3.05 3.06
3.10 3.02 3.10 3.10 2.81 | 2.97 2.78 2.80 2.71 2.62      TFLOPS
```

Monotonic climb from 1.14 to 3.10 TFLOPS over ~30 dispatches, **2.7x spread**,
then a slight thermal decline. First ten versus last ten: **1.83x**.

This biases any autotuner that evaluates candidates in a fixed order, since
later candidates are measured on a faster GPU. After a 5 s sustained warmup
(`kernels.warmup()`, now called by every benchmark and by `autotune_matmul`)
the spread over 30 repeats is **1.28x** with no trend (first ten vs last ten
0.95x).

Corrected as a result:

| quantity | cold | warm |
|---|---|---|
| f16 WMMA peak | 13.75 TFLOPS | 15.33 |
| fp32 peak | 4.94 TFLOPS | 5.30 |
| ridge point | 200 FLOP/byte | 224 |
| autotuner gain, `dW qkv` | 4.2x | 1.6x |
| autotuner gain, `dW proj` | 1.3x | 5.8x |
| matmul speedup vs CPU | "4 to 8x" | ~5x, consistent |

Every qualitative conclusion survived. Several headline numbers did not.

Residual 1.28x noise remains. Section 26 tests whether it matters.


## 26. Selection rule vs evaluation order

Section 25 left a worry: with 1.28x residual noise, scoring candidates by their
fastest observed time should favour lucky samples, and a median would be safer.

`bench/selection.py` measures every candidate N times, takes each config's
median over all N as ground truth, then bootstraps subsamples through each
selection rule and scores the ground-truth quality of the pick.

| shape | configs | noise | best-of-3 | median-of-3 | best-of-9 |
|---|---|---|---|---|---|
| 1024x1024x1024 square | 88 | 1.16x | 1.0% | 1.0% | 1.3% |
| 2048x768x192 qkv fwd | 128 | 1.24x | 0.1% | 0.7% | 0.0% |
| 192x768x2048 dW qkv | 46 | 1.76x | 1.2% | 1.4% | 1.5% |
| 192x192x2048 dW proj | 31 | 2.21x | 2.6% | 3.5% | 1.1% |

Regret is 0.1-3.5% for every rule and best-of-k beats median-of-k almost
everywhere. The landscape is flat near its top. **The worry was unfounded** and
switching to a median would have been slightly worse.

That contradicted the observed 0.94x loss on `2048x768x192`, where regret is
0.1%. The bootstrap assumed i.i.d. noise; the real tuner measures each config to
completion before moving on, so its error is correlated with evaluation order.
Warmup kills the big ramp, a slower drift survives, and sequential measurement
bakes it into the comparison.

Fix: measure candidates round-robin instead of one at a time.

| shape | sequential | round-robin |
|---|---|---|
| 2048x768x192 qkv fwd | 0.95x | **1.01x** |
| 1024x1024x1024 square | 1.04x | 1.00x |
| 192x768x2048 dW qkv | 2.05x | 1.97x |
| 192x192x2048 dW proj | 5.71x | 5.45x |

The dominant measurement error on this hardware is systematic and time
correlated, not random, so statistical fixes target the wrong failure mode.
Warm up, and interleave anything being compared.


## 27. Ballast: proving traffic is causal

A kernel that only streams N MiB is appended to the recorded training step, and
the step timed for several N, variants interleaved in one warmed run.

| ballast MiB | step ms | delta ms | implied GB/s |
|---|---|---|---|
| 0 | 18.10 | | |
| 64 | 18.86 | 0.76 | 88.2 |
| 128 | 19.43 | 1.33 | 100.9 |
| 256 | 20.95 | 2.85 | 94.3 |
| 384 | 22.97 | 4.87 | 82.6 |
| 512 | 24.27 | 6.17 | 87.0 |

Fitted marginal bandwidth **85.1 GB/s** vs 79.75 measured independently, ratio
1.07. Added bytes cost time at the full DRAM rate, so the step has no spare
bandwidth and traffic is causal rather than merely correlated.

Exchange rate: **1 MiB saved is ~12.3 us saved.**

## 28. Traffic as a step-time predictor

Amplified bytes divided by the 85.1 GB/s marginal rate, with no measurement:

| d_model | measured | predicted | error |
|---|---|---|---|
| 128 | 11.59 ms | 10.64 ms | -8.2% |
| 192 | 18.17 ms | 18.37 ms | +1.1% |
| 256 | 32.80 ms | 27.51 ms | -16.1% |
| 384 | 54.92 ms | 49.06 ms | -10.7% |

Within ~10-16%, systematically under-predicting, so roughly a tenth of the step
is not bandwidth: launch overhead (278 dispatches x 0.61 us = 0.17 ms) plus
kernels that are not purely streaming. Bandwidth-bound with a ~10% residue.

Priced optimisations, not yet implemented:

| change | bytes saved | predicted gain |
|---|---|---|
| drop duplicated f32 gradient buffers (`dz32`, `dqkv32`) | ~80 MiB | ~5% |
| f16 accumulation for attention scores and dP | ~59 MiB | ~4% |


## 29. Predicted, built, verified: dropping the duplicated f32 gradients

Every tensor needing a column sum for a bias gradient already exists in f16 for
the matmuls. The f32 copies (`dz32`, `dqkv32`, f32 `dyxh`) existed only so the
reduction could read f32. `col_sum` now reads f16 and still accumulates in f32.

Predicted before implementing, from shapes alone: ~81 MiB, ~1.0 ms, **4.7%**.

Measured by alternating old and new code in time, six rounds each, from a git
worktree at the previous commit:

```
old  21.52  21.46  23.13  21.43  21.66  22.34   median 21.59 ms
new  20.33  20.53  20.40  20.37  20.41  21.10   median 20.41 ms
```

| | predicted | actual | error |
|---|---|---|---|
| bytes saved | 81 MiB | 82.9 MiB | 2.3% |
| time saved | 1.02 ms | 1.19 ms | 16% |
| step speedup | 4.7% | **5.5%** | |

Traffic totals: amplified 1174.4 -> 1091.5 MiB untuned.

Unpredicted side effects, both good. Timing became much more stable (1% spread
versus 7.8%), consistent with less pressure on a memory controller shared with
the CPU. Gradient accuracy was unchanged, worst 2.64e-03 vs 2.53e-03, since only
the reduction's inputs were narrowed.

Tuned step after the change: **17.52 ms**, versus a traffic-only prediction of
17.35 ms, error **-1.0%**.


## 30. L2-aware workgroup swizzling

Splitting the matmul traffic properly showed where the step's bytes really go:

| component | MiB | share |
|---|---|---|
| matmul operand re-reads (f16) | 756.3 | 53.7% |
| matmul output writes (f32) | 161.1 | 11.4% |
| elementwise | 490.5 | 34.8% |

Operand re-reads dominate. Timing single matmuls in isolation and converting to
implied DRAM traffic at 85.1 GB/s shows how much of that is avoidable:

| shape | implied | compulsory | implied/amplified |
|---|---|---|---|
| fc1 2048x768x192 | 18.7 MiB | 7.0 | 0.96 |
| fc2 2048x192x768 | 15.2 | 4.8 | 1.19 |
| qkv 2048x576x192 | 11.1 | 5.5 | 0.52 |
| dW fc1 192x768x2048 | 10.9 | 4.3 | 0.40 |
| square 1024^3 | 54.8 | 8.0 | 1.24 |

Implied traffic runs 2.7x to 6.9x compulsory. That gap is revisit order, not
physics: with a natural 2D grid, consecutive workgroups sweep one axis of C and
each pulls fresh operand blocks. Grouping the launch so a band of `group_m`
row-blocks completes across all column-blocks keeps that band's A rows and all
of B resident in L2. Same tiles, same arithmetic, different visit order
(Triton/CUTLASS grouped rasterisation over a flat 1D dispatch).

Interleaved, median of 11, one fixed tile per shape so only ordering varies:

| shape | g=0 | g=2 | g=4 | g=8 | best |
|---|---|---|---|---|---|
| fc1 | 1.24 | 1.37 | 1.38 | 1.27 | 1.11x |
| fc2 | 1.06 | 1.13 | 1.05 | 1.12 | 1.07x |
| qkv | 0.70 | 0.97 | 0.87 | 0.76 | **1.38x** |
| dW fc1 | 1.74 | 1.47 | 1.46 | 1.46 | 1.00x |
| 1024^3 | 2.20 | 2.40 | 2.36 | 2.17 | 1.09x |
| 2048^3 | 3.34 | 4.17 | 3.93 | 3.34 | **1.25x** |

Added to the tuner as a cheap second stage (sweep `group_m` for the winning
tile only, rather than quadrupling the search). New matmul records:

| size | before | after | % of peak | vs CPU |
|---|---|---|---|---|
| 2048 | 3.43 | **4.27** TFLOPS | 27.8% | 6.04x |
| 4096 | 3.76 | **4.47** TFLOPS | 29.1% | 5.91x |

**But the transformer step barely moves.** Interleaved A/B against a worktree
at the pre-swizzle commit:

```
old  16.61  17.01  16.84  17.03  16.85   median 16.85 ms
new  16.64  16.78  16.62  16.62  16.55   median 16.62 ms
```

**1.4%**, not the 7.3% a cross-run comparison suggested (that difference was
the old worktree re-tuning to different tiles). Three reasons the step does not
benefit: at d=192 its matmuls are far smaller than the shapes that gain; the
`dW` shapes are two thirds of matmul work and gain nothing, since M=192 leaves
only three row-blocks to group; and the small shapes pick LDS variants, which
have no swizzled path.

Keep it (free, and large models benefit), but it is not a step-change. This is
the third time an interleaved A/B has overturned a cross-run number.


## 31. int8 cooperative matrix: no compute advantage on RDNA3

int8 matrix units are the most widely available accelerator primitive in
consumer silicon, so if int8 training were viable it would reach hardware with
no CUDA and no path to it. Two things decide it, and the issue rate is the
cheap one to measure. Pure register loop, no memory traffic:

| operands | rate | vs f16 |
|---|---|---|
| f16 x f16 -> f32 | 15.30 TOPS | 1.00x |
| f16 x f16 -> f16 | 15.33 TOPS | 1.00x |
| i8 x i8 -> i32 | 15.13 TOPS | 0.99x |
| u8 x u8 -> i32 | 15.13 TOPS | 0.99x |

**Every operand type issues at the same rate.** This contradicts the NVIDIA
intuition, where int8 tensor cores run at roughly 2x fp16. On RDNA3's WMMA int8
buys no arithmetic at all.

So int8's only benefit here is halved operand bytes. Operand traffic is 53.7%
of a step, capping the gain at ~27% before subtracting quantisation overhead
and accepting backward-pass range problems. Not worth building. Worth knowing
before anyone plans int8 work on AMD consumer hardware.

## 32. Cross-vendor gradient exchange through imported host memory

Portable kernels are only half of vendor independence. Multi-GPU training runs
on NCCL, which is NVIDIA-only, so even portable kernels leave the collective
locked in. There is no cross-vendor equivalent.

`VK_EXT_external_memory_host` is a way around it, and this device supports it
(minimum import alignment 4096 bytes). Ordinary host memory can be imported by
several independent VkDevices at once, making plain system RAM a shared arena
between GPUs. Nothing about it requires the devices to share a vendor, a
driver, or an interconnect.

Verified here:

| check | result |
|---|---|
| host allocation imported, GPU writes visible to the CPU | OK |
| two independent VkDevices sharing one allocation | OK (3 + 4 = 7) |
| 2-worker gradient sum over shared memory | OK, 256 KiB, **0 bytes transferred** |

On unified memory the arena is the same DRAM the GPU already reads, so the
all-reduce degenerates into both workers adding into the same bytes: the
cheapest possible collective. On a discrete card it is the host side of that
card's PCIe path.

Scope note: these run two VkDevices from the one physical GPU available here.
That validates the mechanism (independent devices, independent queues, one
shared allocation, writes visible both ways). It does not demonstrate two
vendors, which needs hardware this machine does not have.


## 33. CPU reads from write-combined memory are 73x slower

Found by trying to add a CPU worker to a training job. 1 MiB of f32, CPU side:

| memory kind | CPU read | CPU write |
|---|---|---|
| `shared` (HOST_VISIBLE + COHERENT, write-combined) | **0.32 GB/s** | 43.69 GB/s |
| `cached` (+ HOST_CACHED) | **23.30 GB/s** | 117.82 GB/s |
| `bar` (DEVICE_LOCAL + HOST_VISIBLE) | **0.06 GB/s** | 10.00 GB/s |
| ordinary numpy array, for scale | 23.51 GB/s | |

Write-combined memory reads at 1/73rd of normal memory, and the small BAR
window at 1/390th. The GPU pays only ~2% for host-cached (67.18 vs 68.70 GB/s),
so for anything the CPU reads, `cached` is strictly the right choice.

Section 3 said exactly this, and the code then put master weights in `shared`
anyway. A latent bug that only heterogeneous work exposed. `Param.w32` is now
`cached`, which left GPU-only training unchanged.

## 34. Heterogeneous CPU + GPU pooling does not pay on an APU

The gradient arena is ordinary host memory, so a numpy worker joins with no
plumbing: it adds into the same array the GPU atomically adds into, and reads
weights straight out of host-visible memory. No copies in either direction.

Three things had to be fixed before the measurement meant anything, each worth
recording on its own:

1. **Weights in `cached`, not `shared`** (section 33), or the worker spends
   3.1 ms per parameter just reading them.
2. **The GPU step must be one recorded submit.** Every ctypes call takes the
   GIL and a numpy thread holds it in 5 ms slices, so ~23 eager dispatches and
   a CPU worker starve each other. One submit blocks inside `vkWaitForFences`
   with the GIL released. This alone took GPU-only from 66,545 to 206,228
   samples/s.
3. **A persistent worker thread.** OpenBLAS builds its thread team per calling
   thread, so spawning a fresh thread each step pays that setup every time and
   turns a 0.8 ms shard into 18 ms.

With all three fixed, MNIST, global batch 256:

| CPU share | split | samples/s | vs GPU only |
|---|---|---|---|
| 0 | 256+0 | 206,228 | 1.00x |
| 0.125 | 224+32 | 157,398 | 0.76x |
| 0.25 | 192+64 | 109,249 | 0.53x |
| 0.375 | 160+96 | 84,729 | 0.41x |
| 0.5 | 128+128 | 72,763 | 0.35x |

**Monotonically worse.** The arithmetic shows why: the GPU alone does 256
samples in 1.24 ms and the CPU does 32 in 0.83 ms, so perfect overlap would
give ~235,000 samples/s, a small gain. Measured is 157,398, meaning the GPU step
got about 50% slower while the CPU contributed 14% more samples.

That is DRAM contention. The GPU is already at the memory ceiling (section 27),
so a CPU worker on the same controller takes bandwidth directly out of it.

This sharpens rather than weakens the multi-device case. The reason to pool
devices is *separate memory systems*. Two workers behind one memory controller
share the bottleneck and cannot win. A discrete GPU brings its own VRAM and its
own controller, which is exactly the configuration the shared-arena collective
of section 32 targets.


## 35. The scalar fallback, and what matrix units are actually worth

`VK_KHR_cooperative_matrix` needs RDNA3+, Turing+ or Arc. `fallback.py` adds an
LDS-tiled register-blocked scalar matmul needing only Vulkan 1.1 and 16-bit
storage, matching numpy **exactly** (0.00e+00: f16 operands with fp32
accumulation reproduce the reference bit-for-bit), with both backward
transposes and the batched form.

Peak ratio predicts a 2.9x penalty (15.33 vs 5.30 TFLOPS). Measured,
interleaved, median of 9:

| shape | coopmat | scalar | ratio |
|---|---|---|---|
| 512x512x512 | 1.12 T | 0.66 T | 1.69x |
| 1024x1024x1024 | 2.77 T | 2.27 T | 1.22x |
| 2048x2048x2048 | 4.21 T | 1.92 T | 2.19x |
| 2048x768x192 | 2.92 T | 1.62 T | 1.80x |

On real training, `VKGRAD_NO_COOPMAT=1`:

| workload | matrix units | scalar | cost |
|---|---|---|---|
| MNIST MLP step | 5.39x CPU | 5.07x CPU | ~6% |
| transformer step | 20.58 ms | 21.81 ms | ~6% |
| MNIST accuracy | 97.69% | 97.39% | none |
| autograd + transformer gradient checks | pass | pass | none |

**Matrix units are worth ~6% on a training step, not 2.9x.** The machine is
bandwidth-bound (section 27), so the matrix units are idle most of the time and
removing them removes unused capacity. Their entire contribution fits inside
the ~10% non-bandwidth residue of section 28.

Hardware target is therefore any Vulkan 1.1 GPU, not just recent high-end
parts. The runtime detects the extension, uses it when present, and falls back
automatically when absent.


## 36. The portability claim was false; feature negotiation fixes it

Section 35 claimed the hardware target was "any GPU with a Vulkan 1.1 driver".
Auditing the runtime showed that was not true: `_create_device` requested 15
features unconditionally, and Vulkan fails device creation outright if any
requested feature is unsupported. On a Vulkan 1.1 GPU (GTX 1060, RX 580, Intel
HD 620) `vkCreateDevice` would have failed and nothing would have run.

Worse, most of them were speculative. `bufferDeviceAddress`, `vulkanMemoryModel`,
`vulkanMemoryModelDeviceScope`, `scalarBlockLayout`, `maintenance4`,
`shaderInt8` and the two 8-bit storage features are not used by any kernel in
the project. They were enabled because they looked useful.

The runtime now queries `vkGetPhysicalDeviceFeatures2` first and requests only
the intersection of wanted and supported, down from 15 features to 7:

| feature | gates | required |
|---|---|---|
| `storageBuffer16BitAccess` | f16 operands, halving operand traffic | yes |
| `uniformAndStorageBuffer16BitAccess` | same | yes |
| `shaderFloat16` | f16 arithmetic in shaders | yes |
| `cooperativeMatrix` | matrix units, worth ~6% | no |
| `subgroupSizeControl` | wave32 for cooperative matrix | no |
| `computeFullSubgroups` | same | no |
| `shaderBufferFloat32AtomicAdd` | transformer embeddings, fast bias reductions | no |

`cooperativeMatrix` is also now verified through the *feature* query rather than
extension presence alone, so a driver that advertises the extension but reports
the feature false falls back correctly instead of failing.

`check_device.py` reports all of this for whatever GPU it is run on, including
what the missing pieces would cost, so the failure mode on unsupported hardware
is a clear message rather than a Vulkan error code.

This is the second time a portability claim in this document did not survive
being checked. The general lesson is the same one as the measurement sections:
claims about behaviour on hardware you have not run on need a mechanism that
makes them true, not an assumption that they are.


## 37. Subgroup width was a silent-wrong-answer bug on non-AMD hardware

The row-reduction kernels (LayerNorm, attention softmax) stride a row across the
lanes of one subgroup. The stride must equal the real subgroup width. It was
hardcoded to 32, alongside a `requiredSubgroupSize=32` request that AMD happens
to grant.

Deliberately mismatching stride and width, summing 192 floats per row:

| workgroup | stride | required size | result |
|---|---|---|---|
| 32 | 32 | 32 | exact |
| 64 | 32 | none | **84% error** |
| 64 | 64 | 64 | exact |

Not a crash. Wrong sums, silently. That would have hit Intel (subgroup width
commonly 8, 16 or 32), older AMD at wave64, and anything without
`subgroupSizeControl`, where the request is simply not honoured.

The runtime now negotiates: it reads the native subgroup width, uses 32 when the
device will grant it and the native width otherwise, and generates the row
kernels for whatever width was chosen. Two further bugs surfaced while testing
that path:

- The dispatch multiplier was also hardcoded to 32, so at width 64 only half
  the rows were processed.
- Cooperative matrix kernels assume `local_size = sg*32` tiled by
  `gl_SubgroupID`. Without a 32-wide subgroup that tiling mis-indexes, so
  cooperative matrix is now gated on the device being natively 32 wide (NVIDIA,
  Intel) or granting 32 on request (AMD). Otherwise the scalar path is used.

`VKGRAD_NATIVE_SUBGROUP=1` simulates a device that cannot set subgroup size, so
the adaptive path is exercised on hardware that does not need it. All five
suites now pass in four configurations:

| configuration | approximates | result |
|---|---|---|
| default | RDNA3: coopmat, wave32 | pass |
| `VKGRAD_NO_COOPMAT=1` | Vega, Pascal, pre-Arc Intel | pass |
| `VKGRAD_NATIVE_SUBGROUP=1` | no subgroup size control, wave64 | pass |
| both | oldest supported tier | pass |

Third portability claim in this document that did not survive being checked. The
pattern is now unambiguous: every assumption about hardware not physically
present has been wrong, and the only thing that has ever caught it is building a
way to run the other configuration.


## 38. Memory selection assumed a discrete GPU

`find_memory_type` matched one exact flag combination per role and raised if
nothing matched. The `device` role required `DEVICE_LOCAL` and **forbade**
`HOST_VISIBLE`, which describes a discrete card with private VRAM.

A fully unified device exposes no such memory: on Intel integrated, Apple via
MoltenVK, Mali, Adreno and some AMD APU configurations, every memory type is
host-visible. Since every model buffer defaults to `device`, the first
allocation would have raised and the framework would have been dead on arrival
on all of them. The same applied to `bar`, which many discrete GPUs without
resizable BAR do not expose, and to `shared`, which forbade `HOST_CACHED`.

Each role is now an ordered preference list ending in a permissive entry, and
`Device.memory_tier` records which level was actually satisfied so a weaker
placement is visible rather than silent:

| role | preferred | then | last resort |
|---|---|---|---|
| `device` | device-local, not host-visible | device-local | anything |
| `shared` | host-visible + coherent, uncached | host-visible + coherent | host-visible |
| `cached` | host-visible + cached | host-visible + coherent | host-visible |
| `bar` | device-local + host-visible | host-visible | |

`VKGRAD_UMA=1` simulates a device with no private VRAM by forcing `device` onto
host-visible memory. Under it, all five suites pass and MNIST still trains to
96.8%.

## 39. Five-configuration portability matrix

Three simulation switches now cover the hardware classes this project claims to
support, and the suites run under each:

| configuration | approximates | result |
|---|---|---|
| baseline | RDNA3: coopmat, wave32, split heaps | 5/5 |
| `VKGRAD_NO_COOPMAT=1` | Vega, Pascal, Intel HD: no matrix units | 5/5 |
| `VKGRAD_NATIVE_SUBGROUP=1` | wave64, no subgroup size control | 5/5 |
| `VKGRAD_UMA=1` | Intel, Mali, Adreno: unified memory | 5/5 |
| all three | oldest and most constrained tier | 5/5 |

This does not replace running on real hardware from another vendor. It does mean
the four assumptions that were found to be wrong (unconditional feature
requests, hardcoded subgroup stride, hardcoded dispatch multiplier, discrete-only
memory selection) each now have a configuration that would catch them again, and
that a fifth assumption of the same kind has somewhere to be caught.


## 40. What can actually be trained on this hardware

Everything up to here used 1.8M-parameter demos. The question that matters for
"can people train on hardware they own" is what the ceiling actually is.

Finding it first required removing a limit of my own making. `Kernel` allocated
a single descriptor pool of 64 sets, and a recorded step binds the same kernel
to one set per tensor, so a 6-layer model exhausted it and failed at
construction. Nothing about the hardware required that. Pools are now chained,
allocating another whenever the current one fills.

With that gone, the frontier on a Radeon 780M (12 CU, unified DDR5-5600):

| params | d_model | layers | seq | batch | step | tokens/s | memory |
|---|---|---|---|---|---|---|---|
| 1.8 M | 192 | 4 | 128 | 16 | 20.2 ms | 101,593 | 0.33 GiB |
| 4.8 M | 256 | 6 | 128 | 16 | 67.5 ms | 30,359 | 0.65 GiB |
| 10.8 M | 384 | 6 | 256 | 8 | 89.1 ms | 22,984 | 1.15 GiB |
| 25.4 M | 512 | 8 | 256 | 8 | 225.5 ms | 9,083 | 2.13 GiB |
| 85.4 M | 768 | 12 | 256 | 8 | 801.1 ms | 2,556 | 5.26 GiB |
| 151.6 M | 1024 | 12 | 256 | 4 | 704.6 ms | 1,453 | 5.09 GiB |
| 202.0 M | 1024 | 16 | 256 | 4 | 980.3 ms | 1,045 | 6.78 GiB |
| **315.4 M** | 1280 | 16 | 256 | 2 | 803.2 ms | 637 | **7.31 GiB** |

**315 million parameters trains on an integrated GPU**, in 7.3 of the 11.8 GiB
available. That is past GPT-2 medium. Nothing here is inference: full forward,
backward and AdamW, gradients verified against numpy.

Turned into the number a person actually cares about, at a Chinchilla-optimal
20 tokens per parameter:

| params | tokens needed | wall time on this laptop |
|---|---|---|
| 1.8 M | 36.8 M | **6 minutes** |
| 4.8 M | 96.4 M | **53 minutes** |
| 10.8 M | 216 M | **2.6 hours** |
| 25.4 M | 509 M | **15.6 hours** |
| 85.4 M | 1.71 B | 7.2 days |

A 25 million parameter language model, trained to Chinchilla-optimal, overnight,
on an integrated laptop GPU with no CUDA and no ROCm. A 10 million parameter one
over lunch.

Above ~85M, training from scratch stops being practical (7 days and rising), but
the models still fit and still run, so fine-tuning at those sizes is on the
table.

This is the most direct statement of the project's thesis. The barrier to
training on hardware people already own was never that the silicon cannot do it.


## 41. Sustained throughput, and an over-correction

Section 40's time-to-train figures came from step times measured as best-of-four
back-to-back submits. `examples/train_lm.py` runs the same 10.8M model for a
wall-clock budget with real data loading, loss readback, held-out validation and
checkpointing.

A 2-minute run reported 18,538 tokens/s and I recorded that as the
extrapolation being "19% optimistic". That conclusion was wrong: two minutes is
not long enough to measure sustained throughput, because the cumulative average
is still dominated by startup.

A 12-minute run:

| step | train | val | tokens/s (cumulative) |
|---|---|---|---|
| 500 | 2.5573 | 2.1464 | 22,124 |
| 2500 | 1.2084 | 1.2119 | 21,553 |
| 5000 | 0.9531 | 0.9845 | 21,473 |
| 7500 | 0.8784 | 0.8981 | 21,436 |

7,518 steps, 15.4M tokens, loss 4.7821 to 0.8717, validation 0.9123.

The apparent decline is a converging cumulative average, not throttling: the
successive deltas shrink (-392, -92, -64, -23, ... -8), and instantaneous
throughput over the final six minutes is **21,672 tokens/s**, slightly above the
cumulative figure. There is no thermal decay over 12 minutes of saturation.

| | tokens/s | Chinchilla time |
|---|---|---|
| extrapolated from step times | 22,984 | 2.6 h |
| 2-minute run (too short) | 18,538 | 3.2 h |
| **12-minute sustained** | **21,382** | **2.8 h** |

So the extrapolation was about **7% optimistic**, not 19%. Section 40's figures
should be read as roughly 7% longer.

The meta-result is the more useful one. Having spent five sections establishing
that quantities measured in isolation do not predict behaviour in situ, I then
over-corrected from a two-minute sample and wrote it down as fact. The failure
mode is identical to the clock-ramp bug in section 25, one level up: a
measurement taken before the system reaches steady state, treated as steady
state. Two minutes was the new "cold clocks".

What is not in question: a 10.8M-parameter transformer reaches validation loss
0.91 on held-out Python source in 12 minutes on an integrated laptop GPU, with
no CUDA and no ROCm.


## 42. What 12 minutes of training on a laptop iGPU produces

A validation loss is abstract. `examples/sample_lm.py` loads the checkpoint and
generates, which is also the only check that catches a model optimising
something other than what you intended.

Primed with real corpus text (a tkinter test file), the 10.8M model trained for
12 minutes continues:

```python
def filter(pad, pad, pad, pad, pad, pad, pad, pad, pad, pad)
        self.assertEqual(self.rowcode, pad, pad)
        self.rowcode = rowcode

    def test_pad(self):
        self.rowcode = self.rowcode
        self.do_do_do_do_do_document = self.rowcode

        self.do_document = self._create()
        self.sock
```

It has learned Python block structure, consistent 4- and 8-space indentation,
`self.` attribute access, the `test_` naming convention, and the `assertEqual`
idiom, and it correctly inferred from context that it was inside a unittest
file. The repetition is what a 10.8M model looks like at **7% of
Chinchilla-optimal** (15.4M of 216M tokens).

The first attempt produced 300 `?` characters in a row. That was not the model:
the sampler left-padded the 256-token window with the unknown token, which
barely appears in training, so the context was far outside the training
distribution and the model sensibly continued the padding. Priming with real
text fixed it. Worth recording because a broken prompt and a broken model look
identical from the output, and the instinct is to blame the model.

Total cost of this result: an integrated GPU, no CUDA, no ROCm, no downloaded
dataset, and twelve minutes.


## 43. GPT-2 small's architecture runs on an integrated GPU

Training from scratch is not how most people would use this. Fine-tuning an
existing model is. So: does a real model's architecture even fit?

GPT-2 small (768 d_model, 12 layers, 12 heads, vocab 50257, learned positional
embeddings, pre-LayerNorm, GELU) is architecturally what `transformer.py`
already builds. Measured on the 780M:

| seq | batch | tokens/step | step | tokens/s | memory |
|---|---|---|---|---|---|
| 256 | 2 | 512 | 458.5 ms | **1,117** | 3.92 GiB |
| 256 | 4 | 1024 | 1213.7 ms | 844 | 5.12 GiB |
| 256 | 8 | 2048 | 3064.7 ms | 668 | 7.51 GiB |
| 512 | 2 | 1024 | 1078.7 ms | 949 | 5.54 GiB |
| 1024 | 2 | 2048 | 3181.2 ms | 644 | 10.05 GiB |

**~956-1,117 tokens/s at 3.92 GiB** (the spread is cross-run variation; see
section 46). In fine-tuning terms:

| tokens | wall time |
|---|---|
| 1 M | **15 minutes** |
| 10 M | **2.5 hours** |
| 50 M | 12.4 hours |

Domain adaptation on a laptop iGPU with no CUDA is a matter of hours, not a
matter of renting a GPU.

### Bigger batches collapse near the memory ceiling

**Corrected in section 46.** This originally claimed throughput falls
monotonically with batch and called it an inversion of standard practice.
Re-measured, throughput rises or stays flat with batch until the footprint nears
the memory ceiling, then collapses: 956, 965, then 669 tokens/s at batch 2, 4, 8.
On a smaller model it rises all the way to batch 4. Ordinary behaviour, with the
ceiling arriving sooner than on a discrete card.

### What this does not claim

This measures the architecture with random weights. Loading actual GPT-2
checkpoints would additionally need the weight file and a BPE tokenizer, neither
of which is implemented here, plus weight tying between the embedding and the
output head, which this model does not do (hence 162M parameters rather than
124M). The throughput and memory figures are what a fine-tune would see; the
loading is not written.


## 44. Gradient accumulation helps large models and hurts small ones

Section 43 found throughput falling as batch grows, which suggested a large
effective batch should be assembled from small microbatches. Accumulation is
the same primitive as the multi-device collective from section 32, with workers
separated in time rather than across devices, so `dataparallel.GradAccum` reuses
those kernels.

The first version of this section claimed 84% faster and "inverts standard
practice". Both were wrong, and in the two ways this document keeps getting
things wrong.

Measured in clean processes, effective batch 8 either way:

| model | one batch of 8 | batch 1 x 8 accum | effect |
|---|---|---|---|
| 10.8M (384d x6), 1.15 GiB | 22,930 tok/s | 15,660 | **32% slower** |
| 162M (768d x12), 7.51 GiB | 751 tok/s | 1,160 | **54% faster** |

**It depends on model size, and the direction reverses.** The 84% came from
comparing across separate runs (a clean-process batch-8 measures 751 tok/s, not
the 631 originally recorded), and the "inverts standard practice" generalisation
came from measuring one model size and assuming.

The mechanism is ordinary once the numbers are right. Accumulation trades a
fixed per-microbatch overhead (one submit and fence each, plus the push kernels)
against a memory footprint that shrinks with microbatch size. On a small model
the per-microbatch work is cheap, so the overhead dominates and accumulation
loses. On a large model near the memory ceiling, the footprint drops from
7.51 GiB to 3.93 GiB and that is worth more than the overhead costs.

So the practical rule is the conventional one after all: use accumulation when
the large-batch configuration is close to the memory limit, not otherwise. What
remains specific to this hardware is that the memory limit arrives sooner,
because bandwidth pressure and capacity pressure rise together.

`test_accum.py` verifies correctness independently of any of this: N
microbatches match a single batch of N times the size to 8.8e-06, after the
factor of N that comes from each microbatch's loss being a mean over its own
rows. `GradAccum.record_apply` now applies that 1/N scale itself, and the
multi-device path scales by worker count for the same reason, which it was
previously not doing.

One bug found on the way. The multi-device push uses `atomicAdd`, needed when
several devices write one arena concurrently. Reusing it for single-device
accumulation issues ~162 million atomics per step at GPT-2 scale, slow enough to
trip the 2-second TDR watchdog and return `VK_ERROR_DEVICE_LOST`. Single-device
accumulation has one writer, so a plain read-modify-write is correct and much
faster; the atomic path remains for multi-device.

Eighth correction in this document, and the second of the same two kinds:
comparing across runs, and generalising from a single configuration.


## 45. Audit: the matrix-unit claim was measured the wrong way

"Matrix units are worth ~6%" is the most quoted result here, and it came from
comparing two separate `charlm` runs. That is the cross-run error this document
has made four times.

`Ctx` now takes a `scalar_only` override so both paths can exist in one process
and be alternated. Median of 7, interleaved:

| model | coopmat | scalar | matrix units worth |
|---|---|---|---|
| 1.8M (192d x4) | 20.9 ms | 22.0 ms | +5.4% |
| 10.8M (384d x6) | 90.5 ms | 106.6 ms | +17.8% |
| 25.4M (512d x8) | 124.5 ms | 118.7 ms | **-4.7%** |
| 85.4M (768d x12) | 195.7 ms | 220.8 ms | +12.8% |
| 162M (768d x12, vocab 50k) | 255.5 ms | 275.4 ms | +7.8% |

Range -5% to +18%, median about 8%, no clean trend with size, and one shape
where the scalar path wins outright because the two tile choosers pick
differently.

The published "~6%" was too precise and slightly low. The qualitative
conclusion is unchanged and now rests on five model sizes instead of one:
training without matrix units costs far less than the 2.9x peak ratio implies.

Ninth correction. Same cause as the first, third, sixth and eighth.


## 46. Audit continued: two more claims did not survive

Sections 43 and 44 rested on cross-run measurements. Re-measured properly.

### Record-once was 3.61x, not 4.45x

Eager and recorded paths now coexist, so they can be alternated. Median of 15:

| model | eager | recorded | speedup |
|---|---|---|---|
| MNIST MLP, batch 128 | 2.383 ms | 0.661 ms | **3.61x** |
| MLP 1024x512, batch 256 | 3.524 ms | 1.586 ms | 2.22x |

The published 4.45x came from comparing before and after a code change in
separate runs: the eager path measures 2.383 ms interleaved, not the 2.937
recorded then. The recorded figure was right (0.661 vs 0.660).

The speedup also shrinks with model size, which it should: larger models do more
GPU work per dispatch, so fixed launch overhead is a smaller share. Still the
difference between viable and not on small models, but 3.61x, not 4.45x.

### Throughput does not fall monotonically with batch

Section 43 claimed throughput falls as batch grows and called it an inversion of
standard practice. Interleaved on the 10.8M model, median of 9:

| batch | tokens/s | memory |
|---|---|---|
| 1 | 12,970 | 0.30 GiB |
| 2 | 17,503 | 0.42 GiB |
| 4 | **21,128** | 0.66 GiB |
| 8 | 19,499 | 1.15 GiB |

Throughput **rises** with batch to a peak at 4. Entirely conventional.

At GPT-2 scale, clean processes:

| batch | tokens/s | memory |
|---|---|---|
| 2 | 956 | 3.92 GiB |
| 4 | 965 | 5.12 GiB |
| 8 | 669 | 7.51 GiB |

Flat, then a collapse at batch 8. The originally published 1,117 / 844 / 668 was
again cross-run: batch 2 measures 956 here.

**The correct statement**: throughput rises or stays flat with batch until the
footprint approaches the memory ceiling, then collapses. That is ordinary
behaviour. It also explains section 44 exactly: accumulation helps only for
configurations past the collapse point, which is why it wins at GPT-2 scale and
loses at 10.8M.

### The pattern in these corrections

Three separate "this inverts standard practice" findings have now dissolved into
conventional behaviour once measured interleaved:

| claimed | actual |
|---|---|
| matrix units worth ~6% | -5% to +18%, median ~8% |
| throughput falls as batch grows | rises, then collapses at the memory ceiling |
| accumulation is faster, inverting practice | helps only near the memory ceiling, as usual |

Every one was exciting, counterintuitive, and an artifact of comparing across
runs. The boring conventional behaviour was the real one each time. The
mechanism is not subtle: cross-run variation on this machine reaches 20-30%,
which is the same size as most of these effects, and a difference measured that
way is a coin flip dressed as a result.

What survives all of it, because it was measured interleaved from the start: the
machine is bandwidth-bound (section 27, ballast), matmul speedups do not become
step speedups (1.22-2.19x isolated versus under 20% end to end), and the
hardware floor for training is far below what the peak FLOPS ratio implies.


## 47. The noise floor, measured, and where the corrections really came from

Nine corrections here were blamed on "cross-run variation of 20-30%". That
explanation was itself asserted rather than measured, so `bench/ab.py --noise`
measures it: the same tuned matmul, seven separate processes.

| | spread |
|---|---|
| across-run spread of medians | **1.05x (5%)** |
| across-run spread of bests | 1.06x (6%) |
| across-run stdev / mean | 2.0% |

**Raw cross-run noise is 5%, not 20-30%.** So the stated reason for the
corrections was wrong.

The actual cause is the autotuner. Running the same tuning sweep in three
separate processes:

| shape | configs chosen | resulting TFLOPS | spread |
|---|---|---|---|
| 2048x768x192 | three different | 2.48 / 3.10 / 3.17 | **1.28x** |
| 192x768x2048 (dW) | three different | 4.18 / 4.20 / 4.40 | 1.05x |
| 1024x1024x1024 | three different | 3.39 / 3.27 / 3.30 | 1.04x |

**The tuner picks a different configuration nearly every run**, and on one shape
that cost 28% because it missed the swizzled variant entirely. Its sweep measures
~85 candidates once each, and 5% noise across candidates that differ by less than
that is enough to choose wrongly.

So any measurement that includes a fresh autotune is not reproducible across
runs to better than about 30%, while the underlying hardware is reproducible to
5%. That is a property of the framework, not the silicon, and it is the real
mechanism behind the corrections.

### A fix that did not work

The obvious remedy is a runoff: re-measure the top three candidates interleaved
and pick between those. Implemented and tested across three runs, it made
selection *worse*, spread 1.28x to 1.80x. Reverted rather than kept.

Why it failed is not established. The plausible explanation is that many
candidates sit within noise of each other while a few are genuinely much better,
so a runoff among the top three of a noisy ranking often does not contain the
real winner at all. Fixing that properly means making the first pass less noisy,
not re-ranking its output.

### What to do instead, for now

Use the autotune disk cache, which is on by default: a shape is tuned once and
the choice reused, so runs are reproducible even though the tuning itself is
not. Every benchmark in this document that disabled the cache should be read as
carrying up to 30% selection variance on top of the 5% hardware variance.

`bench/ab.py` also provides `interleaved(variants)` so the correct comparison
method is the convenient one.

## 48. The tuner fix: half the regret, none of the reproducibility

Section 47 claimed the autotuner was nondeterministic, on the evidence that three
processes picked three configs spanning 1.28x. That claim was not established.
Scoring a pick by re-measuring it in its own process carries 1.13-1.22x of
cross-process noise on these shapes, which swallows the effect. The 5% noise
floor measured in section 47 came from one shape with one configuration and does
not generalise. Tenth correction, same cause as the other nine.

Two real defects were found by reading the tuner, and both are fixed:

1. **Stage two compared across measurement sessions.** `gbest = (best["tflops"], 0)`
   seeded the swizzle comparison with a stage-one number taken minutes earlier
   under a different protocol; the `group_m=0` baseline was never re-measured.
   Now all four variants are measured round-robin in stage two.
2. **Equal repeat count is not equal measurement effort.** `min(reps, 0.2/warm)`
   timed a 0.1 ms config over 0.3 ms of work and a 70 ms config over one
   dispatch, so per-submit fence overhead contaminated each measurement in
   inverse proportion to kernel speed, systematically penalising the fastest
   tiles. `_repeat_for()` now sizes each measurement to ~20 ms of GPU time.

Note this inverts the mechanism first proposed. The prediction was that `min`
over a noisy estimator would flatter slow configs; the dominant effect runs the
other way and is simpler. The fix is right, the reasoning behind it was wrong.

### Measuring it properly

`bench/determinism.py` runs both arms in alternating processes, then measures
every configuration anyone picked ONCE, interleaved, in a single process. Picks
are scored against that shared table, so spread reflects selection alone.

| shape | arm | distinct picks / 3 | mean regret | worst |
|---|---|---|---|---|
| 2048x768x192 qkv forward | fixed | 3 | **8.5%** | 19.4% |
| 2048x768x192 qkv forward | legacy | 1 | **19.4%** | 19.4% |
| 1024^3 square | fixed | 1 | 2.1% | 2.1% |
| 1024^3 square | legacy | 2 | 1.4% | 2.1% |
| 192x768x2048 dW qkv | fixed | 3 | 2.2% | 4.1% |
| 192x768x2048 dW qkv | legacy | 3 | 2.0% | 4.1% |

Aggregate mean regret 7.6% -> 4.3%. But the shape of the win is not what was
predicted: legacy picked the *same* config every run and that config was always
19.4% below optimum. The fix picks differently each run and sometimes finds the
best. It is better on average and less consistent. Reliably wrong became
intermittently right.

### The residual is a design flaw, not noise

All three fixed picks on the qkv shape were `wm2 wn8 lds0 g2`, differing only in
`sg`: same tile width, same swizzle, three tile heights spanning 3.01 to 3.74
TFLOPS. Width and swizzle are chosen consistently; height is not resolved at all.

The likely cause is structural. Stage one ranks tiles with `group_m=0`, then
stage two swizzles only the winner, on the stated assumption that workgroup
ordering is "nearly orthogonal to tile shape". If the swizzle's benefit depends
on tile height, that assumption is false and the two-stage split cannot find the
optimum however well each stage is measured. Untested; the cheap test is to
sweep `group_m` over the top-K tiles rather than the top one.

### Stopping here

That test is not being run. Section 8 and section 20 already establish that
matmul speedups do not become step speedups: 1.22-2.19x in isolation became
under 20% end to end, flat across five model sizes. An 8.5% tile-selection
regret is worth perhaps 1-2% of a training step. Several iterations have now
gone into this autotuner, which is precisely the failure mode this file
documents elsewhere. The bytes moved are the lever; the tuner is not.
