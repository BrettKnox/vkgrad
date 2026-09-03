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
