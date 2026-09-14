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

**Re-measured 2026-09-09, and one run was never enough.** `python bench/mnist_repro.py
--runs 5` re-ran this exact command five times on the same machine and wrote
`bench/mnist-repro.json`. Accuracy 97.51 to 97.71,
median **97.61%**. Speedup 4.90 to 5.65, median **5.34x**. So the single-run 5.09x above
is not optimistic, it sits *below* the median, and the honest headline is a range rather
than either endpoint. A sixth run earlier the same day returned 4.75x, which is why the
range matters: quoting any one of these as the number is a coin flip dressed up as a
measurement.

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

**Corrected in section 52.** The validation losses here were measured by an
`evaluate()` that also ran AdamW on each validation batch: by step 7,500 the model
had taken 300 optimiser steps on validation data. Re-run on today's code with
forward-only evaluation at `PYTHONHASHSEED` 0 to 2, this configuration ends at
validation 1.0068 to 1.0351, not 0.9123, and the pre-fix evaluation reads 0.023
to 0.086 lower than the fixed one at every matched step. Today's code also trains
to a higher loss than this run at matched steps, for reasons section 52 does not
separate, so that range is not this run with the defect removed. The claim below
that validation reaches 0.91 in 12 minutes does not stand. The throughput figures
were not re-measured.

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

**See section 52.** The checkpoint sampled here is section 41's, so it had also
taken 300 optimiser steps on validation batches (the corpus's last 52 files,
`urllib/request.py` to `zoneinfo`). The prime, a tkinter test file, is in the
training split, and 7% of Chinchilla-optimal stands with those steps counted:
16.0M of 216M tokens.

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

**Corrected in section 51.** Timed for 240 s, the same model sustains 975.4,
854.5, 812.4 and 615.2 tokens/s at 256x2, 256x4, 512x2 and 1024x2, measured while
other processes used 43.7 to 63.2% of the same GPU; this section recorded neither
its method nor its conditions. 256x8 was not re-measured. Memory reproduces. The
956-1,117 spread below is within 0.2% of the rates from the median and the fastest
step of one 240 s run, so it may be two statistics of one run rather than
cross-run variation. That is an inference: neither method was recorded, and
458.5 ms is faster than all 458 steps of that run.

**~956-1,117 tokens/s at 3.92 GiB** (the spread is cross-run variation; see
section 46). In fine-tuning terms:

| tokens | wall time |
|---|---|
| 1 M | **17 minutes** |
| 10 M | **2.9 hours** |
| 50 M | 14.5 hours |

Domain adaptation on a laptop iGPU with no CUDA is a matter of hours, not a
matter of renting a GPU.

### Bigger batches collapse near the memory ceiling

**Corrected in section 46.** This originally claimed throughput falls
monotonically with batch and called it an inversion of standard practice.
Re-measured, throughput rises or stays flat with batch until the footprint nears
the memory ceiling, then collapses: 956, 965, then 669 tokens/s at batch 2, 4, 8.
On a smaller model it rises all the way to batch 4. Ordinary behaviour, with the
ceiling arriving sooner than on a discrete card. Section 51's sustained runs do
not confirm the flat part: 975.4 then 854.5 tokens/s at batch 2 and 4, one
process each.

### What this does not claim

**Corrected in section 50.** Loading is written now: `hf_gpt2.py` reads the
checkpoint, tiktoken supplies the BPE tokenizer, and `GPT(tie=True)` ties the
head (124M parameters). A tied model is also about 14% faster than this untied
one (section 51, with its caveat on GPU load), so the figures above are not what a
GPT-2 fine-tune sees.

This measures the architecture with random weights. Loading actual GPT-2
checkpoints would additionally need the weight file and a BPE tokenizer, neither
of which is implemented here, plus weight tying between the embedding and the
output head, which this model does not do (hence 162M parameters rather than
124M). The throughput and memory figures are what a fine-tune would see; the
loading is not written.


## 44. Gradient accumulation helps large models and hurts small ones

**Checked 2026-09-14, unaffected by section 52:** both rows came from inline
`python -c` scripts, not in the repo, that fed random token ids to the training
graphs alone and kept the best of 3 steps after 2 warmups, so `train_lm.py`'s
`evaluate()` never ran and this section reports no loss.

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

**Revisited in section 51.** 956 and 1,117 are within 0.2% of the rates from the
median and the fastest step of one 240 s run at batch 2, which would explain them
as two statistics of one run rather than cross-run variation. That is an
inference, since neither method was recorded, and it has a misfit: 1,117's
458.5 ms step is faster than all 458 steps of that run. The same run's sustained
rate is 975.4, measured while other processes used 60.5% of the same GPU.

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

## 49. The driver does use the matrix units, and the backward pass costs 64% more registers

The largest open number in this project was ~3 TFLOPS against a nominal ~17
TFLOPS f16 peak. Two hypotheses could explain a 5x gap without any data-reuse
problem, and both are now dead:

  * *The driver emulates cooperative matrix.* If `VK_KHR_cooperative_matrix`
    lowered to scalar FMA rather than hardware WMMA, the matrix units would
    never be touched and no tiling would help.
  * *The kernels spill registers.* Spilling round-trips through memory, which on
    a bandwidth-bound machine is fatal.

`bench/isa.py` settles both offline, with no GPU and no profiler GUI, by
compiling SPIR-V to RDNA3 ISA with Radeon GPU Analyzer 2.14.2 targeting gfx1103
(which RGA names "AMD Radeon 780M Graphics" -- the exact part).

| variant | v_wmma | scalar fma | VGPRs | LDS | spills |
|---|---|---|---|---|---|
| forward | 8 | 0 | 76 | 0 | 0 |
| trans_a (dW) | 8 | 0 | **125** | 0 | 0 |
| trans_b (dX) | 8 | 0 | 75 | 0 | 0 |
| batched | 8 | 0 | **125** | 0 | 0 |
| swizzled | 8 | 0 | 76 | 0 | 0 |

The instruction is `v_wmma_f32_16x16x16_f16` in every case, eight per kernel,
with zero scalar FMA and zero scratch memory. **The driver honours cooperative
matrix, including on both transposed backward paths.** A deep-research report
asserted the opposite -- that the Windows driver was "almost certainly falling
back to non-WMMA emulation or scalar LDS fetching" on transposed operands. It is
not. The SPIR-V is also identical in matrix-op count across all five variants,
so nothing is degraded during compilation either.

So the gap is neither emulation nor spilling. What remains is data reuse, and
the repo's own model already says so: `Matmul.traffic_bytes` charges
`reads_a = m*k*2*(n/bn)`, which at `bn=64` and `n=1024` re-reads every element of
A sixteen times. A square matmul has arithmetic intensity `K/4`, so clearing the
~200 FLOP/byte ridge point needs `K >= 803` -- 1024^3 *should* be compute-bound
and measures ~3.2 TFLOPS. It is not reaching `K/4` because the tiling does not
let it.

### The unplanned finding: transposed kernels are register-hungry

The dW kernel uses **125 VGPRs against the forward kernel's 76**, a 64%
increase, with identical tile parameters (sg=4, wm=2, wn=4). Batched is also
125. At the granularity `occupancy()` uses, that is a drop from roughly 12 to 8
waves per SIMD on the kernel shape that dominates the backward pass, which is
two thirds of a training step.

This offers the first mechanical explanation for section 24's anomaly, where the
traffic model predicted forward matmuls well (rho +0.76) and failed on exactly
the transposed dW shapes (rho near zero or negative). Traffic alone cannot
predict a kernel whose occupancy regime differs, and now there is a measured
reason why those shapes differ.

Two consequences worth acting on:

  * `occupancy()` currently reads VGPR counts from the driver at runtime via
    `VK_AMD_shader_core_properties`, which makes the pruner AMD-only and, per
    RESEARCH.md, "undertuned, rejecting only 2 of 84 candidates". RGA supplies
    the same numbers **offline, for any target architecture**, without owning
    the hardware. The autotune search space could be pruned for a GPU that is
    not present.
  * Register pressure on the transposed path is now a named, measurable target
    rather than a suspicion.

Cost of this entire result: one 227 MB download and about ten minutes. It should
have been the first thing done about the 3 TFLOPS question rather than the
tenth.

## 50. Real GPT-2 weights load, and two biases the forward pass never added

Section 43 listed what loading a real checkpoint would need: the weight file, a
tokenizer, and weight tying. All three exist now, and writing them exposed a
defect that every gradient check in this file had missed.

`hf_gpt2.py` reads `.safetensors` with numpy alone (an 8-byte header length, a
JSON header, then raw little-endian tensors, memory-mapped) and maps Hugging
Face's GPT-2 names onto `transformer.GPT` at pinned revisions. `GPT(tie=True)`
makes the head the token embedding transposed, as GPT-2 does: at seq 512 the
tied model has 124,058,112 parameters against 162,717,280 untied
(`bench/section43_remeasure.json`, arms B and A). The BPE tokenizer is not
written here. It is tiktoken's `gpt2` encoding, seeded from the checkpoint's own
`merges.txt` and `vocab.json` after a sha256 check.

`test_hf_gpt2.py` checks it against an independent numpy GPT-2 that shares only
the file reader: no vkgrad kernel, and not the name mapping.

| check | result |
|---|---|
| argmax over an 89-token paragraph | 89/89 agree |
| max logit difference | 0.0970, with logits spanning -191.7 to 60.2 |
| perplexity over 88 predictions | vkgrad 23.2020, numpy 23.1802 |
| 20-token greedy continuation | identical |
| CodeGPT-small-py, 64 random ids | 64/64 agree, max difference 0.0211 |

`attn.c_proj` is 768 x 768, so a transposed load passes every shape check. The
test transposes it on purpose: max logit difference 173.76, argmax 12/89,
perplexity 1803.3. The check can fail.

### The qkv bias was never added

`split_qkv`, the fused kernel that cuts the qkv projection into the three
attention operands, did not add the projection's bias. Backward computed its
gradient and AdamW updated it, so every training run here moved a parameter the
forward pass ignored. Invisible at a zero bias, wrong for any checkpoint whose
bias is not zero.

`test_transformer.py` could not see it, and not because the reference was wrong:
the numpy model does add the bias (`h @ qkv.W + qkv.b`). Every bias started at
its zero init, and the test compares one forward and backward from init, so a
forward pass that skipped the bias and a reference that added it computed the
same thing. At 40ad1b8 the test passes on the defective code, all 20 gradient
tensors, at every hash seed tried: from a `git archive` of 40ad1b8 at
`PYTHONHASHSEED` 0 to 4, worst gradient error 1.39e-03 to 3.44e-03. The seed
matters because `Dense` initialises from `hash(name)`, which Python randomises
per process.

340bb29 draws every bias except the untied head's from a normal distribution (std
0.1) before the check. Against that test, 340bb29's own code with only the three
`bias[...]` terms in `split_qkv` multiplied by zero, run with the loss assertion
reporting instead of stopping so the gradient checks are reached (untied head,
from a `git archive` of 340bb29):

| `PYTHONHASHSEED` | loss, tolerance 5e-3 | gradient tensors failing, tolerance 3e-2 | their errors |
|---|---|---|---|
| 0 | 1.24e-03, passes | 18 of 20 | 7.55e-02 to 2.55e-01 |
| 1 | 4.39e-04, passes | 18 of 20 | 1.42e-01 to 2.68e-01 |
| 2 | 2.07e-04, passes | 18 of 20 | 1.30e-01 to 2.98e-01 |
| 3 | 1.03e-05, passes | 18 of 20 | 4.53e-02 to 1.85e-01 |
| 4 | 6.26e-04, passes | 18 of 20 | 1.15e-01 to 3.08e-01 |

The loss check alone would have missed it on every seed. Only the gradients catch
it.

### The same blind spot, a second time

340bb29's test drew nonzero biases for everything except the untied head, with a
comment saying why: forward did not apply `head.b` either. `GPT.forward`
returned the head's `x @ W` and never added its bias, while backward computed a
gradient for it. Same defect, second place, and the test had been written around
it rather than against it.

`tkernels.py` now has an in-place `add_bias`, dispatched once after the head, and
the test draws `head.b` at std 1.0. Against that test, 340bb29's `transformer.py`
with everything else current fails the loss check before any gradient is
compared: relative error 5.61e-02, 4.71e-02, 4.53e-02, 5.98e-02 and 5.94e-02 at
`PYTHONHASHSEED` 0 to 4.

With `head.b` drawn at std 0.1 like the other biases, the same code still fails
at all six of seeds 0 to 5, but only two
(seeds 1 and 2, at 5.97e-03 and 6.29e-03) fail the loss check. The other four
pass it at 4.6e-03 to 4.8e-03 and are caught by the gradient checks, all 20
tensors at 3.35e-02 to 2.05e-01. Std 1.0 is what makes the loss check itself
catch a dropped head bias.

### What it touches

The runs in sections 41, 42 and 44 trained models whose attention and untied head
had no working bias. Their losses and throughputs are real measurements of that
model, not of the one the code described. Section 52 re-runs section 41's
configuration with both biases working, but does not separate their effect from
the hash seed and other code changes since, and found a second defect in how
those runs measured validation loss. A
checkpoint trained before this fix carries biases that forward never used;
resuming or sampling one now applies them, with an effect that has not been
measured. The throughput cost of adding both biases is in section 51: at most
about 3%, not resolved from noise.

Eleventh correction, and the second of this file's two usual causes: a check run
in one configuration, every bias at zero, and read as covering all of them.
"Gradients verified against numpy" (section 40) was true only where the defect
could not show.

## 51. Section 43 re-measured: best steps, not sustained throughput

Commit 340bb29 recorded 640 to 693 tokens/s fine-tuning loaded GPT-2 weights at
seq 512 to 1024, below section 43's 949 with random weights, and did not isolate
why. A commit message cannot be amended, so this section is its correction too.

`bench/section43_remeasure.py` runs every measurement as its own process,
samples CPU load before and during each, and writes every run with its per-step
times to `bench/section43_remeasure.json`. Section 43's own script is not in the
repo: commit 427284e changed only this file. The committed JSON keeps per-run
totals of other processes' CPU and GPU use, plus the benchmark's own CPU seconds
and GPU share. Other processes' names, pids and CPU seconds were stripped before
committing, and the script now prints those to the console and writes only the
totals. Every number below recomputes from the JSON except the two load facts
marked as notes, which rest on the stripped per-process data.

### Sustained rates

Section 43's model (random weights, untied head), each shape timed for 240 s in
one process after three untimed steps. The last two rate columns come from the
same run's per-step times. Every run in this section was measured while other
processes used 43.7 to 63.2% of the same GPU (see "The GPU was not idle" below).

| seq x batch | sec 43 | sec 46 | **sustained** | from median step | from fastest step | memory, 43 / now |
|---|---|---|---|---|---|---|
| 256 x 2 | 1,117 | 956 | **975.4** | 954.6 | 1,115.0 | 3.92 / 3.920 GiB |
| 256 x 4 | 844 | 965 | **854.5** | 822.7 | 972.1 | 5.12 / 5.116 GiB |
| 256 x 8 | 668 | 669 | not measured | | | 7.51 / not measured |
| 512 x 2 | 949 | | **812.4** | 782.9 | 928.6 | 5.54 / 5.541 GiB |
| 1024 x 2 | 644 | | **615.2** | 638.3 | 678.5 | 10.05 / 10.048 GiB |

Nothing decays over the run: the second half of every one is at least as fast as
the whole (979.4, 855.7, 820.9, 619.2). Memory reproduces to the figure section
43 rounded to.

**949 is not a sustained rate.** Sustained, 512x2 is 812.4, 14% lower. Three
interleaved 45 s runs of the same model gave 828.7, 799.5 and 807.4. Section
43's 1078.7 ms step equals the fastest single step in the 426 steps timed for
this model at this shape, preflight included, and that step occurred once, in
the 40ad1b8 arm below.

**The two published 256x2 figures sit within 0.2% of two statistics of one run.** Within a
single 240 s run, the rate from the median step is 954.6 and from the fastest
step 1,115.0. Section 46's 956 and section 43's 1,117 sit within 0.2% of them.
Section 46 called that gap cross-run variation. One section taking a median and
the other a best step explains it at least as well. Neither method is recorded,
so this is an inference, and it has a clear misfit: section 43's 458.5 ms is
faster than all 458 steps of that run, whose fastest is 459.2 ms.

The fastest step overstates the sustained rate by 10 to 14% at every shape
measured. Section 41 found best-of-four step times 7% optimistic on the 10.8M
model; at GPT-2 scale the same error is twice the size.

**Step times at 512x2 have two modes.** Counting a step as fast when it is within
8% of its run's fastest, fast steps are 14 to 49% of each 512x2 run, and the
median slow step is 1.136 to 1.183x the median fast step, so a fast step is 12.0
to 15.5% quicker. That holds in every arm below, including 40ad1b8, so it
predates this work. The other shapes are less clean. At 256x2 and 256x4 the same
ratio is 1.127 and 1.164. At 1024x2 the rule counts 66% of steps fast at 1.081x,
but that is not two modes: 63 of 73 steps spread continuously from 3018 to
3487 ms, and 10 sit apart at 4082 to 4309 ms. Section 43's step times do not
consistently sit on a mode either: 458.5 ms is faster than every step of the
256x2 run, 1078.7 ms is the single fastest of the 426 steps above, 1213.7 ms
falls between the 256x4 run's fast and slow medians (1076.2 and 1252.3 ms), and
3181.2 ms is inside the 1024x2 spread. The cause of the modes is not established.

CPU, the confound 340bb29 suspected, was quiet: other processes averaged 0.40 to
1.15 of 16 logical cores across every run.

### The GPU was not idle

Other processes' GPU engine use on the same iGPU, summed per run over every
process sampled at 1% or more, was 43.7 to 63.2% in all 23 non-preflight runs,
the failed 256x8 run included. That was the Claude desktop app and dwm, the
desktop compositor, plus at most 1.7% from a browser. In the preflight probes it
was 5.5 to 17.3%, but those probes fired as their 10 s runs ended or after (the
benchmark process itself sampled at or below 0%, and one probe never fired), so
they describe the machine without the benchmark running, not a quieter run.
Which processes carried the load, and when the probes fired, are notes
(`analysis.load_notes` in the JSON): the per-process samples behind them were
stripped, so neither can be recomputed from the committed file.

So every absolute rate in this section, the sustained column included, was
measured under that load, and section 43 recorded nothing about its own
conditions: how much of any gap to section 43 comes from load is unknown.

The arm ratios below are the fairer comparison, since every arm ran interleaved
under the load. It was not uniform across arms, though: 44.2 to 51.6% in the tied
arms (B, C, F) against 56.6 to 63.0% in the untied ones (A, D, E). Whether it is
display work competing for the GPU, which would have slowed the untied arms more,
or an accounting effect of the benchmark's own use of the shared engine was not
separated.

### Loaded weights are not slower. A tied head is faster.

Six arms at 512x2, 45 s each, as separate processes in the order ABCDEF, FEDCBA,
ABCDEF:

| arm | model | tokens/s, 3 runs | median | vs A |
|---|---|---|---|---|
| A | random weights, untied (section 43's model) | 828.7, 799.5, 807.4 | 807.4 | 1.000 |
| B | loaded GPT-2, tied | 914.4, 926.5, 927.7 | 926.5 | 1.148 |
| C | random weights, tied | 895.0, 918.7, 917.4 | 917.4 | 1.136 |
| D | loaded GPT-2, untied | 808.0, 808.7, 813.1 | 808.7 | 1.002 |
| E | random weights, untied, 40ad1b8 (neither bias add) | 816.8, 838.0, 834.2 | 834.2 | 1.033 |
| F | `bench/hf_gpt2_speed.py` unchanged (loaded, tied) | 932, 952, 940 | 940 | 1.164 |

  * **Tying the head is the whole difference.** C over A is 1.136. Loaded against
    random weights is 1.002 untied (D over A) and 1.010 tied (B over C). The tied
    model also allocates 4.893 GiB against 5.541.
  * **The 640 to 693 did not reproduce at 512x2.** The same unchanged script
    measures 932 to 952. Its median is 1.5% above arm B's, and single runs sit
    0.6 to 2.8% above B's median. What produced the
    original range is not established. 340bb29 quoted it for seq 512 to 1024 and
    no tied run at seq 1024 was made, so part of it may be a different
    configuration rather than a failure to reproduce.
  * **The bias fixes cost at most about 3%, not resolved from noise.** E over A was
    0.986, 1.048 and 1.033 round by round, and 1.007 by pooled median step. Arm E
    ran from a temporary worktree of 40ad1b8 that no longer exists; the artefact
    records the label, not a hash of the code that ran.

### 256x8 was not re-measured

The run lost the device: `vkQueueSubmit` returned VkResult -4
(`VK_ERROR_DEVICE_LOST`) after 100 s of process wall time in a 240 s run, and
Windows logged a display-driver timeout. It was not retried, because a retry
could reset the display driver again. Section 43's 668 and section 46's 669 stand
unconfirmed.

### What changes

  * Under the GPU load above, section 43's rates are the sustained column. Its
    fine-tuning table was computed from 956 and stands for that model at 256x2
    under that load, about 2% conservative (975.4 gives 17 minutes, 2.8 hours and
    14.2 hours). An actual GPT-2 fine-tune is tied, and at seq 512 batch 2
    measures 926.5, 18 minutes per million tokens.
  * Section 46's "flat" at GPT-2 scale is not confirmed: 854.5 at batch 4 is 12.4%
    below 975.4 at batch 2. One process each, and section 48 measured
    cross-process spread of 1.13-1.22x on matmul shapes, so this does not settle
    the direction either.

Twelfth correction. The same error as section 41, twice the size: a statistic of
step times quoted as throughput, with the method that produced it not recorded
next to the number.

## 52. Validation in train_lm.py was also training, on the validation set

`examples/train_lm.py`'s `evaluate()` submitted the training graph. With
`--accum 1`, the default and the configuration of sections 41 and 42, that graph
ends in the fused AdamW step, so every validation batch was also an optimiser
step on validation data. With `--accum` above 1 it submitted the microbatch
graph, which pushes each validation batch's gradients into the arena for the
next optimiser step to apply. The loss read back for a batch was computed before
its own update, but every later validation batch, and all training after it, ran
on weights already fitted to the validation batches before it.

Confirmed before the fix, on a 27,024-parameter GPT with the graphs recorded as
`main()` recorded them and the body of `evaluate()` copied verbatim: one call
changed all 30 parameter tensors with `--accum 1` (max |delta| 1.725e-02 at
PYTHONHASHSEED=0, 1.730e-02 on a rerun at the same seed), and with `--accum 2`
left the weights alone but took the arena from all zeros to a max |x| of 1.078
(seed 0, one run). These are control magnitudes showing the defect, not results.

The fix: `GPT.record(..., backward=False)` records the forward pass and the loss
and nothing else, `train_lm.eval_graph` records the validation graph with it, and
`evaluate()` submits that. `test_accum.py` checks that evaluation leaves weights
and arena bit-for-bit unchanged and returns the training forward's loss. It also
submits the two graphs evaluation used to, so the check is seen to fail: they
move the weights (2.35e-03 at PYTHONHASHSEED=0) and the arena (3.38e-01,
4.79e-01 and 4.56e-01 at seeds 0 to 2; 4.91e-01 and 8.94e-01 in two unpinned
runs).

At section 41's cadence, one evaluation of 20 batches every 500 steps, its
7,518-step run made 15 evaluations before the final one: 300 optimiser steps and
614,400 tokens of validation batches, drawn with replacement from an
822,815-token validation split.

### Re-measured

Section 41's configuration on today's code, both biases from section 50 working:

```
python -m examples.train_lm --minutes 12 --eval-every 500 --resume RUN_DIR
```

Defaults are 384d x6, 6 heads, seq 256, batch 8, vocab 96. Section 41's own log
(`runs/lm_384x6/log.jsonl`, not committed) evaluated every 500 steps rather than
the default 250, so these do too. Each of `PYTHONHASHSEED` 0, 1 and 2 ran twice as
separate processes: with the fixed evaluation, and from a copy of the working
tree taken before the fix. Batches come from a fixed generator, so a pair trains
on the same batches and differs by its evaluation plus run-to-run noise, which is
not zero: the same code at the same seed gives train loss 3.184825 and 3.184793
at step 100, and 2.430466 and 2.432321 at step 300. Every log is in
`bench/section52_eval.json`, section 41's rows included.

Train (mean of the last 500 steps) and validation loss, as ranges over the three
seeds:

| step | section 41, train / val | fixed evaluation, train | val | pre-fix evaluation, train | val |
|---|---|---|---|---|---|
| 500 | 2.5573 / 2.1464 | 2.5604 to 2.5782 | 2.2608 to 2.3021 | 2.5610 to 2.5761 | 2.2216 to 2.2744 |
| 2500 | 1.2084 / 1.2119 | 1.2852 to 1.5397 | 1.3108 to 1.5571 | 1.2693 to 1.5356 | 1.2542 to 1.5212 |
| 5000 | 0.9531 / 0.9845 | 0.9653 to 1.0660 | 1.0395 to 1.1078 | 0.9625 to 1.0600 | 0.9857 to 1.0603 |
| 6500 | 0.8978 / 0.9853 | 0.9098 to 0.9652 | 1.0629 to 1.1167 | 0.9053 to 0.9578 (2 seeds) | 0.9902 to 1.0428 (2 seeds) |
| end | 7,518 steps: 0.8717 / 0.9123 | 6,889 to 7,108 steps: 0.8888 to 0.9367 | 1.0068 to 1.0351 | 5,964 to 7,157 steps: 0.8832 to 0.9356 | 0.9196 to 0.9906 |

The end rows are the last 100 steps' mean and a final 20-batch evaluation, as
section 41 reported them. No run reached step 7,500 in 12 minutes.

  * **The old evaluation understated validation loss.** Paired at the same seed,
    it read 0.023 to 0.086 lower at all 37 matched rows (steps 500 to 6,500; 6,000
    and 6,500 have two pairs, because the pre-fix seed 1 run stopped at 5,964).
    Train loss differences have no consistent sign, -0.049 to +0.019. At step 500
    the gap is already 0.023 to 0.039 while train loss agrees within 0.0021, so
    that part can only come from within the one evaluation, each batch scored
    after steps on the batches before it, or from run-to-run noise.
  * **Section 41's validation 0.9123 does not stand.** With forward-only
    evaluation this configuration ends at 1.0068 to 1.0351, and at step 6,500
    reads 1.0629 to 1.1167 against section 41's 0.9853. The pre-fix evaluation on
    the same code lands near section 41, and its gap between validation and train
    loss at step 6,500, 0.085 on both seeds, matches section 41's 0.0875. With the
    fix that gap is 0.152 to 0.158.
  * **Train loss is higher than section 41's with either evaluation**, on all three
    seeds: by 0.012 to 0.113 at step 5,000 with the fix. The evaluation cannot
    explain that. Today's code adds both biases, the seeds differ from section
    41's unrecorded one, and other code has changed since 4ab6f0a; this section
    does not separate them. The seed alone spans 0.10 of train loss at step 5,000
    (0.9653 to 1.0660). So 1.0068 to 1.0351 is today's code with honest
    validation, not section 41's run with the defect removed.
  * **Tokens/s is not measured here.** The runs made 16,963 to 20,357 tokens/s
    with the CPU shared with other work, and the pre-fix seed 1 run lost about a
    minute between steps 2,000 and 2,500. Section 41's 21,382 was measured with
    300 full training steps of evaluation inside its timed loop, where evaluation
    is now forward only; its throughput is not re-measured.

Section 42's checkpoint is section 41's, and the training split, the tkinter prime
and 7% of Chinchilla-optimal are unaffected (see the note there). Section 44's
10.8M comparison is throughput only.

Thirteenth correction, and a new kind: a measurement that changed the thing it
measured. Every validation number `train_lm.py` has printed was taken by a model
that had just trained on the validation batches before it.
