# Training neural networks on a consumer iGPU through Vulkan

**What this is.** A complete training stack, built from nothing but Python's
`ctypes` and a shader compiler, that runs forward pass, backward pass and
optimiser on AMD's matrix cores through Vulkan compute. It trains MNIST to
97.69% and a 1.84M parameter transformer, on an integrated GPU under Windows,
where ROCm does not officially reach.

**Why it might matter.** CUDA is the only mature training stack. Everything
else is inference-only or vendor-locked. Vulkan's `cooperative_matrix`
extension exposes matrix-multiply hardware on AMD, Intel, NVIDIA, Qualcomm and
ARM, on Windows, Linux and Android. Before this, it was used for inference
(llama.cpp, ncnn, a FlashAttention-2 forward-pass demo) and nothing ran a
backward pass on it.

**Hardware.** AMD Radeon 780M (RDNA3 / gfx1103), 12 CU, integrated in a Ryzen 7
7840HS, sharing 32 GB of DDR5-5600 with the CPU. Vulkan 1.4.315, AMD's Windows
driver. Everything below is measured on this machine; the numbers are in
[bench/RESULTS.md](bench/RESULTS.md) and reproducible from this repo.

---

## 1. The number that determines everything

| | measured |
|---|---|
| f16 WMMA peak (cooperative matrix) | 15.33 TFLOPS (92% of theoretical) |
| fp32 vector peak | 5.30 TFLOPS |
| DRAM read bandwidth | 79.75 GB/s (89% of theoretical) |
| **ridge point** | **~224 FLOP/byte** |

(All compute figures here are taken after a sustained warmup. See section 2.7:
without one they read roughly 2x low, and an earlier draft of this document
under-reported the WMMA peak as 13.75 TFLOPS for exactly that reason.)

Two hundred and twenty-four FLOP per byte is a brutal ratio. For scale: an unfused elementwise
op is ~0.1 FLOP/byte, a fused bias+GELU chain maybe 2, a 64x64-tiled f16 matmul
about 32. **Nothing in a small training workload is compute-bound on this
machine.**

So the compiler's job is minimising bytes moved. Tile sizes and instruction
selection are second-order until byte traffic is already minimal. Every design
decision below follows from that, and several of them contradict what a CUDA
background would suggest.

## 2. Five results

### 2.1 Unified memory does not mean "never copy"

The obvious idea on an APU is to stop copying: CPU and GPU share DRAM, so map
the memory once and let both touch it. Measured:

| memory kind | read GB/s |
|---|---|
| host-visible, zero-copy | 68.70 |
| device-local, needs staging | 79.75 |
| staging upload cost | 23.16 one-way |

Host-visible runs at 86% of device-local. So neither "always copy" nor "never
copy" is right, and the correct policy is per tensor, by role:

- **Input batches and labels** cross the CPU boundary every step and are read
  once. Zero-copy wins; staging would cost more than the 14% gap.
- **Weights and optimiser state** never touch the CPU after init and are read
  every step. Device-local wins; the staging cost amortises over the whole run.
- **Loss and gradient norms** are read back by the CPU. These need host-*cached*
  memory: write-combined memory is fast to write and painfully slow to read.

This is why the allocator exposes `shared` / `device` / `cached` and picks by
role rather than having one global answer.

### 2.2 Launch overhead, not FLOPs, is the binding constraint on small models

| | cost |
|---|---|
| dispatch, batched into one command buffer | 0.61 us |
| submit + fence | 91.89 us |

A factor of 150. The MLP training step is 15 dispatches; the transformer step
is 278. Submitting each separately spends 1.4 ms and 26 ms respectively on
nothing but talking to the driver.

Recording the entire step once and replaying it:

| | GPU step | vs CPU |
|---|---|---|
| one submit per kernel | 2.937 ms | 1.53x |
| recorded once, replayed | **0.660 ms** | **4.72x** |

**4.45x from changing nothing but submission strategy.** Without it the whole
project reads as a 1.5x curiosity.

Two things make the recording possible, and both are consequences of unified
memory rather than cleverness. The batch and labels are written straight into
mapped memory the GPU already sees, so a new batch needs no re-recording. And
step-varying scalars (learning rate, Adam bias correction) live in a mapped
hyperparameter buffer instead of push constants, which are baked in at record
time. The CPU writes six floats per step.

### 2.3 Two access-pattern bugs cost more than every matmul combined

The first working transformer ran a step in 47.4 ms at 1.15x the CPU, which was
suspicious given the MLP reached 5x. Profiling each kernel rather than guessing:

| kernel | per call | calls/step | total |
|---|---|---|---|
| `attn_softmax` | 3.33 ms | 4 | 13.3 ms |
| `attn_softmax_bwd` | 4.39 ms | 4 | 17.6 ms |
| all 18 matmul shapes | | | ~13 ms |
| everything else | | | ~3 ms |

**The attention softmax was 31 ms of a 47 ms step, six times the cost of every
matmul combined.** The cause was not arithmetic. Assigning one invocation per
attention row makes adjacent lanes read addresses `T*4` bytes apart, so every
access is its own cache line. Restructured to one wave32 per row, with the row
strided across lanes and the max and sum done as subgroup reductions, the reads
become contiguous. LayerNorm had the identical bug for the identical reason.

| change | step | vs CPU |
|---|---|---|
| first working version | 47.4 ms | 1.15x |
| subgroup-per-row attention softmax | 25.0 ms | 2.21x |
| subgroup-per-row LayerNorm | 22.6 ms | 2.40x |
| autotuned matmul tiles | **18.3 ms** | **3.14x** |

2.6x, entirely from memory access patterns and tile selection, with no change
to the mathematics and no change to the number of dispatches.

### 2.4 Negative result: LDS staging is not the lever it is on NVIDIA

The textbook CUDA optimisation is staging tiles through shared memory. The
direct-from-global kernel re-reads B once per subgroup, so staging should cut B
traffic by a factor of `sg`. At 512x512x512:

| variant | best config | TFLOPS |
|---|---|---|
| direct from global | sg4 wm4 wn4 | 1.445 |
| LDS staged | sg4 wm2 wn4 | 1.303 |

LDS was **slower**. All subgroups in a workgroup read the same B columns at the
same time, so the L2 already serves the reuse LDS was going to provide, and
staging adds two barriers and an LDS round trip per K-block. It only turns
positive once tiles get wide.

What actually reduces traffic is tile *width*. Global traffic for a tiled
matmul is `M*K*(N/BN) + K*N*(M/BM)`; only BM and BN appear. Widening the search
from `wn <= 4` to `wn <= 8` moved 512x512x512 from 1.44 to 2.19 TFLOPS, +52%,
with no other change.

### 2.5 The autotuner beats heuristics precisely where intuition fails

A reasonable heuristic picks the widest tile that divides the shape. That is
right for square forward matmuls and wrong for the transposed backward ones,
which are short and fat:

Measured by interleaving the tuned and heuristic configs in one warmed run and
taking the median of 15 samples each, which is the only comparison method that
survived section 2.8:

| shape | heuristic | autotuned | gain | tile chosen |
|---|---|---|---|---|
| 192 x 192 x 2048 (`dW` proj) | 0.34 TFLOPS | **1.86** | **5.5x** | 16 x 16 |
| 192 x 768 x 2048 (`dW` qkv) | 2.07 TFLOPS | 4.07 | 2.0x | 32 x 32 |
| 2048 x 192 x 768 (`dX` proj) | 2.64 TFLOPS | 3.21 | 1.2x | 128 x 64 |
| 2048 x 768 x 192 (qkv fwd) | 2.58 TFLOPS | 2.59 | 1.01x | 256 x 128 |
| 1024 x 1024 x 1024 (square) | 3.16 TFLOPS | 3.18 | 1.00x | 256 x 64 |

The tuner wins by 1.2x to 5.5x on the transposed and awkward shapes and is
exactly neutral on the forward ones, where the heuristic already picks
correctly. It never loses. The gain comes from choosing a *smaller* tile,
because a wide tile leaves only 18 workgroups for 12 CUs.

(Two earlier drafts of this table were wrong. The first claimed 4.2x on
`dW qkv`, inflated by the clock ramp of section 2.7. The second showed the
tuner *losing* 0.94x on a forward shape, which section 2.8 traced to
evaluation-order drift inside the sweep and fixed.)

A related trap: real models have dimensions that are not powers of two. A
192-wide model with 4 heads has head dim 48, and with only power-of-two tile
counts the widest tile dividing 48 is 16. Adding `wn=3` to the search space
made a 48-wide tile reachable and nearly doubled one attention matmul
(0.31 to 0.59 TFLOPS). Search spaces built from powers of two quietly exclude
the right answer for real architectures.

The pruning pass uses `VK_KHR_pipeline_executable_properties` to read VGPR and
LDS usage from the driver after compiling but before running, combined with
`VK_AMD_shader_core_properties` for the register file size, to reject
configurations that cannot reach useful occupancy. It works and costs one
compile per candidate, but currently prunes only 2 of ~84 configs: the
threshold is set conservatively and is not yet earning much.

### 2.6 The cost model predicts forward matmuls and is actively wrong on backward ones

Section 3 shows the step runs at the DRAM limit. That invites an obvious
simplification: if the machine is bandwidth-bound, the config moving the fewest
bytes should win, and tile selection becomes arithmetic rather than a benchmark
sweep. `bench/predict.py` tests it by computing predicted traffic and measured
throughput for every candidate config on a shape.

| shape | role | traffic model picks | Spearman rho |
|---|---|---|---|
| 2048 x 768 x 192 | qkv forward | **100%** of best | +0.76 |
| 1024 x 1024 x 1024 | square | **100%** | +0.67 |
| 2048 x 192 x 768 | dX proj | 51% | +0.71 |
| 2048 x 96 x 192 | head forward | 69% | +0.41 |
| 192 x 768 x 2048 | **dW qkv** | 52% | **+0.11** |
| 192 x 192 x 2048 | **dW proj** | 22% | **-0.32** |

The split is by shape family, not by size. For forward matmuls the model picks
the measured optimum, and predicted and measured orderings agree strongly. For
the transposed weight-gradient matmuls it collapses: rho near zero on one and
*negative* on the other, so traffic is not merely uninformative there, it can
point the wrong way.

The mechanism is visible. Those shapes have M=192, so minimising traffic favours
a tile as wide as the whole M dimension, which leaves 12 workgroups for 12 CUs
and no slack to hide DRAM latency. The obvious correction, filtering to configs
with a minimum number of workgroups per CU before minimising traffic, does not
rescue it: a `>=8 per CU` floor takes `dW proj` from 23% to 96% but drops
`dW qkv` to 38% and costs the forward shapes several percent. No single
threshold works.

**Conclusion: empirical autotuning cannot be replaced by this cost model**, and
it earns its cost precisely where the model fails. Since backward runs two
matmuls for every forward one, the shapes the analytic model gets wrong are the
majority of the work. This is the concrete reason section 2.5's autotuner found
gains of up to 5.8x on exactly these shapes.

### 2.7 Benchmarks on this hardware are invalid without a sustained warmup

This one invalidated several of my own earlier numbers, so it goes in the
results rather than the appendix.

Running the *same* matmul 40 times in a row, back to back, with nothing else
changing:

```
1.14 1.23 1.19 1.25 1.59 | 1.55 1.91 1.99 1.84 2.21 | 2.26 2.37 2.38 2.49 2.54
2.62 2.58 2.75 2.79 2.78 | 2.95 2.58 2.51 2.74 3.03 | 2.84 2.96 3.08 3.05 3.06
3.10 3.02 3.10 3.10 2.81 | 2.97 2.78 2.80 2.71 2.62      TFLOPS
```

Throughput climbs monotonically from 1.14 to 3.10 TFLOPS over roughly 30
dispatches, a **2.7x spread**, then drifts down slightly as heat accumulates.
The GPU starts at idle clocks and takes seconds of sustained load to reach its
boost state. Mean of the first ten measurements versus the last ten: **1.83x**.

This is fatal for an autotuner that walks a candidate list in a fixed order,
because configurations evaluated late are measured on a faster GPU than
identical ones evaluated early. It silently rewards whatever happens to be at
the end of the list.

After a 5 second sustained warmup, the spread over 30 repeats falls to **1.28x**
with no trend (first ten versus last ten: 0.95x). `kernels.warmup()` now runs
before every benchmark and at the top of `autotune_matmul`.

What this corrected, once re-measured with clocks settled:

| claim | before warmup | after |
|---|---|---|
| f16 WMMA peak | 13.75 TFLOPS | **15.33** |
| fp32 peak | 4.94 TFLOPS | **5.30** |
| ridge point | 200 FLOP/byte | **224** |
| autotuner gain on `dW qkv` | 4.2x | **1.6x** |
| autotuner gain on `dW proj` | 1.3x | **5.8x** |
| matmul speedup vs CPU | "4 to 8x" | **~5x, consistent** |

The qualitative conclusions all survived. Several of the headline numbers did
not. The general lesson is that on a laptop APU, where the GPU shares a power
and thermal budget with the CPU and sits at idle clocks most of the time, a
benchmark that does not explicitly reach steady state is measuring the power
management policy rather than the kernel.

Residual noise of 1.28x remains. I assumed that mattered, because taking a max
over ~85 candidates should favour lucky samples. Section 2.8 tested that
assumption and it was wrong.

### 2.8 The selection rule was not the problem; evaluation order was

Section 2.7 left a plausible-sounding worry: with 1.28x residual noise, scoring
each candidate by its *fastest* observed time should reward whichever config got
a lucky sample. The obvious fix is a median instead of a max. That was worth
testing before believing.

`bench/selection.py` measures every candidate N times, defines ground truth as
each config's median over all N, then bootstraps: repeatedly draw k samples per
config, apply a selection rule, and score the *ground-truth* quality of whatever
it picked. Regret is how far below the true optimum the pick lands.

| shape | configs | per-config noise | best-of-3 | median-of-3 | best-of-9 |
|---|---|---|---|---|---|
| 1024x1024x1024 square | 88 | 1.16x | 1.0% | 1.0% | 1.3% |
| 2048x768x192 qkv fwd | 128 | 1.24x | 0.1% | 0.7% | 0.0% |
| 192x768x2048 dW qkv | 46 | 1.76x | 1.2% | 1.4% | 1.5% |
| 192x192x2048 dW proj | 31 | 2.21x | 2.6% | 3.5% | 1.1% |

**The worry was unfounded.** Regret is 0.1% to 3.5% for every rule, and
best-of-k is equal or better than median-of-k everywhere. The throughput
landscape is flat near its top, so selecting a slightly wrong config costs
almost nothing. Switching to a median would have made things marginally worse.

But that result created a contradiction. If selection regret on
`2048 x 768 x 192` is 0.1%, why did the tuner pick a config 6% worse than the
heuristic on exactly that shape? The answer is that the bootstrap modelled
**i.i.d.** noise, while the real tuner measures each candidate to completion
before starting the next, so its noise is **correlated with evaluation order**.
Warmup removes the large ramp; a slower residual drift across a sweep survives,
and measuring one config at a time bakes it into the comparison.

The fix is not a better statistic, it is a better schedule: measure the
candidates round-robin, so drift is spread evenly across all of them rather
than accumulating along the list. After that change the tuner stops losing:

| shape | before (one config at a time) | after (round-robin) |
|---|---|---|
| 2048 x 768 x 192 qkv fwd | 0.95x vs heuristic | **1.01x** |
| 1024 x 1024 x 1024 square | 1.04x | 1.00x |
| 192 x 768 x 2048 dW qkv | 2.05x | 1.97x |
| 192 x 192 x 2048 dW proj | 5.71x | 5.45x |

The general lesson, and the reason both 2.7 and 2.8 exist: on this hardware the
dominant measurement error is **systematic and correlated with time**, not
random. Statistical fixes address the wrong failure mode. What works is
removing the correlation, by warming up before measuring and by interleaving
anything being compared. Every A/B comparison in this document is now measured
interleaved within a single warmed run.

### 2.9 Traffic is causal, not just correlated, and it prices optimisations

Section 3 shows effective traffic sitting on the DRAM ceiling. That is
consistent with being bandwidth-bound but does not establish it: a step can sit
near the roofline and still be limited by something else, in which case removing
bytes would buy nothing.

`bench/ballast.py` is the controlled version. A kernel that does nothing but
stream N MiB is appended to the recorded training step, and the step is timed
for several N, with the variants interleaved inside one warmed run.

| ballast | step | delta | implied GB/s |
|---|---|---|---|
| 0 MiB | 18.10 ms | | |
| 64 | 18.86 ms | 0.76 | 88.2 |
| 128 | 19.43 ms | 1.33 | 100.9 |
| 256 | 20.95 ms | 2.85 | 94.3 |
| 384 | 22.97 ms | 4.87 | 82.6 |
| 512 | 24.27 ms | 6.17 | 87.0 |

Least-squares slope over the whole range gives a **marginal bandwidth of
85.1 GB/s**, against 79.75 GB/s measured independently by a pure streaming
kernel. Ratio 1.07.

**Added bytes cost time at the full DRAM rate.** The step has no spare
bandwidth whatsoever, so traffic is causal: every byte removed is time removed.
The exchange rate is concrete and useful, **1 MiB saved is about 12.3 us
saved**, which prices any proposed optimisation before writing it.

Turning that around, the analytic byte count becomes a step-time predictor with
no measurement at all:

| d_model | measured step | predicted from traffic | error |
|---|---|---|---|
| 128 | 11.59 ms | 10.64 ms | -8.2% |
| 192 | 18.17 ms | 18.37 ms | +1.1% |
| 256 | 32.80 ms | 27.51 ms | -16.1% |
| 384 | 54.92 ms | 49.06 ms | -10.7% |

Within roughly 10 to 16%, and **systematically under-predicting**, which is
itself informative: about a tenth of the step is time that traffic does not
explain. Some is launch overhead (278 dispatches at 0.61 us is 0.17 ms), the
rest is kernels that are not purely streaming. So the honest summary is that
this workload is bandwidth-bound with a ~10% residue, not perfectly
bandwidth-bound.

Two consequences worth stating plainly. First, no amount of extra arithmetic
throughput would help this machine on this workload; the 15.33 TFLOPS of matrix
hardware is not the constraint and never becomes it. Second, optimisations can
now be ranked before being built. Eliminating the duplicated f32 gradient
buffers (`dz32`, `dqkv32`, which exist only so the bias reduction can read
f32) is worth about 80 MiB, or **5% of the step**; moving the attention score
and gradient tensors to f16 accumulation is worth about 59 MiB, or **4%**.
Neither is transformative, which is the useful part: the model says where the
ceiling is, and it is close.

## 3. The headline comparison: an iGPU against the CPU on its own die

Same silicon, same DRAM, same model. This is the question that matters for
"can people train on hardware they already own", and it is not one anybody
publishes an answer to.

**Matmul** (f16 WMMA vs numpy/OpenBLAS f32 on 8 cores):

| size | GPU TFLOPS | % of WMMA peak | CPU TFLOPS | speedup |
|---|---|---|---|---|
| 256 | 0.51 | 3.4% | 0.19 | 2.7x |
| 512 | 2.22 | 14.5% | 0.45 | 4.9x |
| 1024 | 3.28 | 21.4% | 0.61 | 5.3x |
| 2048 | 3.43 | 22.3% | 0.70 | 4.9x |
| 4096 | 3.76 | 24.5% | 0.76 | 5.0x |

**MNIST MLP**, full training step including backward and AdamW: 0.471 ms vs
2.398 ms, **5.09x**, reaching 97.69% test accuracy in 1.6 seconds of wall time.

**Transformer**, full training step, across a 9x range of model sizes:

| d_model | heads | params | GPU step | CPU step | speedup |
|---|---|---|---|---|---|
| 128 | 2 | 0.83 M | 11.55 ms | 30.54 ms | 2.64x |
| 192 | 4 | 1.84 M | 19.23 ms | 54.76 ms | 2.85x |
| 256 | 4 | 3.24 M | 32.64 ms | 74.54 ms | 2.28x |
| 384 | 6 | 7.22 M | 49.33 ms | 132.15 ms | 2.68x |

(The CPU figure counts only the dominant matmuls, with no LayerNorm, softmax or
optimiser, so it flatters the CPU.)

### The interesting part: the transformer speedup is flat

Matmul speedup settles at about 5x. Transformer training sits at 2.3 to 2.9x and
**does not improve with model size** across a 9x parameter range.

That is not a bug, it is the architecture, and it is measured rather than
inferred. `bench/traffic.py` totals the bytes every dispatch in a step moves and
divides by the step time. Two bounds are reported, because neither is exactly
right alone: *compulsory* traffic touches every buffer once (a lower bound,
assuming perfect reuse), *amplified* traffic assumes each matmul tile re-reads
its operands from DRAM (an upper bound, assuming the cache catches nothing).

| d_model | params | step | compulsory | amplified |
|---|---|---|---|---|
| 128 | 0.83 M | 11.48 ms | 52.0 GB/s | **78.8 GB/s** |
| 192 | 1.84 M | 17.72 ms | 53.0 GB/s | **88.2 GB/s** |
| 256 | 3.24 M | 32.06 ms | 38.3 GB/s | **73.0 GB/s** |
| 384 | 7.22 M | 47.84 ms | 39.8 GB/s | **87.3 GB/s** |

The measured DRAM ceiling on this machine is **79.75 GB/s**. The amplified
figure sits on it, within +/-10%, flat across a 9x parameter range. **The step
runs at the memory bandwidth limit**, and effective traffic is close to the
no-reuse bound.

So: the GPU and the CPU are on the same die and share one memory controller. A
large matmul has enough arithmetic intensity to sit above the ridge point, so
the GPU's compute advantage shows. Transformer training at these sizes is
saturating DRAM, so both processors converge on the same ceiling and the ratio
settles at a bandwidth ratio rather than a compute ratio.

One refinement this forces on section 2.4. Effective traffic tracking the
*amplified* bound means the L2 is catching very little of the cross-workgroup
re-reads, which sounds like it contradicts "the L2 already serves the reuse LDS
was going to provide". It does not: LDS staging deduplicates reads happening
*simultaneously* within one workgroup, which the cache does handle, while the
amplification counted here is different row-blocks re-reading the same B columns
at different times, which a 2 MB L2 cannot hold. Caches catch simultaneous
reuse, not distant reuse. That is also why tile *width* was the lever: wider
tiles reduce the distant re-reads that nothing else is catching.

**On an APU, the iGPU's advantage over its own CPU is capped by the shared
memory bus, not by arithmetic.** The 15.33 TFLOPS of matrix hardware is mostly
unreachable for training workloads, and buying more of it would not help. This
is the opposite of the discrete-GPU intuition, where compute is the scarce
resource and the interconnect is the thing you optimise around.

The practical consequences: below roughly 256x256 matmuls the GPU is not worth
the trouble; between there and the point where memory saturates, expect 2 to 5x
rather than the 10 to 100x that discrete-GPU experience suggests; and the way to
get more is to move fewer bytes, not to find more FLOPs. Concretely, of the
1,491 MiB an amplified step moves at d=192, matmul operand re-reads are 61.5%,
so tiling and blocking are where the remaining headroom lives.

## 4. What this cost, and what it needs

The whole stack is **2,799 lines of Python** for the runtime, kernels, autograd
and transformer, plus 1,466 more for tests, benchmarks and examples, plus the
GLSL it generates. Dependencies:
numpy, and `glslc` from the Vulkan SDK. No C compiler, no Rust, no vendor
bindings, no PyTorch. Vulkan is driven directly through `ctypes`.

Verification, all against independent numpy implementations:

| | result |
|---|---|
| WMMA 16x16x16 vs numpy | 1.9e-06 |
| matmul, 5 shapes and both backward transposes | ~1e-06 |
| f32 accumulation over K=1024 | exact |
| MLP gradients vs numpy | 4e-04 |
| MLP finite differences | 2e-03 |
| transformer loss vs numpy | 2.2e-05 |
| transformer, all 20 gradient tensors | worst 3.0e-03 |

The transformer check covers causal attention, LayerNorm, both attention
transposes, residual branches, and the embedding scatter-add through
`VK_EXT_shader_atomic_float`.

## 5. Honest limitations

- **One GPU tested.** Everything here is measured on gfx1103. The code paths
  are generic Vulkan, and the runtime degrades gracefully when
  `cooperative_matrix` is absent, but no other device has been run.
- **Batched attention matmuls remain slow** at 0.6 to 1.0 TFLOPS. The shapes
  are small and awkward (head dim 48 or 64 against a 16-wide tile granularity).
- **No split-K.** The tall-skinny weight-gradient matmuls (192 x 768 x 2048)
  have too few workgroups to fill 12 CUs. Splitting the K loop across
  workgroups is the obvious fix and is not implemented. Section 2.9 caps what
  it could win: the step is bandwidth-bound with only a ~10% non-bandwidth
  residue, so scheduling fixes cannot buy much.
- **Known traffic savings, unimplemented.** Duplicated f32 gradient buffers are
  worth ~5% of a step and f16 attention accumulation another ~4%, priced using
  the exchange rate in 2.9. Neither has been built.
- **Tile-aligned shapes only.** Ragged dimensions are handled by padding the
  allocation, not by a masked epilogue in the kernel.
- **f16 only for matmul inputs.** The hardware exposes f16 and int8 cooperative
  matrix but not bf16, so training uses f16 storage with f32 accumulation and
  f32 master weights. No dynamic loss scaling has been needed at these model
  sizes, but it would be at larger ones.
- **The occupancy pruner is undertuned**, rejecting only 2 of 84 candidates.
- **Autotuning is per shape and not free.** A full sweep is ~85 compiles plus
  measurement per shape, cached to disk afterwards. Section 2.8 shows the
  selection rule itself is not the problem, but the sweep still costs seconds.
- **CPU baselines use numpy/OpenBLAS**, which is a strong but not maximal CPU
  implementation. The transformer CPU baseline counts matmuls only.

## 6. Reproducing

```bash
python test_runtime.py && python test_kernels.py && python test_autograd.py && python test_transformer.py
```

```bash
python -m bench.roofline --json bench/full.json && python -m bench.matmul_sweep --json bench/matmul.json
```

```bash
python -m examples.mnist --epochs 4
```

```bash
python -m examples.charlm --steps 300 --tune
```

Benchmarks want an idle machine: DDR5 bandwidth is shared with the CPU, so
anything else running contaminates the result. `bench/nightly.ps1` runs the
full sweep unattended and records the machine's load alongside the numbers so
a polluted run is identifiable afterwards rather than quietly wrong.
