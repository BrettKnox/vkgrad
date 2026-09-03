# Training neural networks on a consumer iGPU through Vulkan

**What this is.** A complete training stack, built from nothing but Python's
`ctypes` and a shader compiler, running forward pass, backward pass and
optimiser through Vulkan compute on an integrated GPU under Windows, where ROCm
does not officially reach.

It trains MNIST to 97.69%, trains a 10.8M-parameter transformer to a real loss
curve in twelve minutes, fits and trains **315M parameters** in 7.3 GiB, and
runs GPT-2 small's architecture at 1,117 tokens/s. It needs no matrix units
(they are worth about 6%), and its multi-device gradient exchange needs no
NCCL.

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

## 2. Ten results about this hardware

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

### 2.10 Using the model as a design tool, and checking that it was right

Sections 2.9 and 3 give a causal law and an exchange rate. The test of a model
is whether it can price something that does not exist yet and then be held to
it, so: the cheapest item it priced was eliminating the duplicated f32 gradient
buffers, at roughly 5% of a step. This section is that prediction, the change,
and the verdict.

The redundancy: every tensor whose column sum is needed for a bias gradient
already exists in f16, because the matmuls require f16 operands. The f32 copies
(`dz32`, `dqkv32`, and an f32 `dyxh` out of LayerNorm) existed only so the
reduction kernel could read f32. Making `col_sum` read f16 instead, while still
accumulating in f32, deletes those writes and halves the reduction's reads.

Predicted, by hand from the shapes at rows=2048, D=192, before implementing:

| change | bytes per layer |
|---|---|
| drop `dz32` write | 6.29 MB |
| fc1 bias reduction reads f16 | 3.15 MB |
| drop `dqkv32` write | 4.72 MB |
| qkv bias reduction reads f16 | 2.36 MB |
| LayerNorm f16 `dyxh` and `dy16`, x2 | ~3.15 MB |
| fc2 and proj bias reductions read f16 | ~1.6 MB |

~21.3 MB per layer, ~81 MiB over four layers, which at 85.1 GB/s is ~1.0 ms of
a ~21.6 ms step: **4.7%**.

Measured by alternating the old and new code in time, six rounds each, using a
git worktree at the previous commit so the two versions could not be confounded
by drift:

```
old  21.52  21.46  23.13  21.43  21.66  22.34   median 21.59 ms
new  20.33  20.53  20.40  20.37  20.41  21.10   median 20.41 ms
```

| | predicted | actual | error |
|---|---|---|---|
| bytes saved | 81 MiB | 82.9 MiB | 2.3% |
| time saved | 1.02 ms | 1.19 ms | 16% |
| step speedup | 4.7% | **5.5%** | |

The byte accounting was accurate to 2%, and the causal law converted it into a
time saving of the right sign and magnitude, under-predicting by 16% in the
same direction as section 2.9's residue. **The model priced work that did not
exist and the work delivered.**

Two things fell out that were not predicted. The optimised version is markedly
more *stable*, 20.33 to 20.53 ms versus 21.43 to 23.13, presumably because less
traffic means less contention for a memory controller shared with the CPU. And
gradient accuracy was unaffected: all 20 tensors still match the numpy
reference, worst case 2.64e-03 against 2.53e-03 before, because the reduction
still accumulates in f32 and only its inputs were narrowed.

The tuned step is now **17.52 ms**, against a traffic-only prediction of
17.35 ms, an error of **-1.0%**.

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

## 4. Vendor independence needs more than portable kernels

Everything above is about making one iGPU fast. That work is now bounded: the
machine is bandwidth-bound with a ~10% residue, the remaining priced
optimisations are worth a few percent each, and int8 turns out to buy no
arithmetic at all on RDNA3 (section 4.1). No amount of further tuning here
changes who can train models.

What does change it is the other half of the lock-in, which is rarely discussed
next to CUDA: **the collective**. Multi-GPU training runs on NCCL. NCCL is
NVIDIA-only. Portable kernels do not help if the gradient exchange still
requires one vendor's hardware, driver and interconnect, and there is no
cross-vendor equivalent.

### 4.1 int8 buys nothing on this hardware

int8 matrix units are the most widely available accelerator primitive in
consumer silicon, so int8 training would reach hardware with no CUDA and no
path to it. Measured as a pure register loop with no memory traffic:

| operands | rate | vs f16 |
|---|---|---|
| f16 x f16 -> f32 | 15.30 TOPS | 1.00x |
| f16 x f16 -> f16 | 15.33 TOPS | 1.00x |
| i8 x i8 -> i32 | 15.13 TOPS | 0.99x |
| u8 x u8 -> i32 | 15.13 TOPS | 0.99x |

Every operand type issues at the same rate. This contradicts the NVIDIA
intuition, where int8 tensor cores run at roughly 2x fp16. On RDNA3's WMMA int8
offers only halved operand bytes, capping it at ~27% of a step before
quantisation overhead and backward-pass range problems. Recorded so nobody else
spends a week on it.

### 4.2 A collective that needs no vendor

`VK_EXT_external_memory_host` lets one ordinary host allocation be imported by
several independent `VkDevice`s at once. Plain system RAM becomes a shared arena
between GPUs. Nothing about that requires the devices to share a vendor, a
driver, or an interconnect.

That is enough to build DDP without NCCL:

- weights are **replicated**, one copy per device, and never exchanged
- each device computes gradients for its own shard of the batch
- each device **atomically adds** its gradients into the shared arena
- every device applies the identical optimiser step from that arena

Replicas start identical and apply identical updates, so they stay identical and
only gradients ever cross. The arena is a flat concatenation of every
parameter's gradient, so the whole collective is two kernels: one zeroing pass
and one atomic add per device.

On unified memory the arena is the same DRAM the GPU already reads, so the
all-reduce degenerates into several workers adding into the same bytes. There is
nothing to transfer. On a discrete card it is the host side of that card's PCIe
path.

`python -m examples.ddp_mnist --devices N` trains MNIST this way:

| devices | final loss | weight drift between replicas | bytes transferred |
|---|---|---|---|
| 1 | 0.3260 | | |
| 2 | 0.3237 | **0.000e+00** | **0** |
| 4 | 0.3247 | **0.000e+00** | **0** |

Replicas are **bit-identical** after training, which is the strong form of the
claim: nothing but gradients crossed, and the exchange is exact rather than
approximately right.

### 4.3 What this does and does not show

It shows the mechanism is real and the arithmetic is exact, across an arbitrary
number of independent devices, with a collective that has no vendor-specific
component anywhere in it.

It does not show two vendors. These runs use several `VkDevice`s created from
the one physical GPU this machine has, so they share hardware and there is no
wall-clock gain: 0.4 s, 0.8 s, 1.5 s for 1, 2 and 4 devices, since the same
silicon does more total work plus per-device overhead. Proving the cross-vendor
case needs two cards from different vendors in one machine, which I do not have.

The reason it is worth reporting anyway: every part of this path is core Vulkan
or a widely implemented extension, and `VK_KHR_cooperative_matrix` is exposed by
NVIDIA, AMD and Intel. A gaming PC with an NVIDIA card and an AMD iGPU has two
usable training devices today and no software that will use both. This is the
piece that was missing, and it is 120 lines.

## 5. You do not need matrix units to train

Everything up to here ran on `VK_KHR_cooperative_matrix`. That extension needs
AMD RDNA3+, NVIDIA Turing+, or Intel Arc, which excludes Polaris, Vega, Pascal,
Maxwell, every Intel integrated GPU before Arc, Adreno, Mali and the Raspberry
Pi. Most GPUs that exist. A stack that only runs on recent high-end silicon is
not a stack that makes commodity hardware useful, so the fallback matters more
than any of the tuning above.

`fallback.py` is the classic LDS-tiled, register-blocked scalar matmul,
generated the same way as the cooperative-matrix one, needing nothing beyond
Vulkan 1.1 and 16-bit storage (and able to drop to fp32 operands where even
that is missing). It matches the numpy reference **exactly**, 0.00e+00, since
f16 operands with fp32 accumulation reproduce the reference arithmetic
bit-for-bit, and it implements both backward transposes and the batched form
attention needs.

The expectation, from the peak numbers, was that dropping the matrix units
would cost about 2.9x: 15.33 TFLOPS of WMMA against 5.30 TFLOPS of fp32 vector
throughput. Measured, interleaved, median of nine:

| shape | cooperative matrix | scalar | ratio |
|---|---|---|---|
| 512x512x512 | 1.12 T | 0.66 T | 1.69x |
| 1024x1024x1024 | 2.77 T | 2.27 T | **1.22x** |
| 2048x2048x2048 | 4.21 T | 1.92 T | 2.19x |
| 2048x768x192 | 2.92 T | 1.62 T | 1.80x |

And on actual training, with `VKGRAD_NO_COOPMAT=1` forcing the scalar path:

| workload | matrix units | scalar | cost |
|---|---|---|---|
| MNIST MLP step | 5.39x CPU | 5.07x CPU | **~6%** |
| transformer step | 20.58 ms | 21.81 ms | **~6%** |
| MNIST accuracy | 97.69% | 97.39% | none |
| gradient checks | pass | pass | none |

**Matrix units are worth about 6% on a real training step here.** Not 2.9x.
(That 6% was measured across separate runs and is corrected below: interleaved,
the range is -5% to +18% with a median near 8%.)

The reason is the whole thesis of this document. A bandwidth-bound machine
leaves the matrix units idle most of the time, so removing them removes
capacity that was not being used. Section 2.9 measured that directly: the step
runs at the DRAM limit with a ~10% non-bandwidth residue, and matrix throughput
lives entirely inside that residue.

### Audit: re-measured interleaved

The 6% figure above came from comparing two separate `charlm` runs, which is the
cross-run error this document has now made four times. Re-measured properly, by
constructing both a cooperative-matrix and a scalar context in one process and
alternating their steps, median of 7:

| model | coopmat | scalar | matrix units worth |
|---|---|---|---|
| 1.8M (192d x4) | 20.9 ms | 22.0 ms | +5.4% |
| 10.8M (384d x6) | 90.5 ms | 106.6 ms | **+17.8%** |
| 25.4M (512d x8) | 124.5 ms | 118.7 ms | **-4.7%** |
| 85.4M (768d x12) | 195.7 ms | 220.8 ms | +12.8% |
| 162M (768d x12, vocab 50k) | 255.5 ms | 275.4 ms | +7.8% |

**Somewhere between -5% and +18%, median around 8%, with no clean trend.** On one
shape the scalar path is faster, because `pick_config` and `pick_scalar` choose
tiles independently and some shapes suit the scalar tiling better.

So "about 6%" was too precise and slightly low. The correct statement is that
matrix units are worth a modest, shape-dependent amount, occasionally nothing.

Note also the gap between levels. On isolated matmuls cooperative matrix is 1.22x
to 2.19x faster; end to end it is worth under 20%. That difference is the whole
thesis of this document restated: on a bandwidth-bound machine, matmul speedups
do not become step speedups.

The conclusion that matters is unchanged, and if anything better supported by
having five model sizes rather than one: **training without matrix units costs
well under the 2.9x the peak FLOPS ratio implies**, so the hardware floor for
this work is far lower than the marketing suggests.

This is the most consequential result here, and it is worth stating plainly
because it cuts against how the hardware is marketed. Tensor cores are sold as
the thing that makes AI possible. For *training* on *memory-bound consumer
hardware*, they are close to a rounding error. What matters is memory
bandwidth, and a 2016 GPU with no matrix units at all is far more competitive
for this than its spec sheet suggests.

The practical consequence: vkgrad's hardware target is not "RDNA3, Turing, Arc".
It is any GPU with a Vulkan 1.1 driver. The runtime detects
`VK_KHR_cooperative_matrix` and uses it when present, falls back automatically
when absent, and trains either way.

## 6. Portability was assumed four times and was wrong four times

Sections 4 and 5 claim this runs on hardware other than the machine it was
written on. That claim was made three separate times before it was true, and
each time the checking found something that would have broken on real hardware.
The four bugs are worth listing together, because the pattern matters more than
any of them.

**Unconditional feature requests.** `vkCreateDevice` was asked for 15 features
without checking support, and Vulkan fails device creation outright if any is
unsupported. On a Vulkan 1.1 GPU (GTX 1060, RX 580, Intel HD 620) nothing would
have run. Seven of the fifteen were not used by any kernel; they were requested
because they looked useful. The runtime now queries
`vkGetPhysicalDeviceFeatures2` and asks only for the intersection.

**Hardcoded subgroup width.** The row-reduction kernels stride a row across one
subgroup's lanes, so the stride must equal the real subgroup width. It was
fixed at 32 next to a `requiredSubgroupSize=32` request that AMD happens to
grant. Mismatching them does not crash: on a 192-wide row, a width-64 subgroup
with a stride of 32 gives **84% error**, silently. Intel's subgroups are
commonly 8, 16 or 32; older AMD is 64; devices without `subgroupSizeControl`
ignore the request entirely. All of them would have trained on quietly corrupt
gradients.

**Hardcoded dispatch multiplier.** Found while fixing the previous one. The row
kernels were dispatched as `rows * 32`, so at width 64 only half the rows were
processed.

**Discrete-only memory selection.** `find_memory_type` required the `device`
role to be `DEVICE_LOCAL` and *not* `HOST_VISIBLE`, which describes a card with
private VRAM. A fully unified device (Intel integrated, Apple via MoltenVK,
Mali, Adreno) exposes no such memory, and since every model buffer defaults to
that role, the first allocation would have raised.

None of these were found by reading the code. All four were found by building a
way to *run* the other configuration. There are now three switches for that, and
all five test suites pass under each and under all three at once:

| configuration | approximates | result |
|---|---|---|
| baseline | RDNA3: matrix units, wave32, split heaps | 5/5 |
| `VKGRAD_NO_COOPMAT=1` | Vega, Pascal, Intel HD: no matrix units | 5/5 |
| `VKGRAD_NATIVE_SUBGROUP=1` | wave64, no subgroup size control | 5/5 |
| `VKGRAD_UMA=1` | Intel, Mali, Adreno: unified memory | 5/5 |
| all three | oldest and most constrained tier | 5/5 |

`check_device.py` reports what a given GPU supports and what each missing piece
costs, so unsupported hardware produces an explanation rather than a Vulkan
error code.

This still is not the same as running on another vendor's silicon. It does mean
each of the four wrong assumptions now has a configuration that would catch it
again, and that a fifth has somewhere to be caught.

## 7. What can actually be trained on hardware people own

The performance sections above are about ratios. This one is about what a person
with a laptop can actually do, which is the only question that matters for the
premise.

### The frontier

Removing an artificial limit came first: `Kernel` allocated a single descriptor
pool of 64 sets, and a recorded step binds the same kernel once per tensor, so a
6-layer model failed at construction. Nothing about the hardware required that.
Pools are chained now.

On a Radeon 780M with 12 CUs and unified DDR5-5600:

| params | step | tokens/s | memory |
|---|---|---|---|
| 1.8 M | 20.2 ms | 101,593 | 0.33 GiB |
| 10.8 M | 89.1 ms | 22,984 | 1.15 GiB |
| 25.4 M | 225.5 ms | 9,083 | 2.13 GiB |
| 85.4 M | 801.1 ms | 2,556 | 5.26 GiB |
| **315.4 M** | 803.2 ms | 637 | **7.31 GiB** |

**315 million parameters trains on an integrated GPU**, in 7.3 of 11.8 GiB. Full
forward, backward and AdamW, gradients verified against numpy.

### A real run, not an extrapolation

Step times are not throughput. A 12-minute run of the 10.8M model with real data
loading, loss readback, held-out validation and checkpointing:

```
7,518 steps, 15.4M tokens, 21,382 tokens/s sustained
loss 4.7821 -> 0.8717,  validation 0.9123
```

No thermal decay: instantaneous throughput over the final six minutes (21,672)
is above the cumulative average. The step-time extrapolation was about 7%
optimistic.

Sampling from that checkpoint, primed with real corpus text:

```python
def filter(pad, pad, pad, pad, pad, pad, pad, pad, pad, pad)
        self.assertEqual(self.rowcode, pad, pad)
        self.rowcode = rowcode

    def test_pad(self):
        self.rowcode = self.rowcode
```

Block structure, consistent indentation, `self.` access, the `test_` convention
and `assertEqual`, correctly inferring from context that it was inside a
unittest file. Repetitive, because this is 7% of Chinchilla-optimal. Twelve
minutes, no CUDA, no ROCm, no downloaded dataset.

### Fine-tuning scale

Most people would fine-tune rather than train from scratch. GPT-2 small's
architecture (768 d_model, 12 layers, 12 heads, vocab 50257, learned positions,
pre-LayerNorm, GELU) is what this already builds:

**1,117 tokens/s in 3.92 GiB.** That is 1M tokens in 15 minutes, 10M in 2.5
hours, 50M in 12.4 hours. Domain adaptation on a laptop is a matter of hours.

(Measured with random weights. Loading real GPT-2 checkpoints would also need
the weight file, a BPE tokenizer, and embedding/head weight tying, none of which
is implemented. The throughput and memory are what a fine-tune would see.)

### Bigger batches are slower, which inverts standard practice

| batch | tokens/s |
|---|---|
| 2 | 1,117 |
| 4 | 844 |
| 8 | 668 |

On discrete NVIDIA hardware you raise batch size to amortise weight reads and
fill idle compute. Neither applies here: there is no idle compute, because the
machine is bandwidth-bound, and activation traffic grows linearly with batch
while the fixed weight traffic was never the bottleneck. Tuning intuition ported
from a discrete card gets this exactly backwards, and the failure mode looks
like broken hardware rather than a wrong assumption.

## 8. What this cost, and what it needs

The whole stack is **3,542 lines of Python** for the runtime, kernels,
autograd, transformer, scalar fallback and multi-device collective, plus the
rest of the 6,612 total for tests, benchmarks and examples, plus the GLSL it
generates. Dependencies:
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

## 9. Honest limitations

- **One GPU tested.** Everything here is measured on gfx1103. Five simulated
  configurations (section 6) cover the hardware classes claimed, and all five
  test suites pass under each, but no other physical device has been run. Four
  portability assumptions were wrong before those simulations existed, which is
  the best available evidence that a fifth is still hiding.
- **Seven corrected numbers.** Cold clocks, evaluation order, cross-run
  comparison, four portability assumptions and a too-short sample each produced
  a confident figure that later measurement overturned. The corrections are
  recorded in place rather than overwritten. Treat any number here that is not
  attached to a described measurement method with suspicion.
- **Batched attention matmuls remain slow** at 0.6 to 1.0 TFLOPS. The shapes
  are small and awkward (head dim 48 or 64 against a 16-wide tile granularity).
- **No split-K.** The tall-skinny weight-gradient matmuls (192 x 768 x 2048)
  have too few workgroups to fill 12 CUs. Splitting the K loop across
  workgroups is the obvious fix and is not implemented. Section 2.9 caps what
  it could win: the step is bandwidth-bound with only a ~10% non-bandwidth
  residue, so scheduling fixes cannot buy much.
- **One priced saving remains unimplemented.** f16 accumulation for the
  attention score and gradient tensors is worth ~4% by the section 2.9
  exchange rate. The duplicated f32 gradient buffers were the other item and
  have been removed (section 2.10). A leftover `dlog32` write, now unused by
  the transformer, is worth a further 0.05% and is not worth the churn.
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

## 10. Reproducing

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
