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
| f16 WMMA peak (cooperative matrix) | 13.75 TFLOPS (83% of theoretical) |
| fp32 vector peak | 4.94 TFLOPS |
| DRAM read bandwidth | 79.75 GB/s (89% of theoretical) |
| **ridge point** | **~200 FLOP/byte** |

Two hundred FLOP per byte is a brutal ratio. For scale: an unfused elementwise
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
| dispatch, batched into one command buffer | 0.62 us |
| submit + fence | 93.61 us |

A factor of 151. The MLP training step is 15 dispatches; the transformer step
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

| shape | heuristic | autotuned | tile chosen |
|---|---|---|---|
| 192 x 768 x 2048 (`dW` qkv) | 0.78 TFLOPS | **3.26** | 64 x 32 |
| 192 x 192 x 2048 (`dW` proj) | 0.88 TFLOPS | 1.14 | 64 x 16 |
| 2048 x 768 x 192 (qkv fwd) | 2.71 TFLOPS | 2.60 | 128 x 128 |

**4.2x on the worst shape by choosing a smaller tile**, because a wide tile left
only 18 workgroups for 12 CUs.

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

## 3. The headline comparison: an iGPU against the CPU on its own die

Same silicon, same DRAM, same model. This is the question that matters for
"can people train on hardware they already own", and it is not one anybody
publishes an answer to.

**Matmul** (f16 WMMA vs numpy/OpenBLAS f32 on 8 cores):

| size | GPU TFLOPS | % of WMMA peak | CPU TFLOPS | speedup |
|---|---|---|---|---|
| 256 | 0.40 | 2.9% | 0.18 | 2.2x |
| 512 | 2.19 | 16.0% | 0.49 | 4.5x |
| 1024 | 3.11 | 22.7% | 0.38 | 8.2x |
| 2048 | 3.24 | 23.6% | 0.47 | 6.9x |
| 4096 | 3.63 | 26.4% | 0.62 | 5.8x |

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

Matmul speedup peaks at 8.2x. Transformer training sits at 2.3 to 2.9x and
**does not improve with model size** across a 9x parameter range.

That is not a bug, it is the architecture, and it is measured rather than
inferred. `bench/traffic.py` totals the bytes every dispatch in a step moves and
divides by the step time. Two bounds are reported, because neither is exactly
right alone: *compulsory* traffic touches every buffer once (a lower bound,
assuming perfect reuse), *amplified* traffic assumes each matmul tile re-reads
its operands from DRAM (an upper bound, assuming the cache catches nothing).

| d_model | params | step | compulsory | amplified |
|---|---|---|---|---|
| 128 | 0.83 M | 11.60 ms | 51.4 GB/s | **78.0 GB/s** |
| 192 | 1.84 M | 18.31 ms | 51.3 GB/s | **85.4 GB/s** |
| 256 | 3.24 M | 33.21 ms | 37.0 GB/s | **70.5 GB/s** |
| 384 | 7.22 M | 48.66 ms | 39.1 GB/s | **85.8 GB/s** |

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
memory bus, not by arithmetic.** The 13.75 TFLOPS of matrix hardware is mostly
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
  workgroups is the obvious fix and is not implemented.
- **Tile-aligned shapes only.** Ragged dimensions are handled by padding the
  allocation, not by a masked epilogue in the kernel.
- **f16 only for matmul inputs.** The hardware exposes f16 and int8 cooperative
  matrix but not bf16, so training uses f16 storage with f32 accumulation and
  f32 master weights. No dynamic loss scaling has been needed at these model
  sizes, but it would be at larger ones.
- **The occupancy pruner is undertuned**, rejecting only 2 of 84 candidates.
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
