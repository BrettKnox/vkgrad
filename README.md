# vkgrad

A training stack for GPUs that aren't NVIDIA, built on Vulkan compute.

The point is not another inference runtime. Vulkan cooperative matrix (matrix
cores) is already used for inference by llama.cpp and ncnn. Nothing runs a
**backward pass** on it. This does.

Target: any GPU with a Vulkan 1.1 driver. Cooperative matrix is used when the
device has it and a scalar fallback runs when it does not, which costs about 6%
on a real training step rather than the 2.9x the peak numbers imply. Developed
against an AMD Radeon 780M (RDNA3 iGPU, 12 CU) under Windows, where ROCm does
not officially reach.

## State

| phase | status |
|---|---|
| 0. toolchain | done. Vulkan SDK, `glslc` compiles `GL_KHR_cooperative_matrix` |
| 1. runtime + zero-copy allocator | done, verified |
| 2. WMMA matmul + autotuner | done, verified. Forward and both backward transposes |
| 3. autograd + training | done. MNIST 97.69%, and a 1.84M param transformer trains |
| 4. multi-device without NCCL | mechanism verified: DDP over shared host memory, 0 bytes transferred |
| 5. runs without matrix units | scalar fallback, exact vs numpy, ~6% slower on a real step |

Write-up: [RESEARCH.md](RESEARCH.md). Raw measurements: [bench/RESULTS.md](bench/RESULTS.md).

Headline: **a 315M-parameter transformer trains on an integrated laptop GPU**,
in 7.3 GiB, with no CUDA and no ROCm. At Chinchilla-optimal token counts that
means a 10M model in 2.6 hours and a 25M model overnight. Full forward, backward
and AdamW, with every gradient verified against numpy.

On MNIST it runs 5.1x faster than the CPU sharing the same die and the same
DRAM, reaching 97.69% test accuracy.

## Layout

```
vk.py              Vulkan compute runtime via ctypes. No SDK bindings, no C compiler.
compile.py         GLSL -> SPIR-V via glslc, content-hashed disk cache
kernels.py         matmul generators (direct + LDS), fused elementwise kernels, autotuner
autograd.py        tape, Linear/MLP, fused AdamW, recorded TrainStep
tkernels.py        LayerNorm, causal attention softmax, head permutes, embeddings
transformer.py     decoder-only transformer: attention, blocks, GPT
fallback.py        scalar tiled matmul for GPUs with no matrix units
dataparallel.py    DDP over shared host memory: replicated weights, atomic gradient arena
test_runtime.py    runtime self-checks
test_kernels.py    kernel correctness vs numpy
test_autograd.py   gradient checks vs numpy and finite differences
test_transformer.py  full transformer fwd+bwd vs an independent numpy model
test_shared.py     host memory imported by several independent VkDevices
test_accum.py      gradient accumulation equals one large batch
check_device.py    will vkgrad run on this GPU, and what will it get
examples/mnist.py  trains an MLP, races it against the same model on the CPU
examples/charlm.py trains a char-level transformer on local text
examples/ddp_mnist.py  data-parallel MNIST across N devices, no NCCL
examples/train_lm.py   train a char LM for a wall-clock budget, resumable
examples/sample_lm.py  generate text from a checkpoint
bench/roofline.py  bandwidth, dispatch overhead, fp32 and WMMA ceilings
bench/matmul_sweep.py  matmul throughput vs the CPU baseline
bench/nightly.ps1  unattended full sweep, records machine load alongside results
```

Dependencies: numpy, and the Vulkan SDK for `glslc`. That's it.

## Running

Start here on any new machine:

```bash
python check_device.py
```

```bash
python test_runtime.py
```

```bash
python test_kernels.py
```

Benchmarks want an idle machine, because DDR5 bandwidth is shared with the CPU:

```bash
python -m bench.roofline --json bench/full.json
```

```bash
python -m bench.matmul_sweep --json bench/matmul.json
```

Add `--quick` to either for a small, safe run on a busy machine.

Three switches simulate hardware this machine is not, so the portable paths get
exercised anyway. All five suites pass under each and under all three at once:

```bash
VKGRAD_NO_COOPMAT=1 python test_transformer.py      # no matrix units
```

```bash
VKGRAD_NATIVE_SUBGROUP=1 python test_transformer.py # wave64, no size control
```

```bash
VKGRAD_UMA=1 python test_transformer.py             # no private VRAM
```

Set `VKGRAD_VALIDATE=1` to enable Vulkan validation layers. Do this while
developing and not while benchmarking: they cost ~30% on submit.

**Benchmarks must warm up, and comparisons must interleave.** GPU clocks ramp
2.7x over the first ~30 dispatches, so anything measured cold is measuring power
management rather than the kernel. Every bench script calls `kernels.warmup()`.
Residual drift after warmup is still correlated with time, so the autotuner
measures candidates round-robin and every A/B comparison alternates the two
sides within one run. Both mistakes produced wrong published numbers here before
they were caught.

## Three design choices worth knowing

**Tensors are numpy arrays and GPU buffers at the same time.** Host-visible
memory is mapped once and exposed through `np.frombuffer`, so there is no
`.to(device)` and no transfer. But the measurements said this is not universally
right: host-visible memory runs at 85% of device-local bandwidth, so weights
that live on the GPU for the whole run are better off staged once into
device-local. The allocator picks by role (`shared` / `device` / `cached`), and
`Device.KINDS` is where that decision lives.

**Bytes moved, not FLOPs.** The ridge point on this hardware is ~224 FLOP/byte.
Effectively nothing in a small training workload is compute-bound, so fusion and
tile width dominate, and instruction-level tuning is second-order.

**Record once, replay.** A batched dispatch costs 1.7 us; a submit with a fence
costs 127 us. A training step is one pre-recorded command buffer (15 dispatches
for the MLP, 278 for the transformer), replayed each step. Measured effect on
the MLP: 2.94 ms per step becomes 0.66 ms. Anything that changes per step lives
in mapped memory rather than push constants, which are baked in at record time.

**Gradient accumulation pays only for large models.** At GPT-2 scale, near the
memory ceiling, eight microbatches of 1 are 54% faster than one batch of 8 and
use half the memory. On a 10.8M model it is 32% *slower*, because the fixed
per-microbatch overhead dominates. Use it when the large-batch configuration is
close to the memory limit.

**Row kernels get one subgroup per row, not one thread per row.** The obvious
version makes adjacent lanes read a whole row apart, so every access is its own
cache line. This cost 31 ms of a 47 ms transformer step before it was fixed.

**The iGPU's lead over its own CPU is capped by the shared memory bus.** Matmul
alone hits ~5x, but full transformer training sits at 2.3-2.9x and does not
improve with model size: both processors share one memory controller, and
training at these sizes is memory-bound end to end. Expect 2-5x here, not the
10-100x that discrete-GPU experience suggests.
