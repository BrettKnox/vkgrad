# vkgrad

A training stack for GPUs that aren't NVIDIA, built on Vulkan compute.

The point is not another inference runtime. Vulkan cooperative matrix (matrix
cores) is already used for inference by llama.cpp and ncnn. Nothing runs a
**backward pass** on it. This does.

Target: integrated and consumer GPUs people already own. Developed against an
AMD Radeon 780M (RDNA3 iGPU, 12 CU) under Windows, where ROCm does not
officially reach.

## State

| phase | status |
|---|---|
| 0. toolchain | done. Vulkan SDK, `glslc` compiles `GL_KHR_cooperative_matrix` |
| 1. runtime + zero-copy allocator | done, verified |
| 2. WMMA matmul + autotuner | done, verified. Forward and both backward transposes |
| 3. autograd + training | done. MNIST 97.69%, and a 1.84M param transformer trains |

Write-up: [RESEARCH.md](RESEARCH.md). Raw measurements: [bench/RESULTS.md](bench/RESULTS.md).

Headline: **5.1x faster than the CPU sharing the same die and the same DRAM**
on a full MNIST training step (forward, backward, AdamW), 97.69% test accuracy.
A 1.84M parameter transformer trains at 3.14x the CPU, with every gradient
tensor matching an independent numpy model.

## Layout

```
vk.py              Vulkan compute runtime via ctypes. No SDK bindings, no C compiler.
compile.py         GLSL -> SPIR-V via glslc, content-hashed disk cache
kernels.py         matmul generators (direct + LDS), fused elementwise kernels, autotuner
autograd.py        tape, Linear/MLP, fused AdamW, recorded TrainStep
tkernels.py        LayerNorm, causal attention softmax, head permutes, embeddings
transformer.py     decoder-only transformer: attention, blocks, GPT
test_runtime.py    runtime self-checks
test_kernels.py    kernel correctness vs numpy
test_autograd.py   gradient checks vs numpy and finite differences
test_transformer.py  full transformer fwd+bwd vs an independent numpy model
examples/mnist.py  trains an MLP, races it against the same model on the CPU
examples/charlm.py trains a char-level transformer on local text
bench/roofline.py  bandwidth, dispatch overhead, fp32 and WMMA ceilings
bench/matmul_sweep.py  matmul throughput vs the CPU baseline
bench/nightly.ps1  unattended full sweep, records machine load alongside results
```

Dependencies: numpy, and the Vulkan SDK for `glslc`. That's it.

## Running

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

**Row kernels get one subgroup per row, not one thread per row.** The obvious
version makes adjacent lanes read a whole row apart, so every access is its own
cache line. This cost 31 ms of a 47 ms transformer step before it was fixed.

**The iGPU's lead over its own CPU is capped by the shared memory bus.** Matmul
alone hits ~5x, but full transformer training sits at 2.3-2.9x and does not
improve with model size: both processors share one memory controller, and
training at these sizes is memory-bound end to end. Expect 2-5x here, not the
10-100x that discrete-GPU experience suggests.
