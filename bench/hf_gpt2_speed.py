"""Real GPT-2 weights on vkgrad: fine-tune throughput and per-token forward cost.

  python bench/hf_gpt2_speed.py train --seq 512 --batch 2
  python bench/hf_gpt2_speed.py train --seq 1024 --batch 1 --accum 2
  python bench/hf_gpt2_speed.py latency --seq 512
  add --untied for a separate head initialised from wte

Data: GPT-2 BPE over the first 2 MiB of data/pycorpus.npy (local Python
sources), random windows. Memory is reported two ways: the sum of vkgrad's own
buffer allocations, and Windows' GPU Process Memory counters for this process
sampled after the timed loop.
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hf_gpt2  # noqa: E402
from autograd import AdamW  # noqa: E402
from dataparallel import GradAccum  # noqa: E402
from kernels import warmup  # noqa: E402
from transformer import TCtx  # noqa: E402
from vk import Device  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def memory(ctx):
    pid = os.getpid()
    cmd = (f"(Get-Counter '\\GPU Process Memory(pid_{pid}_*)\\Shared Usage',"
           f"'\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage').CounterSamples | "
           "ForEach-Object { $_.Path + '=' + $_.CookedValue }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                         capture_output=True, text=True).stdout
    shared = dedicated = 0.0
    for line in out.splitlines():
        path, _, val = line.rpartition("=")
        if path.lower().endswith("shared usage"):
            shared += float(val)
        elif path.lower().endswith("dedicated usage"):
            dedicated += float(val)
    alloc = sum(b.size for b in ctx._owned)
    return (f"vkgrad buffers {alloc / 2**30:.2f} GiB; GPU process memory "
            f"shared {shared / 2**30:.2f} GiB, dedicated {dedicated / 2**30:.2f} GiB")


def train(args, dev, ctx):
    enc = hf_gpt2.encoding()
    raw = np.load(os.path.join(ROOT, "data", "pycorpus.npy"))[:2 << 20].tobytes()
    tokens = np.array(enc.encode_ordinary(raw.decode("utf-8", "replace")), np.uint32)
    B, T, A = args.batch, args.seq, args.accum
    model = hf_gpt2.build(ctx, B, T, tie=not args.untied)
    params = list(model.params())
    opt = AdamW(ctx, params, lr=args.lr, wd=0.01)
    rows = B * T
    idb, tgb = ctx.buf(rows * 4, "shared"), ctx.buf(rows * 4, "shared")
    idv, tgv = idb.array(np.uint32, (rows,)), tgb.array(np.uint32, (rows,))
    rng = np.random.default_rng(0)

    def fill():
        idx = rng.integers(0, len(tokens) - T - 1, B)[:, None] + np.arange(T)
        idv[:] = tokens[idx].reshape(-1)
        tgv[:] = tokens[idx + 1].reshape(-1)
        idb.flush()
        tgb.flush()

    if A == 1:
        g = dev.graph("train")
        model.record(idb, tgb, g)
        opt.record(g)
        g.finish()

        def step():
            fill()
            opt.advance()
            g.submit()
            return model.read_loss()
    else:
        ga = GradAccum(ctx, params, accum=A)
        gz = dev.graph("zero")
        ga.record_zero(gz)
        gz.finish()
        gz.submit()
        gm = dev.graph("micro")
        model.record(idb, tgb, gm)
        ga.record_push(gm)
        gm.finish()
        gs = dev.graph("step")
        ga.record_apply(gs, opt)
        ga.record_zero(gs)
        gs.finish()

        def step():
            tot = 0.0
            for _ in range(A):
                fill()
                gm.submit()
                tot += model.read_loss()
            opt.advance()
            gs.submit()
            return tot / A

    print(f"seq {T} batch {B} accum {A} tie {not args.untied}: "
          f"{model.n_params():,} params, corpus {len(tokens):,} tokens")
    losses = [step() for _ in range(3)]  # untimed
    times = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        ts = time.perf_counter()
        losses.append(step())
        times.append(time.perf_counter() - ts)
    el = time.perf_counter() - t0
    tok = len(times) * rows * A
    print(f"  {len(times)} steps, {tok:,} tokens in {el:.1f} s = {tok / el:,.0f} tokens/s")
    print(f"  step median {np.median(times) * 1e3:.1f} ms, "
          f"min {min(times) * 1e3:.1f}, max {max(times) * 1e3:.1f}")
    print(f"  loss: first 3 steps {np.mean(losses[:3]):.3f}, last 10 {np.mean(losses[-10:]):.3f}")
    print(f"  {memory(ctx)}")


def latency(args, dev, ctx):
    model = hf_gpt2.build(ctx, 1, args.seq, tie=not args.untied)
    fwd = hf_gpt2.Forward(model)
    ids = np.random.default_rng(0).integers(0, model.vocab, (1, args.seq))
    for _ in range(3):
        fwd(ids, last=True)
    sub, cop, tot = [], [], []
    for _ in range(args.runs):
        t = time.perf_counter()
        int(fwd(ids, last=True)[0].argmax())
        tot.append(time.perf_counter() - t)
        sub.append(fwd.submit_s)
        cop.append(fwd.copy_s)
    ms = lambda a: f"median {np.median(a) * 1e3:.1f} ms (min {min(a) * 1e3:.1f}, max {max(a) * 1e3:.1f})"
    print(f"batch 1, T={args.seq}, tie {not args.untied}, {args.runs} runs")
    print(f"  forward submit     {ms(sub)}")
    print(f"  logits copy        {ms(cop)}  ({model.rows * model.pad_vocab * 4 / 2**20:.0f} MiB, whole buffer)")
    print(f"  per token, total   {ms(tot)} = {1 / np.median(tot):.2f} tokens/s")
    print(f"  {memory(ctx)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("train", "latency"))
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--untied", action="store_true")
    args = ap.parse_args()
    dev = Device()
    print(dev)
    warmup(dev)
    ctx = TCtx(dev)
    try:
        (train if args.mode == "train" else latency)(args, dev, ctx)
    finally:
        ctx.destroy()
        dev.destroy()


if __name__ == "__main__":
    main()
