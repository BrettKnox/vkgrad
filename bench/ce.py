"""How much of a GPT-2-scale step is the cross-entropy kernel?

`softmax_ce` (kernels.py) is a flat elementwise kernel: one thread per row, 256
threads per workgroup. That shape is fine at the vocab sizes this repo has
mostly used (96), and possibly catastrophic at 50,257:

  * Occupancy. rows = B*T = 512 at GPT-2 small batch 2, so the dispatch is two
    workgroups on a 12-CU GPU.
  * Coalescing. Thread i walks `logits[i * stride + c]`, so the 32 lanes of a
    subgroup are simultaneously reading addresses `stride * 4` bytes apart --
    201 KiB at vocab 50,272. Every lane is its own cache line, every time.

Both are fixed by the pattern already used for attention softmax in
tkernels.py: one subgroup per row, lanes striding *within* a row, which
coalesces and raises the workgroup count by the subgroup width.

This measures the kernel in isolation against the traffic it must move, so the
gap is attributable rather than inferred. Compares vocab 96 (where the repo's
own models live) against 50,272 (GPT-2), because the whole effect is expected to
be vocab-scaling.

  python -m bench.ce
"""

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels import make_kernels, warmup  # noqa: E402
from vk import Device  # noqa: E402

BW = 79.75e9  # measured ceiling, bench/roofline.py


def time_ce(dev, K, rows, vocab, reps=9):
    """One softmax_ce dispatch at this shape, and the bytes it must move."""
    stride = (vocab + 15) // 16 * 16
    logits = dev.buffer(rows * stride * 4, "device")
    labels = dev.buffer(rows * 4, "device")
    dl16 = dev.buffer(rows * stride * 2, "device")
    dl32 = dev.buffer(rows * stride * 4, "device")
    loss = dev.buffer(rows * 4, "device")
    k = K["softmax_ce"]
    try:
        for _ in range(3):
            k([logits, labels, dl16, dl32, loss], rows, stride, vocab)
        ts = [k([logits, labels, dl16, dl32, loss], rows, stride, vocab)
              for _ in range(reps)]
        t = statistics.median(ts)
    finally:
        for b in (logits, labels, dl16, dl32, loss):
            b.destroy()

    # Three read passes over the logits (max, sum, gradient) plus both gradient
    # writes. The f32 gradient is dead in the transformer path -- allocated and
    # written, never read -- so it is counted separately.
    reads = 3 * rows * stride * 4
    w16 = rows * stride * 2
    w32 = rows * stride * 4
    total = reads + w16 + w32
    return t, total, w32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=512, help="B*T")
    args = ap.parse_args()

    dev = Device()
    warmup(dev)
    K = make_kernels(dev)
    print(f"{dev.name}, {args.rows} rows (B*T), median of 9\n")
    print(f"{'vocab':>8}{'time':>11}{'traffic':>11}{'floor':>10}"
          f"{'achieved':>12}{'of peak':>9}")
    print("-" * 63)
    rows_out = []
    try:
        for vocab in (96, 1024, 8192, 50257):
            t, total, w32 = time_ce(dev, K, args.rows, vocab)
            floor = total / BW
            gbs = total / t
            rows_out.append((vocab, t, total, floor, gbs, w32))
            print(f"{vocab:>8}{t * 1e3:>9.2f}ms{total / 1e6:>9.1f}MB"
                  f"{floor * 1e3:>8.2f}ms{gbs / 1e9:>10.2f}GB/s"
                  f"{100 * gbs / BW:>8.1f}%")
    finally:
        for k in K.values():
            k.destroy()
        dev.destroy()

    v, t, total, floor, gbs, w32 = rows_out[-1]
    print()
    print(f"  At GPT-2 vocab this one dispatch takes {t * 1e3:.1f} ms.")
    print(f"  The measured step at this shape is 535.6 ms (956 tok/s, "
          f"RESULTS.md sec 46),")
    print(f"  so it is {100 * t / 0.5356:.0f}% of a training step.")
    print(f"  Its own traffic floor is {floor * 1e3:.1f} ms, a "
          f"{t / floor:.0f}x gap.")
    print(f"  The dead f32 gradient alone is {w32 / 1e6:.1f} MB "
          f"({100 * w32 / total:.0f}% of this kernel's traffic).")


if __name__ == "__main__":
    main()
