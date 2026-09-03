"""Kernel correctness against numpy. Small shapes: safe on a busy machine."""

import numpy as np

from fallback import pick_scalar
from kernels import Matmul, autotune_matmul, matmul_reference
from vk import Device


def _run_case(dev, m, n, k, sg, wm, wn, trans_a=False, trans_b=False, seed=0):
    rng = np.random.default_rng(seed)
    # A is (K,M) when transposed, B is (N,K) when transposed.
    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)
    a = rng.standard_normal(a_shape).astype(np.float16)
    b = rng.standard_normal(b_shape).astype(np.float16)

    ba = dev.buffer(a.size * 2, "shared")
    bb = dev.buffer(b.size * 2, "shared")
    bc = dev.buffer(m * n * 4, "cached")
    # Fall back automatically when the device has no usable matrix units, so
    # the same cases cover both paths.
    mm = (Matmul(dev, sg, wm, wn, trans_a, trans_b) if dev.has_coop_matrix
          else pick_scalar(dev, m, n, k, trans_a, trans_b))
    try:
        ba.array(np.float16, a_shape)[:] = a
        bb.array(np.float16, b_shape)[:] = b
        ba.flush()
        bb.flush()
        mm(ba, bb, bc, m, n, k)
        bc.invalidate()
        got = bc.array(np.float32, (m, n)).copy()

        ref = matmul_reference(a, b, trans_a, trans_b)
        err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-6)
        assert err < 2e-3, (
            f"{m}x{n}x{k} sg={sg} wm={wm} wn={wn} ta={trans_a} tb={trans_b}: "
            f"relative error {err:.3e}\ngot\n{got[:4, :4]}\nref\n{ref[:4, :4]}")
        return err
    finally:
        mm.destroy()
        ba.destroy()
        bb.destroy()
        bc.destroy()


def test_matmul_shapes(dev):
    cases = [
        (256, 256, 256, 2, 2, 2),
        (128, 128, 128, 1, 1, 1),
        (512, 256, 128, 4, 2, 2),
        (256, 512, 256, 2, 4, 4),
        (64, 64, 512, 1, 2, 2),
    ]
    for m, n, k, sg, wm, wn in cases:
        err = _run_case(dev, m, n, k, sg, wm, wn)
        print(f"  matmul {m:4d}x{n:4d}x{k:4d} sg={sg} wm={wm} wn={wn}   "
              f"rel err {err:.2e}  OK")


def test_matmul_transposes(dev):
    """dX = dY @ W^T and dW = X^T @ dY are the whole backward pass."""
    for ta, tb, label in ((True, False, "A^T @ B  (dW = X^T dY)"),
                          (False, True, "A @ B^T  (dX = dY W^T)")):
        err = _run_case(dev, 256, 256, 256, 2, 2, 2, trans_a=ta, trans_b=tb)
        print(f"  {label:24s} rel err {err:.2e}  OK")


def test_accumulate_is_f32(dev):
    """A long K with values f16 cannot sum exactly proves f32 accumulation.

    In pure f16, summing 1024 products of ~1.0 saturates the mantissa and the
    result drifts badly. With f32 accumulation it stays exact.
    """
    m = n = 64
    k = 1024
    a = np.full((m, k), 1.0, np.float16)
    b = np.full((k, n), 1.0, np.float16)
    ba = dev.buffer(a.size * 2, "shared")
    bb = dev.buffer(b.size * 2, "shared")
    bc = dev.buffer(m * n * 4, "cached")
    mm = (Matmul(dev, 1, 2, 2) if dev.has_coop_matrix
          else pick_scalar(dev, m, n, k))
    try:
        ba.array(np.float16, a.shape)[:] = a
        bb.array(np.float16, b.shape)[:] = b
        ba.flush()
        bb.flush()
        mm(ba, bb, bc, m, n, k)
        bc.invalidate()
        got = bc.array(np.float32, (m, n))
        assert np.all(got == float(k)), (
            f"expected exactly {k}, got min {got.min()} max {got.max()}; "
            "accumulation is not f32")
        print(f"  f32 accumulation over K={k}          exact  OK")
    finally:
        mm.destroy()
        ba.destroy()
        bb.destroy()
        bc.destroy()


def test_autotune(dev):
    if not dev.has_coop_matrix:
        print("  autotune                             skipped (scalar path)")
        return
    mm, best = autotune_matmul(dev, 512, 512, 512, verbose=True, use_cache=False)
    try:
        print(f"  best config sg={best['config'][0]} wm={best['config'][1]} "
              f"wn={best['config'][2]}: {best['tflops']:.2f} TFLOPS "
              f"(evaluated {best['evaluated']}, pruned {best['pruned']})")
        assert best["tflops"] > 0
        # The tuned kernel still has to be correct, not just fast.
        sg, wm, wn = best["config"]
        err = _run_case(dev, 512, 512, 512, sg, wm, wn)
        print(f"  tuned config correctness             rel err {err:.2e}  OK")
    finally:
        mm.destroy()


def main():
    dev = Device()
    print(dev)
    try:
        test_matmul_shapes(dev)
        test_matmul_transposes(dev)
        test_accumulate_is_f32(dev)
        print("\n  autotuning 512x512x512:")
        test_autotune(dev)
        print("\nall kernel checks passed")
    finally:
        dev.destroy()


if __name__ == "__main__":
    main()
