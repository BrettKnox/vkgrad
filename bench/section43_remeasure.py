"""Re-measure RESULTS.md section 43: GPT-2 small's shape (768d x12, vocab 50257).

Section 43's numbers came from a script that is not in the repo, and a fine-tune
with loaded weights later measured 640-693 tokens/s against its 949 at seq 512
batch 2. Every measurement here is its own clean process; CPU load is sampled
before and during each one; results go to bench/section43_remeasure.json after
every run, so a crash keeps what was measured.

  python bench/section43_remeasure.py plan --phase preflight --worktree DIR
  python bench/section43_remeasure.py plan --phase main --worktree DIR
  python bench/section43_remeasure.py one --seq 512 --batch 2 [--tie] [--gpt2] [--root DIR]

DIR is a checkout of 40ad1b8 (before the qkv bias fix), for arm E.
"""

import argparse
import ctypes as C
import datetime
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.abspath(__file__)
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "bench", "section43_remeasure.json")
BUDGET_S = 58 * 60  # GPU-process wall time across every run, preflight included

# Section 43 as published (step ms, tokens/s, GiB), and section 46's clean-process re-run.
SECTION43 = {"256x2": (458.5, 1117, 3.92), "256x4": (1213.7, 844, 5.12),
             "256x8": (3064.7, 668, 7.51), "512x2": (1078.7, 949, 5.54),
             "1024x2": (3181.2, 644, 10.05)}
SECTION46 = {"256x2": 956, "256x4": 965, "256x8": 669}

ARMS = {
    "A": ("random weights, untied head (section 43's model), working tree", dict(tie=False, gpt2=False, tree="wt")),
    "B": ("loaded GPT-2 weights, tied head, working tree", dict(tie=True, gpt2=True, tree="wt")),
    "C": ("random weights, tied head, working tree", dict(tie=True, gpt2=False, tree="wt")),
    "D": ("loaded GPT-2 weights, untied head, working tree", dict(tie=False, gpt2=True, tree="wt")),
    "E": ("random weights, untied head, 40ad1b8 (before the qkv bias fix)", dict(tie=False, gpt2=False, tree="40ad1b8")),
    "F": ("bench/hf_gpt2_speed.py verbatim (loaded weights, tied), working tree", dict(verbatim=True, tree="wt")),
}


# ---------------------------------------------------------------- one measurement

def one(a):
    sys.path.insert(0, a.root)
    import numpy as np
    from autograd import AdamW
    from kernels import warmup
    from transformer import GPT, TCtx
    from vk import Device

    t_start = time.perf_counter()
    dev = Device()
    warmup(dev)
    ctx = TCtx(dev)
    B, T = a.batch, a.seq
    rows = B * T
    rng = np.random.default_rng(0)
    try:
        if a.gpt2:
            import hf_gpt2
            enc = hf_gpt2.encoding()
            raw = np.load(os.path.join(a.root, "data", "pycorpus.npy"))[:2 << 20].tobytes()
            tokens = np.array(enc.encode_ordinary(raw.decode("utf-8", "replace")), np.uint32)
            model = hf_gpt2.build(ctx, B, T, tie=a.tie)
        else:
            tokens = rng.integers(0, 50257, 1 << 20).astype(np.uint32)
            kw = {"tie": True} if a.tie else {}  # 40ad1b8's GPT has no tie argument
            model = GPT(ctx, B, T, 768, 12, 12, 50257, **kw)
        opt = AdamW(ctx, list(model.params()), lr=5e-5, wd=0.01)
        idb, tgb = ctx.buf(rows * 4, "shared"), ctx.buf(rows * 4, "shared")
        idv, tgv = idb.array(np.uint32, (rows,)), tgb.array(np.uint32, (rows,))
        g = dev.graph("train")
        model.record(idb, tgb, g)
        opt.record(g)
        g.finish()

        # The same step as bench/hf_gpt2_speed.py, with the submit time kept.
        def step():
            idx = rng.integers(0, len(tokens) - T - 1, B)[:, None] + np.arange(T)
            idv[:] = tokens[idx].reshape(-1)
            tgv[:] = tokens[idx + 1].reshape(-1)
            idb.flush()
            tgb.flush()
            opt.advance()
            sub = g.submit()
            return model.read_loss(), sub

        losses = [step()[0] for _ in range(3)]  # untimed
        setup_s = time.perf_counter() - t_start
        wall, subs, starts = [], [], []
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < a.seconds:
            ts = time.perf_counter()
            loss, sub = step()
            wall.append(time.perf_counter() - ts)
            subs.append(sub)
            starts.append(ts - t0)
            losses.append(loss)
        el = time.perf_counter() - t0
        half = [w for s, w in zip(starts, wall) if s >= a.seconds / 2]
        print("RESULT " + json.dumps(dict(
            n_params=model.n_params(), alloc_gib=round(sum(b.size for b in ctx._owned) / 2**30, 3),
            setup_s=round(setup_s, 2), steps=len(wall), tokens=len(wall) * rows, elapsed_s=round(el, 3),
            tok_s_wall=round(len(wall) * rows / el, 1),
            tok_s_second_half=round(len(half) * rows / sum(half), 1) if half else None,
            step_ms_median=round(statistics.median(wall) * 1e3, 2),
            step_ms_min=round(min(wall) * 1e3, 2), step_ms_max=round(max(wall) * 1e3, 2),
            submit_ms_median=round(statistics.median(subs) * 1e3, 2),
            submit_ms_min=round(min(subs) * 1e3, 2),
            tok_s_from_median_step=round(rows / statistics.median(wall), 1),
            tok_s_from_best_submit=round(rows / min(subs), 1),
            loss_first3=round(float(np.mean(losses[:3])), 4), loss_last10=round(float(np.mean(losses[-10:])), 4),
            step_ms=[round(w * 1e3, 1) for w in wall], submit_ms=[round(s * 1e3, 1) for s in subs])))
    finally:
        ctx.destroy()
        dev.destroy()


# ---------------------------------------------------------------- load sampling

k32 = C.windll.kernel32


def sys_times():
    idle, kern, user = C.c_ulonglong(), C.c_ulonglong(), C.c_ulonglong()
    k32.GetSystemTimes(C.byref(idle), C.byref(kern), C.byref(user))
    return idle.value, kern.value + user.value  # 100 ns; kernel time includes idle


def busy_pct(t0, t1):
    di, dt = t1[0] - t0[0], t1[1] - t0[1]
    return 100.0 * (dt - di) / dt if dt else 0.0


def proc_cpu_s(handle):
    c, e, k, u = (C.c_ulonglong() for _ in range(4))
    ok = k32.GetProcessTimes(C.c_void_p(int(handle)), C.byref(c), C.byref(e), C.byref(k), C.byref(u))
    return (k.value + u.value) / 1e7 if ok else None


def processes():
    cmd = "Get-Process | ForEach-Object { \"$($_.Id)`t$($_.ProcessName)`t$($_.CPU)\" }"
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                         capture_output=True, text=True).stdout
    procs = {}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0].isdigit():
            procs[int(parts[0])] = (parts[1], float(parts[2]) if parts[2] else 0.0)
    return procs


def typeperf(counters, samples=1, interval=1):
    out = subprocess.run(["typeperf", *counters, "-sc", str(samples), "-si", str(interval)],
                         capture_output=True, text=True).stdout
    rows = [l for l in out.splitlines() if l.startswith('"')]
    if len(rows) < 2:
        return [], []
    head = [h.strip('"') for h in rows[0].split(",")]
    vals = [[v.strip('"') for v in r.split(",")] for r in rows[1:]]
    return head, vals


def adapter_memory_gib():
    head, vals = typeperf([r"\GPU Adapter Memory(*)\Shared Usage", r"\GPU Adapter Memory(*)\Dedicated Usage"])
    if not vals:
        return None
    tot = {"shared": 0.0, "dedicated": 0.0}
    for h, v in zip(head[1:], vals[-1][1:]):
        key = "shared" if h.lower().endswith("shared usage") else "dedicated"
        tot[key] += float(v or 0)
    return {k: round(v / 2**30, 3) for k, v in tot.items()}


def gpu_engine_by_pid():
    """Mean utilisation per process over two 5 s samples, summed over engines."""
    head, vals = typeperf([r"\GPU Engine(*)\Utilization Percentage"], samples=2, interval=5)
    use = {}
    for i, h in enumerate(head[1:], 1):
        m = re.search(r"pid_(\d+)_", h)
        if not m:
            continue
        xs = [float(r[i]) for r in vals if i < len(r) and r[i] not in ("", " ")]
        if xs:
            use[int(m.group(1))] = use.get(int(m.group(1)), 0.0) + sum(xs) / len(xs)
    return use


# ---------------------------------------------------------------- driver

def load_doc():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return {"what": "Re-measurement of bench/RESULTS.md section 43 (GPT-2 small shape, 768d x12, vocab 50257)",
            "script": "bench/section43_remeasure.py", "published_section43": SECTION43,
            "published_section46_clean_process": SECTION46,
            "arms": {k: v[0] for k, v in ARMS.items()},
            "python": sys.version.split()[0], "logical_cpus": os.cpu_count(), "runs": []}


def save(doc):
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, OUT)


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def run(doc, phase, label, arm, seq, batch, seconds, worktree):
    spec = ARMS[arm][1]
    root = REPO if spec["tree"] == "wt" else worktree
    if spec.get("verbatim"):
        argv = ["bench/hf_gpt2_speed.py", "train", "--seq", str(seq), "--batch", str(batch),
                "--seconds", str(seconds)]
    else:
        argv = ["bench/section43_remeasure.py", "one", "--seq", str(seq), "--batch", str(batch),
                "--seconds", str(seconds)] + (["--tie"] if spec["tie"] else []) \
               + (["--gpt2"] if spec["gpt2"] else []) \
               + (["--root", "<40ad1b8 worktree>"] if spec["tree"] != "wt" else [])
    shown = "python " + " ".join(argv)
    real = [sys.executable, os.path.join(REPO, argv[0])] + \
        [worktree if x == "<40ad1b8 worktree>" else x for x in argv[1:]]

    used = sum(r.get("wall_s", 0) for r in doc["runs"])
    setups = [r["result"]["setup_s"] for r in doc["runs"] if r.get("result", {}).get("setup_s")]
    est = seconds + (max(setups) + 15 if setups else 90)
    rec = dict(phase=phase, label=label, arm=arm, seq=seq, batch=batch, seconds=seconds,
               command=shown, cwd="repo root" if spec["tree"] == "wt" else "40ad1b8 worktree")
    if used + est > BUDGET_S:
        rec.update(skipped=f"GPU time budget: {used:.0f} s used, this run needs ~{est:.0f} s of {BUDGET_S} s")
        doc["runs"].append(rec)
        save(doc)
        print(f"[{label}] SKIPPED: {rec['skipped']}")
        return rec

    # Idle check before the run: wait up to 2 minutes for the CPU to settle.
    waited, pre = 0, []
    while True:
        pre = []
        for _ in range(3):
            t = sys_times()
            time.sleep(1)
            pre.append(round(busy_pct(t, sys_times()), 1))
        if statistics.mean(pre) < 15 or waited >= 120:
            break
        time.sleep(10)
        waited += 13
    rec["cpu_pre_run_pct"] = pre
    rec["cpu_pre_run_waited_s"] = waited
    rec["contended_before_start"] = statistics.mean(pre) >= 15
    rec["gpu_adapter_memory_before_gib"] = adapter_memory_gib()
    before = processes()

    samples, stop = [], threading.Event()

    def sampler():
        t = sys_times()
        while not stop.wait(2.0):
            t2 = sys_times()
            samples.append(round(busy_pct(t, t2), 1))
            t = t2

    engine = {}
    rec["start"] = now()
    s0 = sys_times()
    t0 = time.perf_counter()
    child = subprocess.Popen(real, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    def engine_probe():
        if not stop.wait(min(40, seconds / 2 + 20)):
            engine.update(gpu_engine_by_pid())

    tg = threading.Thread(target=engine_probe, daemon=True)
    tg.start()
    out = child.communicate()[0]
    wall = time.perf_counter() - t0
    s1 = sys_times()
    child_cpu = proc_cpu_s(child._handle)
    stop.set()
    th.join()
    tg.join(timeout=15)
    rec["end"] = now()
    after = processes()

    ncpu = os.cpu_count()
    total_busy = busy_pct(s0, s1) / 100 * wall * ncpu
    deltas = []
    for pid, (name, cpu) in after.items():
        if pid in (child.pid, os.getpid()):
            continue
        d = cpu - before.get(pid, (name, 0.0))[1]
        if d >= 1.0 or (name.lower().startswith(("python", "ollama")) and d > 0.05):
            deltas.append(dict(name=name, pid=pid, cpu_s=round(d, 2)))
    deltas.sort(key=lambda x: -x["cpu_s"])
    others = [dict(name=after.get(p, before.get(p, ("?", 0)))[0], pid=p, pct=round(u, 1))
              for p, u in sorted(engine.items(), key=lambda x: -x[1])
              if p != child.pid and u >= 1.0]
    # Names and pids go to the console only: the JSON is committed, and a process list is
    # what .gitignore keeps out of committed artefacts.
    print(f"  other processes, CPU s: {deltas[:12]}\n  other processes, GPU %: {others}")
    rec.update(
        wall_s=round(wall, 2), returncode=child.returncode,
        cpu_during_pct_samples=samples, cpu_during_mean_pct=round(busy_pct(s0, s1), 2),
        cpu_total_busy_s=round(total_busy, 1), cpu_benchmark_process_s=round(child_cpu, 1) if child_cpu else None,
        cpu_other_processes_s=round(total_busy - (child_cpu or 0), 1),
        cpu_other_processes_mean_cores=round((total_busy - (child_cpu or 0)) / wall, 3),
        gpu_engine_util_pct_benchmark=round(engine.get(child.pid, 0.0), 1) if engine else None,
        gpu_engine_util_pct_other_sum=round(sum(o["pct"] for o in others), 1) if engine else None)
    res = [l for l in out.splitlines() if l.startswith("RESULT ")]
    if res:
        rec["result"] = json.loads(res[-1][7:])
    elif spec.get("verbatim"):
        m = re.search(r"(\d+) steps, ([\d,]+) tokens in ([\d.]+) s = ([\d,]+) tokens/s", out)
        m2 = re.search(r"step median ([\d.]+) ms, min ([\d.]+), max ([\d.]+)", out)
        m3 = re.search(r"vkgrad buffers ([\d.]+) GiB", out)
        if m and m2:
            rec["result"] = dict(steps=int(m.group(1)), tokens=int(m.group(2).replace(",", "")),
                                 elapsed_s=float(m.group(3)), tok_s_wall=float(m.group(4).replace(",", "")),
                                 step_ms_median=float(m2.group(1)), step_ms_min=float(m2.group(2)),
                                 step_ms_max=float(m2.group(3)),
                                 alloc_gib=float(m3.group(1)) if m3 else None)
        rec["stdout"] = out.splitlines()[-8:]
    if "result" not in rec:
        rec["stdout"] = out.splitlines()[-30:]
    doc["runs"].append(rec)
    save(doc)
    r = rec.get("result", {})
    print(f"[{label}] arm {arm} {seq}x{batch}: rc {child.returncode}, wall {wall:.0f} s, "
          f"{r.get('tok_s_wall')} tok/s, step med {r.get('step_ms_median')} ms, "
          f"submit med {r.get('submit_ms_median')} ms, cpu {rec['cpu_during_mean_pct']}% "
          f"(other {rec['cpu_other_processes_mean_cores']} cores), budget used {used + wall:.0f} s", flush=True)
    if child.returncode:
        time.sleep(10)  # let the driver recover if the device was lost
    return rec


def plan(a):
    doc = load_doc()
    if a.phase == "preflight":
        for arm in "AEBF":
            run(doc, "preflight", f"preflight-{arm}", arm, 512, 2, 10, a.worktree)
        return
    run(doc, "steady", "steady-512x2", "A", 512, 2, a.steady, a.worktree)
    for r in range(a.rounds):
        order = "ABCDEF" if r % 2 == 0 else "FEDCBA"
        for arm in order:
            run(doc, "interleave", f"interleave-r{r + 1}-{arm}", arm, 512, 2, a.short, a.worktree)
    for seq, batch in ((256, 2), (256, 4), (256, 8), (1024, 2)):
        run(doc, "steady", f"steady-{seq}x{batch}", "A", seq, batch, a.steady, a.worktree)
    summarise(doc)
    save(doc)


def summarise(doc):
    ok = [r for r in doc["runs"] if r.get("result") and r.get("returncode") == 0]
    s = {"steady": {}, "interleave_512x2": {}}
    for r in ok:
        if r["phase"] == "steady":
            key, res = f"{r['seq']}x{r['batch']}", r["result"]
            s["steady"][key] = {k: res.get(k) for k in (
                "tok_s_wall", "tok_s_second_half", "tok_s_from_median_step", "tok_s_from_best_submit",
                "step_ms_median", "submit_ms_median", "alloc_gib", "steps")}
            s["steady"][key].update(published_section43_tok_s=SECTION43[key][1],
                                    published_section43_step_ms=SECTION43[key][0],
                                    published_section43_gib=SECTION43[key][2],
                                    wall_vs_published=round(res["tok_s_wall"] / SECTION43[key][1], 3),
                                    cpu_other_mean_cores=r["cpu_other_processes_mean_cores"])
    arms = {}
    for r in ok:
        if r["phase"] == "interleave":
            arms.setdefault(r["arm"], []).append(r["result"])
    for arm, rs in sorted(arms.items()):
        med = lambda k: round(statistics.median(x[k] for x in rs), 1) if all(x.get(k) for x in rs) else None
        s["interleave_512x2"][arm] = dict(
            description=ARMS[arm][0], runs=len(rs), tok_s_wall_per_run=[x["tok_s_wall"] for x in rs],
            tok_s_wall_median=med("tok_s_wall"), step_ms_median=med("step_ms_median"),
            submit_ms_median=med("submit_ms_median"))
    if "A" in s["interleave_512x2"]:
        base = s["interleave_512x2"]["A"]["tok_s_wall_median"]
        for v in s["interleave_512x2"].values():
            v["vs_A"] = round(v["tok_s_wall_median"] / base, 3)
    doc["summary"] = s
    doc["gpu_process_wall_s_total"] = round(sum(r.get("wall_s", 0) for r in doc["runs"]), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("one", "plan", "summarise"))
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=45)
    ap.add_argument("--tie", action="store_true")
    ap.add_argument("--gpt2", action="store_true")
    ap.add_argument("--root", default=REPO)
    ap.add_argument("--worktree")
    ap.add_argument("--phase", choices=("preflight", "main"), default="main")
    ap.add_argument("--steady", type=float, default=240)
    ap.add_argument("--short", type=float, default=45)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    if a.mode == "one":
        one(a)
    elif a.mode == "summarise":
        doc = load_doc()
        summarise(doc)
        save(doc)
    else:
        if not a.worktree or not os.path.exists(os.path.join(a.worktree, "transformer.py")):
            sys.exit("--worktree must point at a checkout of 40ad1b8")
        plan(a)


if __name__ == "__main__":
    main()
