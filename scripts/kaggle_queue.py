"""Resumable job queue for the Week 6-7 GPU battery (Kaggle sessions).

Runs every fine-tune / label-efficiency / unsupervised job that does not yet
have a result JSON, in priority order (Gate 3 evidence first), until a time
budget runs out. Each job is a separate process; finetune saves its state
every epoch and exits cleanly before the deadline, so NOTHING finished is
ever redone: the next session restores artifacts and picks up where this
one stopped.

With 2 GPUs (Kaggle "GPU T4 x2") two jobs run in parallel, one per GPU.
Per-job logs go to artifacts/<mode>/logs/<tag>.log.

Usage:
    python scripts/kaggle_queue.py --budget-hours 11
    python scripts/kaggle_queue.py --restore /kaggle/input --budget-hours 11
    python scripts/kaggle_queue.py --list          # show done / pending only
"""
from __future__ import annotations

import argparse
import glob
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.finetune import EXIT_INCOMPLETE, run_tag  # noqa: E402

SEEDS = [0, 1, 2]
MODELS = [("at", True), ("at", False), ("cnnbilstm", False)]


def build_jobs(mode: str) -> list[tuple[str, list[str]]]:
    """(tag, argv) in priority order. Tag == result JSON stem."""
    def ft(arch, pre, frac, seed):
        argv = ["-m", "src.training.finetune", "--clean", mode,
                "--arch", arch, "--seed", str(seed)]
        if pre:
            argv.append("--pretrained")
        if frac < 1.0:
            argv += ["--label-frac", str(frac)]
        return run_tag(arch, pre, frac, seed), argv

    jobs = []
    # 1. Gate 3 core: at_pre vs at_scratch, full labels and 10% of events
    for frac in (1.0, 0.10):
        for seed in SEEDS:
            jobs.append(ft("at", True, frac, seed))
            jobs.append(ft("at", False, frac, seed))
    # 2. Table 1 remaining rows
    jobs += [ft("cnnbilstm", False, 1.0, s) for s in SEEDS]
    jobs.append(("b7_lstmae", ["-m", "src.training.eval_unsupervised",
                               "--clean", mode]))
    # 3. rest of the label-efficiency curve
    jobs += [ft("cnnbilstm", False, 0.10, s) for s in SEEDS]
    for frac in (0.25, 0.50):
        for seed in SEEDS:
            for arch, pre in MODELS:
                jobs.append(ft(arch, pre, frac, seed))
    return jobs


def restore(src_root: str, art: str, mode: str) -> None:
    """Copy finished results + partial-run state from previous outputs."""
    n = 0
    for d in glob.glob(os.path.join(src_root, "**", "artifacts", mode),
                       recursive=True):
        for f in glob.glob(os.path.join(d, "*")):
            if os.path.isdir(f):
                continue
            dst = os.path.join(art, os.path.basename(f))
            if not os.path.exists(dst):
                shutil.copy(f, dst)
                n += 1
    print(f"restored {n} files from {src_root}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--clean", default="adaptive")
    ap.add_argument("--budget-hours", type=float, default=11.0)
    ap.add_argument("--min-start-minutes", type=float, default=25.0,
                    help="do not START a new job with less time left")
    ap.add_argument("--restore", default=None,
                    help="dir to search for previous artifacts/<mode>/")
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    deadline = t_start + args.budget_hours * 3600
    cfg = yaml.safe_load(open(args.config))
    art = os.path.join(cfg["paths"]["artifacts_dir"], args.clean)
    logs = os.path.join(art, "logs")
    os.makedirs(logs, exist_ok=True)

    if args.restore:
        restore(args.restore, art, args.clean)

    jobs = build_jobs(args.clean)
    done = [t for t, _ in jobs if os.path.exists(os.path.join(art, f"{t}_test.json"))]
    todo = [(t, a) for t, a in jobs if t not in done]
    print(f"jobs: {len(jobs)} total, {len(done)} done, {len(todo)} pending")
    for t, _ in todo:
        part = " (partial, will resume)" if os.path.exists(
            os.path.join(art, f"{t}_last.pt")) else ""
        print(f"  pending: {t}{part}")
    if args.list or not todo:
        return

    if args.gpus is None:
        try:
            import torch
            args.gpus = max(torch.cuda.device_count(), 1)
        except ImportError:
            args.gpus = 1
    print(f"running on {args.gpus} worker(s), budget {args.budget_hours}h\n")

    q: queue.Queue = queue.Queue()
    for j in todo:
        q.put(j)
    stop = threading.Event()
    lock = threading.Lock()

    def worker(gpu: int) -> None:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   PYTHONUNBUFFERED="1")
        while not stop.is_set():
            try:
                tag, argv = q.get_nowait()
            except queue.Empty:
                return
            left_min = (deadline - time.time()) / 60
            if left_min < args.min_start_minutes:
                with lock:
                    print(f"[gpu{gpu}] {left_min:.0f} min left -- not starting {tag}")
                stop.set()
                return
            t0 = time.time()
            with lock:
                print(f"[gpu{gpu}] START {tag}  ({left_min:.0f} min left)", flush=True)
            with open(os.path.join(logs, f"{tag}.log"), "a") as fh:
                rc = subprocess.call([sys.executable, *argv,
                                      "--config", args.config,
                                      *(["--deadline", str(deadline)]
                                        if "finetune" in argv[1] else [])],
                                     stdout=fh, stderr=subprocess.STDOUT, env=env)
            mins = (time.time() - t0) / 60
            with lock:
                if rc == 0:
                    tail = open(os.path.join(logs, f"{tag}.log")).read()
                    line = next((l for l in tail.splitlines()[::-1]
                                 if "TEST  F1" in l or "test F1" in l), "")
                    print(f"[gpu{gpu}] DONE  {tag}  {mins:.0f} min  {line.strip()}",
                          flush=True)
                elif rc == EXIT_INCOMPLETE:
                    print(f"[gpu{gpu}] PAUSED {tag} at deadline (state saved)",
                          flush=True)
                    stop.set()
                else:
                    print(f"[gpu{gpu}] FAILED {tag} rc={rc} -- see {logs}/{tag}.log",
                          flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in range(args.gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    left = [t for t, _ in jobs
            if not os.path.exists(os.path.join(art, f"{t}_test.json"))]
    print(f"\nsession finished in {(time.time() - t_start) / 3600:.1f}h; "
          f"{len(jobs) - len(left)}/{len(jobs)} jobs done")
    if left:
        print("NEXT SESSION: attach this version's output and rerun with --restore")


if __name__ == "__main__":
    main()
