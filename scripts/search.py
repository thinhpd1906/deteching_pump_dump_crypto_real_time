"""A4: time-boxed, fair hyper-parameter search. VALIDATION ONLY.

Every trial is `finetune --search --set ...` in a subprocess, so the test set
is never opened (finetune --search guarantees it; see tests/test_protocol.py).

Fairness (CLAUDE.md s19.2 rule 4): every arm samples the same `common` space
with the same trial budget; arms run ROUND-ROBIN, one trial at a time, so a
time-box stop leaves the arms with equal trial counts (+-1). at_pre also
samples how the pretrained weights are used (its extra knobs are inherent to
having pretrained weights; the budget stays equal).

Resumable / kill -9 safe:
  * Optuna studies live in SQLite (artifacts/search/optuna_<arm>.db)
  * a trial left RUNNING by a killed driver is re-queued with the same
    params and marked FAIL; the re-queued trial maps to the same finetune
    tag, so finetune resumes from its {tag}_last.pt (or skips if done)
  * a finetune child that survived the driver is terminated first (pidfile)
  * pretrained variants are cached by hash (pretrain_cache/), built once

Outputs: artifacts/search/trials.jsonl (one line per finished trial or
confirm run), artifacts/search/leaderboard.md.

Usage:
    python scripts/search.py                    # search, then confirm top-k
    python scripts/search.py --phase confirm    # only the confirm phase
    python scripts/search.py --report           # rewrite leaderboard only
    python scripts/search.py --max-trials 1     # smoke: 1 new trial per arm
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.common import search_result_path  # noqa: E402
from src.training.finetune import EXIT_INCOMPLETE, run_tag  # noqa: E402
from src.training.protocol import (override_suffix,  # noqa: E402
                                   pretrained_ckpt_path)

ARMS = {"at_pre": ("at", True), "at_scratch": ("at", False),
        "cnnbilstm": ("cnnbilstm", False)}


# ------------------------------------------------------------------ space
def sample(trial, arm: str, space: dict) -> dict:
    """Optuna params -> {param_name: value} (names are optuna param names)."""
    def one(name, spec):
        t = spec["type"]
        if t == "loguniform":
            return trial.suggest_float(name, spec["low"], spec["high"], log=True)
        if t == "uniform":
            return trial.suggest_float(name, spec["low"], spec["high"])
        if t == "int":
            return trial.suggest_int(name, spec["low"], spec["high"])
        if t == "categorical":
            return trial.suggest_categorical(name, spec["choices"])
        raise ValueError(f"unknown space type {t} for {name}")

    out = {}
    specs = list(space.get("common", {}).items()) + list(space.get(arm, {}).items())
    for name, spec in specs:
        cond = spec.get("when")
        if cond and any(out.get(k) != v for k, v in cond.items()):
            continue
        out[name] = one(name, spec)
    return out


def params_to_overrides(arm: str, params: dict, variants: dict) -> dict:
    """Optuna params -> finetune --set overrides."""
    ov = {}
    for k, v in params.items():
        if k == "dropout":
            ov["model.cnn_dropout" if arm == "cnnbilstm" else "model.dropout"] = v
        elif k == "pretrain":
            if v != "canonical":
                ov.update(variants[v])
        else:
            ov[k] = v
    return ov


def set_args(ov: dict) -> list[str]:
    out = []
    for k, v in sorted(ov.items()):
        out += ["--set", f"{k}={v!r}" if isinstance(v, float) else f"{k}={v}"]
    return out


# ------------------------------------------------------------- processes
class Runner:
    def __init__(self, cfg_path: str, clean: str, out_dir: str, log_dir: str):
        self.cfg_path, self.clean = cfg_path, clean
        self.out_dir, self.log_dir = out_dir, log_dir
        self.pidfile = os.path.join(out_dir, "running.pid")

    def kill_orphan(self) -> None:
        """A child finetune that outlived a killed driver: stop it first."""
        if not os.path.exists(self.pidfile):
            return
        try:
            pid = int(open(self.pidfile).read().strip())
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError, PermissionError):
            os.remove(self.pidfile)
            return
        print(f"  stopping orphaned child pid {pid}")
        os.kill(pid, signal.SIGTERM)
        for _ in range(60):
            try:
                os.kill(pid, 0)
                time.sleep(1)
            except ProcessLookupError:
                break
        else:
            os.kill(pid, signal.SIGKILL)
        os.remove(self.pidfile)

    def run(self, argv: list[str], log_name: str, deadline: float | None) -> int:
        cmd = [sys.executable, *argv, "--config", self.cfg_path,
               "--clean", self.clean]
        if deadline and "src.training.finetune" in argv:
            cmd += ["--deadline", str(deadline)]
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, log_name + ".log"), "a") as fh:
            fh.write(f"\n$ {' '.join(cmd)}\n")
            fh.flush()
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, PYTHONUNBUFFERED="1"))
            with open(self.pidfile, "w") as pf:
                pf.write(str(p.pid))
            rc = p.wait()
        if os.path.exists(self.pidfile):
            os.remove(self.pidfile)
        return rc


# ----------------------------------------------------------------- search
class Search:
    def __init__(self, args, cfg: dict):
        import optuna
        self.optuna = optuna
        self.args, self.cfg = args, cfg
        S = cfg["automation"]["search"]
        self.S = S
        self.arms = [a for a in S["arms"] if a in ARMS]
        self.art = cfg["paths"]["artifacts_dir"]
        self.art_mode = os.path.join(self.art, args.clean)
        self.out = os.path.join(self.art, "search")
        os.makedirs(self.out, exist_ok=True)
        self.trials_log = os.path.join(self.out, "trials.jsonl")
        self.stop_at = dt.datetime.fromisoformat(S["stop_at"]).timestamp()
        self.runner = Runner(args.config, args.clean, self.out,
                             os.path.join(self.out, "logs"))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.studies = {}
        for i, arm in enumerate(self.arms):
            self.studies[arm] = optuna.create_study(
                study_name=arm, direction="maximize", load_if_exists=True,
                storage=f"sqlite:///{os.path.join(self.out, f'optuna_{arm}.db')}",
                sampler=optuna.samplers.TPESampler(
                    seed=int(S.get("sampler_seed", 0)) + i))

    # --- bookkeeping
    def log(self, row: dict) -> None:
        row = {"time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               **row}
        with open(self.trials_log, "a") as fh:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def recover(self) -> None:
        """Re-queue trials a killed driver left RUNNING (same params)."""
        TS = self.optuna.trial.TrialState
        for arm, st in self.studies.items():
            for t in st.get_trials(deepcopy=False, states=(TS.RUNNING,)):
                print(f"  [{arm}] trial {t.number} was interrupted -> re-queued")
                st.enqueue_trial(t.params, skip_if_exists=False)
                st._storage.set_trial_state_values(t._trial_id, TS.FAIL)

    def completed(self, arm: str):
        TS = self.optuna.trial.TrialState
        return self.studies[arm].get_trials(deepcopy=False, states=(TS.COMPLETE,))

    def arm_done(self, arm: str) -> str | None:
        done = self.completed(arm)
        if len(done) >= int(self.S["budget_trials_per_arm"]):
            return "budget"
        n = int(self.S.get("stop_if_no_improve", 0))
        if n and len(done) > n:
            vals = [t.value for t in sorted(done, key=lambda t: t.number)]
            if int(np.argmax(vals)) < len(vals) - n:
                return f"no improvement in last {n}"
        return None

    # --- one finetune (search mode, val only)
    def finetune(self, arm: str, ov: dict, seed: int, log_name: str):
        arch, pre = ARMS[arm]
        lf = float(self.S.get("label_frac", 1.0))
        tag = run_tag(arch, pre, lf, seed) + override_suffix(ov)
        if pre:
            self.ensure_pretrained(ov)
        argv = ["-m", "src.training.finetune", "--arch", arch,
                "--seed", str(seed), "--search", *set_args(ov)]
        if pre:
            argv.append("--pretrained")
        if lf < 1.0:
            argv += ["--label-frac", str(lf)]
        rc = self.runner.run(argv, log_name, self.stop_at)
        path = search_result_path(self.art, self.args.clean, tag)
        res = json.load(open(path)) if rc == 0 and os.path.exists(path) else None
        return rc, tag, res

    def ensure_pretrained(self, ov: dict) -> None:
        path = pretrained_ckpt_path(self.art_mode, ov)
        if os.path.dirname(path) == self.art_mode:          # canonical
            if not os.path.exists(path):
                raise SystemExit(f"canonical pretrained encoder missing: {path}")
            return
        if os.path.exists(path[:-3] + ".done"):
            return
        pre_ov = {k: v for k, v in ov.items() if k.startswith("train.pretrain_")}
        print(f"  building pretrained variant {pre_ov} -> {path}")
        argv = ["-m", "src.training.pretrain", *set_args(pre_ov)]
        if os.path.exists(path):          # partial from a killed run
            argv.append("--resume")
        rc = self.runner.run(argv, "pretrain" + override_suffix(pre_ov), None)
        if rc != 0:
            raise SystemExit(f"pretrain failed (rc={rc}); see {self.out}/logs")
        open(path[:-3] + ".done", "w").write("ok\n")

    # --- phases
    def search_phase(self) -> None:
        self.recover()
        variants = self.S.get("pretrain_variants", {})
        new = {a: 0 for a in self.arms}
        while True:
            if time.time() >= self.stop_at:
                print("stop_at reached -- search phase ends")
                return
            active = [a for a in self.arms if not self.arm_done(a)
                      and (self.args.max_trials is None
                           or new[a] < self.args.max_trials)]
            if not active:
                return
            # fairness: next trial goes to the arm with the fewest completed
            arm = min(active, key=lambda a: (len(self.completed(a)),
                                             self.arms.index(a)))
            st = self.studies[arm]
            trial = st.ask()
            params = sample(trial, arm, self.S["space"])
            ov = params_to_overrides(arm, params, variants)
            print(f"[{arm}] trial {trial.number} ({len(self.completed(arm))} done) "
                  f"{ov}", flush=True)
            t0 = time.time()
            rc, tag, res = self.finetune(arm, ov, 0, f"{arm}_t{trial.number}")
            mins = (time.time() - t0) / 60
            if res is None:
                if rc == EXIT_INCOMPLETE:
                    print(f"  paused at stop_at ({mins:.0f} min)")
                    return              # stays RUNNING -> re-queued on resume
                st.tell(trial, state=self.optuna.trial.TrialState.FAIL)
                self.log({"phase": "search", "arm": arm, "trial": trial.number,
                          "params": params, "overrides": ov, "tag": tag,
                          "state": "FAIL", "rc": rc, "minutes": round(mins, 1)})
                print(f"  FAILED rc={rc} -- see {self.out}/logs/{arm}_t{trial.number}.log")
                continue
            st.tell(trial, res["f1"])
            new[arm] += 1
            self.log({"phase": "search", "arm": arm, "trial": trial.number,
                      "params": params, "overrides": ov, "tag": tag,
                      "state": "COMPLETE", "val_f1": res["f1"],
                      "val_pr_auc": res.get("pr_auc"),
                      "val_event_recall": res.get("event_recall"),
                      "epochs_run": res.get("epochs_run"),
                      "minutes": round(mins, 1)})
            print(f"  val F1={res['f1']:.4f}  ({mins:.0f} min)  "
                  f"best[{arm}]={st.best_value:.4f}", flush=True)
            self.write_leaderboard()

    def top_configs(self, arm: str):
        k = int(self.S.get("confirm_top_k", 3))
        done = sorted(self.completed(arm), key=lambda t: -t.value)
        seen, out = set(), []
        for t in done:
            ov = params_to_overrides(arm, t.params, self.S.get("pretrain_variants", {}))
            key = override_suffix(ov)
            if key in seen:
                continue
            seen.add(key)
            out.append((t, ov))
            if len(out) == k:
                break
        return out

    def confirm_phase(self) -> None:
        for arm in self.arms:
            for t, ov in self.top_configs(arm):
                for seed in (0, 1, 2):
                    if time.time() >= self.stop_at:
                        print("stop_at reached -- confirm phase ends")
                        return
                    rc, tag, res = self.finetune(arm, ov, seed,
                                                 f"{arm}_confirm_t{t.number}_s{seed}")
                    if res is None:
                        print(f"  [{arm}] confirm t{t.number} s{seed} rc={rc}")
                        if rc == EXIT_INCOMPLETE:
                            return
                        continue
                    if not self.logged_confirm(tag):
                        self.log({"phase": "confirm", "arm": arm,
                                  "trial": t.number, "seed": seed,
                                  "overrides": ov, "tag": tag,
                                  "val_f1": res["f1"]})
                    print(f"  [{arm}] confirm t{t.number} s{seed} "
                          f"val F1={res['f1']:.4f}", flush=True)
            self.write_leaderboard()

    def logged_confirm(self, tag: str) -> bool:
        if not os.path.exists(self.trials_log):
            return False
        return any(json.loads(l).get("tag") == tag and
                   json.loads(l).get("phase") == "confirm"
                   for l in open(self.trials_log) if l.strip())

    # --- leaderboard
    def confirm_stats(self, arm: str, ov: dict):
        arch, pre = ARMS[arm]
        lf = float(self.S.get("label_frac", 1.0))
        vals = []
        for seed in (0, 1, 2):
            tag = run_tag(arch, pre, lf, seed) + override_suffix(ov)
            p = search_result_path(self.art, self.args.clean, tag)
            if os.path.exists(p):
                vals.append(json.load(open(p))["f1"])
        return vals

    def write_leaderboard(self) -> None:
        tgt = self.cfg["automation"].get("target", {})
        base = tgt.get("must_beat", "b5_logreg")
        bp = search_result_path(self.art, self.args.clean, base)
        base_f1 = json.load(open(bp))["f1"] if os.path.exists(bp) else None
        L = ["# A4 search leaderboard (VALIDATION F1 only, test never used)", "",
             f"Updated {dt.datetime.now().isoformat(timespec='minutes')}. "
             f"Budget {self.S['budget_trials_per_arm']} trials/arm, "
             f"stop_at {self.S['stop_at']}.", ""]
        L.append(f"Target: beat `{base}` val F1 = "
                 + (f"{base_f1:.4f}" if base_f1 is not None else
                    "n/a (run `python -m src.models.baselines --search`)")
                 + f" by margin = {tgt.get('margin', 'seed_std')}.")
        L += ["", "| arm | trials done | failed | best seed-0 val F1 | "
                  "proposed config (top by 3-seed mean) | 3-seed val F1 | beats target |",
              "|---|---|---|---|---|---|---|"]
        TS = self.optuna.trial.TrialState
        details = []
        for arm in self.arms:
            done = self.completed(arm)
            nfail = len(self.studies[arm].get_trials(deepcopy=False,
                                                     states=(TS.FAIL,)))
            best = max((t.value for t in done), default=None)
            prop, prop_vals = None, []
            for t, ov in self.top_configs(arm):
                v = self.confirm_stats(arm, ov)
                if len(v) == 3 and (not prop_vals or np.mean(v) > np.mean(prop_vals)):
                    prop, prop_vals = (t, ov), v
            if prop:
                mu, sd = float(np.mean(prop_vals)), float(np.std(prop_vals))
                beats = ("n/a" if base_f1 is None else
                         ("yes" if mu - base_f1 > sd else "no"))
                cell = f"t{prop[0].number}: `{prop[1]}`"
                stat = f"{mu:.4f} ± {sd:.4f}"
            else:
                cell, stat, beats = "(confirm phase pending)", "", ""
            L.append(f"| {arm} | {len(done)} | {nfail} | "
                     f"{'' if best is None else f'{best:.4f}'} | {cell} | {stat} | {beats} |")
            rows = sorted(done, key=lambda t: -t.value)[:10]
            details += ["", f"## {arm}: top 10 trials", "",
                        "| trial | val F1 | params |", "|---|---|---|"]
            details += [f"| {t.number} | {t.value:.4f} | `{t.params}` |" for t in rows]
        L += details
        tmp = os.path.join(self.out, "leaderboard.md.tmp")
        open(tmp, "w").write("\n".join(L) + "\n")
        os.replace(tmp, os.path.join(self.out, "leaderboard.md"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--clean", default="adaptive")
    ap.add_argument("--phase", choices=["all", "search", "confirm"], default="all")
    ap.add_argument("--report", action="store_true", help="rewrite leaderboard only")
    ap.add_argument("--max-trials", type=int, default=None,
                    help="at most N NEW trials per arm in this invocation")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    s = Search(args, cfg)
    if args.report:
        s.write_leaderboard()
        return
    s.runner.kill_orphan()
    base = cfg["automation"].get("target", {}).get("must_beat", "b5_logreg")
    if not os.path.exists(search_result_path(s.art, args.clean, base)):
        print(f"target {base} has no val result yet -> baselines --search")
        rc = s.runner.run(["-m", "src.models.baselines", "--search"],
                          "baselines_search", None)
        if rc != 0:
            print(f"  baselines --search failed rc={rc} (leaderboard shows n/a)")
    if args.phase in ("all", "search"):
        s.search_phase()
    if args.phase in ("all", "confirm") and args.max_trials is None:
        s.confirm_phase()
    s.write_leaderboard()
    print(f"leaderboard -> {os.path.join(s.out, 'leaderboard.md')}")


if __name__ == "__main__":
    main()
