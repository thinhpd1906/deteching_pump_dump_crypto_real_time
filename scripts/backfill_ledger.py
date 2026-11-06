"""Backfill artifacts/test_ledger.jsonl from result JSONs that predate it.

Every artifacts/<mode>/<tag>_test.json without a ledger entry gets one with
"backfilled": true. The config those runs used was not recorded, so the
entry carries the hash of the CURRENT config with "config_hash_assumed": true;
that is the conservative choice: rerunning the same tag with the default
config is refused by the ledger instead of silently re-touching test.

Idempotent: a (mode, tag) that already has any ledger entry is skipped.

Usage:
    python scripts/backfill_ledger.py [--config config/config.yaml] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.protocol import config_hash, ledger_path, read_ledger  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    led = ledger_path(cfg)
    have = {(e.get("clean"), e.get("tag")) for e in read_ledger(led)}
    chash = config_hash(cfg)

    rows = []
    art = cfg["paths"]["artifacts_dir"]
    for f in sorted(glob.glob(os.path.join(art, "*", "*_test.json"))):
        clean = os.path.basename(os.path.dirname(f))
        tag = json.load(open(f)).get("tag") or os.path.basename(f)[:-len("_test.json")]
        if (clean, tag) in have:
            continue
        mtime = dt.datetime.fromtimestamp(os.path.getmtime(f), dt.timezone.utc)
        rows.append({
            "time": mtime.isoformat(timespec="seconds"),
            "clean": clean, "tag": tag, "script": "backfill",
            "git_sha": None, "config_hash": chash,
            "config_hash_assumed": True, "overrides": {},
            "checkpoint": None, "checkpoint_sha256": None,
            "override_reason": None, "backfilled": True,
            "result_file": os.path.relpath(f, art),
        })

    for r in rows:
        print(f"  backfill {r['clean']}/{r['tag']}")
    if args.dry_run or not rows:
        print(f"{len(rows)} entries {'would be ' if args.dry_run else ''}added")
        return
    os.makedirs(os.path.dirname(led) or ".", exist_ok=True)
    with open(led, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    print(f"{len(rows)} entries added -> {led}")


if __name__ == "__main__":
    main()
