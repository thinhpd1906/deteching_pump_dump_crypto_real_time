"""Data manifest: prove two machines hold byte-identical data.

data/ and *.pt checkpoints are gitignored, so a fresh clone (e.g. the Mac)
must fetch them separately (Kaggle datasets, re-download, or scp). This script
records size + sha256 of every such file into config/data_manifest.json, which
IS committed. The other machine runs --verify to confirm parity before any
experiment is trusted.

Usage:
    python scripts/make_data_manifest.py              # write manifest
    python scripts/make_data_manifest.py --verify     # compare, exit 1 on mismatch
    python scripts/make_data_manifest.py --verify --only processed checkpoints
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

MANIFEST = "config/data_manifest.json"
GROUPS = {
    "labels": ["data/labels/*.csv"],
    "raw": ["data/raw/**/*.zip", "data/raw/**/*.csv", "data/raw/**/*.parquet"],
    "processed": ["data/processed/*/*.npz", "data/processed/*/*.json"],
    "checkpoints": ["artifacts/**/*.pt"],
}


def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    if path.endswith(".csv"):
        # git checks text files out as CRLF on Windows and LF on the Mac;
        # hash CSVs with normalised line endings so the same content matches.
        with open(path, "rb") as fh:
            h.update(fh.read().replace(b"\r\n", b"\n"))
        return h.hexdigest()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def collect(groups: list[str]) -> dict:
    out = {}
    for g in groups:
        files = sorted({p.replace("\\", "/")
                        for pat in GROUPS[g] for p in glob.glob(pat, recursive=True)})
        out[g] = {p: {"size": os.path.getsize(p), "sha256": sha256(p)} for p in files}
        print(f"  {g:12s} {len(files):6d} files")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--only", nargs="*", default=list(GROUPS), choices=list(GROUPS))
    args = ap.parse_args()

    if not args.verify:
        man = collect(list(GROUPS))
        tmp = MANIFEST + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(man, fh, indent=1, sort_keys=True)
        os.replace(tmp, MANIFEST)
        print(f"wrote {MANIFEST}")
        return

    ref = json.load(open(MANIFEST))
    have = collect(args.only)
    bad = 0
    for g in args.only:
        want, got = ref.get(g, {}), have[g]
        missing = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want))
        changed = sorted(p for p in set(want) & set(got)
                         if want[p]["sha256"] != got[p]["sha256"])
        status = "OK" if not (missing or changed) else "MISMATCH"
        print(f"[{status}] {g}: missing={len(missing)} changed={len(changed)} "
              f"extra={len(extra)}")
        for p in (missing + changed)[:20]:
            print("    ", p)
        bad += len(missing) + len(changed)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
