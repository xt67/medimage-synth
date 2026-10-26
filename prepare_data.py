#!/usr/bin/env python3
"""Prepare (organize + deduplicate) the TB dataset from raw to processed.

Reads:  data/raw/TB_Chest_Radiography_Database/{Normal,Tuberculosis}
Writes: data/processed/{Normal,Tuberculosis} (deduplicated by MD5)

Reports counts and duplicate removal. Run once after downloading.
"""
import hashlib
import shutil
from pathlib import Path

RAW = Path("data/raw/TB_Chest_Radiography_Database")
PROC = Path("data/processed")
CLASSES = ["Normal", "Tuberculosis"]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not RAW.exists():
        raise SystemExit(f"Raw dataset not found at {RAW}. Download it first.")

    PROC.mkdir(parents=True, exist_ok=True)
    for c in CLASSES:
        (PROC / c).mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0
    total_dup = 0
    for c in CLASSES:
        src_dir = RAW / c
        dst_dir = PROC / c
        # clean destination for reproducibility
        for f in dst_dir.iterdir():
            if f.is_file():
                f.unlink()
        seen = set()
        kept = 0
        dups = 0
        files = sorted(src_dir.glob("*.png")) + sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.jpeg"))
        for f in files:
            total_in += 1
            h = md5(f)
            if h in seen:
                dups += 1
                total_dup += 1
                continue
            seen.add(h)
            shutil.copy2(f, dst_dir / f.name)
            kept += 1
            total_out += 1
        print(f"{c}: input={len(files)} kept={kept} duplicates_removed={dups}")

    print(f"\nTotal: input={total_in} kept={total_out} duplicates_removed={total_dup}")
    print(f"Processed data written to: {PROC}")


if __name__ == "__main__":
    main()
