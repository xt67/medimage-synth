#!/usr/bin/env python3
"""Validate the TB Chest X-ray dataset after download.

Usage:
    python validate_dataset.py [data_dir]

Default data_dir: data/raw/TB_Chest_Radiography_Database
"""
import os
import sys
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

from PIL import Image

VALID_EXTS = {".png", ".jpg", ".jpeg"}


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(data_dir: str) -> dict:
    root = Path(data_dir)
    report = {
        "data_dir": str(root),
        "exists": root.exists(),
        "classes": {},
        "total_images": 0,
        "corrupted": [],
        "unsupported_formats": [],
        "dimensions": Counter(),
        "duplicate_hashes": {},
        "errors": [],
    }

    if not root.exists():
        report["errors"].append(f"Directory not found: {root}")
        return report

    # Discover class folders
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not class_dirs:
        report["errors"].append("No class subfolders found.")
        return report

    all_hashes = defaultdict(list)

    for cdir in class_dirs:
        cname = cdir.name
        files = [f for f in cdir.iterdir() if f.is_file()]
        imgs = [f for f in files if f.suffix.lower() in VALID_EXTS]
        other = [f for f in files if f.suffix.lower() not in VALID_EXTS]

        corrupted = []
        dims = Counter()
        hashes = Counter()
        dup_groups = defaultdict(list)

        for f in imgs:
            try:
                with Image.open(f) as im:
                    im.verify()
                with Image.open(f) as im:
                    dims[im.size] += 1
            except Exception as e:
                corrupted.append((str(f), str(e)))
                continue
            h = md5_file(f)
            hashes[h] += 1
            all_hashes[h].append(str(f))
            dup_groups[h].append(str(f))

        duplicates = {h: paths for h, paths in dup_groups.items() if len(paths) > 1}

        report["classes"][cname] = {
            "num_files_total": len(files),
            "num_images": len(imgs),
            "num_unsupported": len(other),
            "corrupted": corrupted,
            "dimensions": {f"{w}x{h}": n for (w, h), n in dims.items()},
            "num_duplicate_groups": len(duplicates),
            "num_duplicate_extra": sum(len(v) - 1 for v in duplicates.values()),
        }
        report["total_images"] += len(imgs)
        report["corrupted"].extend(corrupted)
        report["unsupported_formats"].extend(str(f) for f in other)
        for (w, h), n in dims.items():
            report["dimensions"][f"{w}x{h}"] += n

    # Cross-class duplicate hashes
    cross_class_dups = {}
    for h, paths in all_hashes.items():
        classes = {Path(p).parent.name for p in paths}
        if len(classes) > 1:
            cross_class_dups[h] = paths
    report["cross_class_duplicates"] = cross_class_dups

    # Summary duplicate groups (all)
    report["duplicate_groups_total"] = sum(
        1 for paths in all_hashes.values() if len(paths) > 1
    )

    return report


def print_report(r: dict) -> None:
    print("=" * 70)
    print("DATASET VALIDATION REPORT")
    print("=" * 70)
    print(f"Data dir : {r['data_dir']}")
    print(f"Exists   : {r['exists']}")
    if r["errors"]:
        for e in r["errors"]:
            print(f"ERROR: {e}")
        print("=" * 70)
        return
    print(f"Total images: {r['total_images']}")
    print()
    for cname, info in r["classes"].items():
        print(f"Class: {cname}")
        print(f"  images            : {info['num_images']}")
        print(f"  unsupported files : {info['num_unsupported']}")
        print(f"  corrupted         : {len(info['corrupted'])}")
        print(f"  dimensions        : {info['dimensions']}")
        print(f"  duplicate groups  : {info['num_duplicate_groups']} "
              f"(extra copies: {info['num_duplicate_extra']})")
        print()
    print(f"Aggregate dimensions: {dict(r['dimensions'])}")
    print(f"Corrupted total      : {len(r['corrupted'])}")
    print(f"Unsupported total   : {len(r['unsupported_formats'])}")
    print(f"Cross-class dups     : {len(r['cross_class_duplicates'])}")
    print(f"Duplicate groups (all): {r['duplicate_groups_total']}")
    print("=" * 70)


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/TB_Chest_Radiography_Database"
    r = validate(data_dir)
    print_report(r)

    out = Path("outputs") / "dataset_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Convert Counter to dict for JSON
    r_json = dict(r)
    r_json["dimensions"] = dict(r["dimensions"])
    r_json["classes"] = {k: {kk: (dict(vv) if isinstance(vv, Counter) else vv)
                            for kk, vv in v.items()} for k, v in r["classes"].items()}
    # Convert Counter in class dimensions already dict; ensure
    for k, v in r_json["classes"].items():
        if isinstance(v.get("dimensions"), Counter):
            v["dimensions"] = dict(v["dimensions"])
    with open(out, "w") as f:
        json.dump(r_json, f, indent=2, default=str)
    print(f"Saved validation report to: {out}")


if __name__ == "__main__":
    main()
