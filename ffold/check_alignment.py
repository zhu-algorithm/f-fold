"""Verify FFOLD_BASE against the OpenFold3-preview reference config.

The reference JSON is extracted from the openfold3==0.4.1 sdist
(`openfold3/projects/of3_all_atom/config/model_config.py`), which is
Apache-2.0. FFOLD_BASE in ffold/config.py is written independently; this
script is what makes the claim "F-Fold matches OpenFold3's hyperparameters"
falsifiable.

Exit code 0 iff every shared field agrees.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BASE_DIMS, FFOLD_BASE, FFOLD_MINI, flatten  # noqa: E402

# Fields present in the OpenFold3 config that describe kernels, memory
# offloading, checkpointing or weight-init policy rather than architecture
# shape. They are out of scope for a hyperparameter-dimension match.
SKIP_SUBSTRINGS = (
    "linear_init_param",  # also covers "linear_init_params"
    "use_deepspeed", "use_cueq", "use_triton", "use_lma",
    "use_reentrant", "clear_cache", "tune_chunk_size", "chunk_size",
    "blocks_per_ckpt", "ckpt_per_template", "offload",
    "fuse_projection", "inf", "eps",
)


def in_scope(key: str) -> bool:
    return not any(s in key for s in SKIP_SUBSTRINGS)


def compare(ref_path: Path) -> tuple[list[dict], dict]:
    ref = json.loads(ref_path.read_text())
    ref_arch = {k: v for k, v in ref["architecture"].items() if in_scope(k)}
    ref_dims = ref["base_dims"]

    base = flatten(FFOLD_BASE)
    mini = flatten(FFOLD_MINI)

    rows: list[dict] = []

    for k in sorted(ref_dims):
        rows.append({
            "scope": "base_dims", "field": k,
            "openfold3": ref_dims[k],
            "ffold_base": BASE_DIMS.get(k, "<absent>"),
            "ffold_mini": "",
            "status": "match" if BASE_DIMS.get(k) == ref_dims[k] else "MISMATCH",
        })

    for k in sorted(ref_arch):
        of3 = ref_arch[k]
        if k not in base:
            rows.append({"scope": "architecture", "field": k, "openfold3": of3,
                         "ffold_base": "<not implemented>", "ffold_mini": "",
                         "status": "not_implemented"})
            continue
        rows.append({
            "scope": "architecture", "field": k, "openfold3": of3,
            "ffold_base": base[k], "ffold_mini": mini.get(k, ""),
            "status": "match" if base[k] == of3 else "MISMATCH",
        })

    extra = sorted(set(base) - set(ref_arch) - {f"shared.{d}" for d in ()})
    for k in extra:
        rows.append({"scope": "architecture", "field": k, "openfold3": "<absent>",
                     "ffold_base": base[k], "ffold_mini": mini.get(k, ""),
                     "status": "ffold_only"})

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return rows, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=str(Path(__file__).with_name(
        "of3_reference_config.json")))
    ap.add_argument("--out", default=str(Path(__file__).with_name(
        "alignment_report.csv")))
    args = ap.parse_args()

    rows, counts = compare(Path(args.reference))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scope", "field", "openfold3",
                                          "ffold_base", "ffold_mini", "status"])
        w.writeheader()
        w.writerows(rows)

    for status in sorted(counts):
        print(f"{status:>16}: {counts[status]}")

    bad = [r for r in rows if r["status"] == "MISMATCH"]
    for r in bad[:40]:
        print(f"  MISMATCH {r['field']}: of3={r['openfold3']!r} "
              f"ffold={r['ffold_base']!r}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
