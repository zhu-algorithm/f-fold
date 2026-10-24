"""Instantiate F-Fold on the meta device and report parameter counts.

Meta-device instantiation allocates no storage, so ffold-base (AF3 scale)
can be audited on a laptop.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import get_config  # noqa: E402
from model import FFold  # noqa: E402

MODULE_GROUPS = {
    "input_embedder": ("linear_restype", "atom_attn_enc", "linear_atom_token",
                       "linear_s_init", "linear_z_i", "linear_z_j", "rel_pos"),
    "recycling": ("ln_recycle_s", "linear_recycle_s", "ln_recycle_z",
                  "linear_recycle_z"),
    "msa_module": ("msa_module",),
    "template_module": ("template_module",),
    "pairformer": ("pairformer",),
    "diffusion_conditioning": ("diffusion_conditioning",),
    "diffusion_atom_encoder": ("diffusion_atom_enc",),
    "diffusion_transformer": ("diffusion_transformer",),
    "diffusion_atom_decoder": ("diffusion_atom_dec",),
    "distogram_head": ("linear_distogram",),
    "confidence_heads": ("confidence",),
}


def audit(preset: str, use_msa: bool, use_template: bool) -> dict[str, int]:
    cfg = get_config(preset)
    with torch.device("meta"):
        model = FFold(cfg, use_msa=use_msa, use_template=use_template)
    per_top: dict[str, int] = {}
    for name, p in model.named_parameters():
        per_top[name.split(".")[0]] = per_top.get(name.split(".")[0], 0) + p.numel()
    grouped = {g: sum(per_top.get(m, 0) for m in members)
               for g, members in MODULE_GROUPS.items()}
    grouped["TOTAL"] = sum(per_top.values())
    assert grouped["TOTAL"] == sum(v for k, v in grouped.items() if k != "TOTAL"), \
        f"ungrouped params: {set(per_top) - {m for ms in MODULE_GROUPS.values() for m in ms}}"
    return grouped


def main() -> None:
    rows = []
    base = audit("ffold-base", use_msa=True, use_template=True)
    mini = audit("ffold-mini", use_msa=False, use_template=False)
    for g in list(MODULE_GROUPS) + ["TOTAL"]:
        rows.append({"module": g, "ffold_base": base[g], "ffold_mini": mini[g]})

    out = Path(__file__).with_name("param_audit.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["module", "ffold_base", "ffold_mini"])
        w.writeheader()
        w.writerows(rows)

    width = max(len(r["module"]) for r in rows)
    print(f"{'module':<{width}}  {'ffold-base':>14}  {'ffold-mini':>12}")
    for r in rows:
        print(f"{r['module']:<{width}}  {r['ffold_base']:>14,}  "
              f"{r['ffold_mini']:>12,}")


if __name__ == "__main__":
    main()
