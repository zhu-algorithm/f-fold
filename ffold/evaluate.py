"""Evaluate ffold-mini against its AlphaFold teacher on the held-out protein.

Reconstructs a Cb point cloud from the predicted distogram (classical MDS
initialisation, then weighted gradient descent on the distance constraints),
superposes it onto the teacher with Kabsch, and reports contact precision,
distogram cross-entropy, pLDDT correlation, RMSD and TM-score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FFold, RESTYPES  # noqa: E402

ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL", "X": "UNK",
}


def classical_mds(d: np.ndarray) -> np.ndarray:
    n = len(d)
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (d ** 2) @ j
    w, v = np.linalg.eigh(b)
    idx = np.argsort(w)[::-1][:3]
    return v[:, idx] * np.sqrt(np.clip(w[idx], 0, None))


def refine(x0, d_target, weight, n_steps=1500, lr=0.05, bond=3.8,
           bond_w=5.0, seed=0):
    torch.manual_seed(seed)
    x = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    dt = torch.tensor(d_target, dtype=torch.float32)
    w = torch.tensor(weight, dtype=torch.float32)
    opt = torch.optim.Adam([x], lr=lr)
    n = x.shape[0]
    offdiag = ~torch.eye(n, dtype=torch.bool)
    for _ in range(n_steps):
        opt.zero_grad()
        d = torch.cdist(x, x)
        loss = (w * (d - dt) ** 2)[offdiag].sum() / w[offdiag].sum()
        seq_d = (x[1:] - x[:-1]).norm(dim=-1)
        loss = loss + bond_w * ((seq_d - bond) ** 2).mean()
        loss.backward()
        opt.step()
    return x.detach().numpy()


def kabsch(p: np.ndarray, q: np.ndarray):
    """Superpose p onto q; returns rotated p and RMSD."""
    pc, qc = p - p.mean(0), q - q.mean(0)
    u, s, vt = np.linalg.svd(pc.T @ qc)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    pr = pc @ r.T
    return pr + q.mean(0), float(np.sqrt(((pr - qc) ** 2).sum(1).mean()))


def tm_score(p_sup: np.ndarray, q: np.ndarray) -> float:
    n = len(q)
    d0 = 1.24 * (n - 15) ** (1 / 3) - 1.8 if n > 15 else 0.5
    d0 = max(d0, 0.5)
    d = np.linalg.norm(p_sup - q, axis=1)
    return float(np.mean(1.0 / (1.0 + (d / d0) ** 2)))


def contact_precision(pred_prob_contact, true_d, seq_sep, topk_div):
    n = len(true_d)
    iu = np.triu_indices(n, k=seq_sep)
    p, t = pred_prob_contact[iu], (true_d[iu] < 8.0)
    k = max(1, n // topk_div)
    top = np.argsort(p)[::-1][:k]
    return float(t[top].mean()), int(k)


def write_pdb(path, seq, xyz, bfac):
    lines = [
        "REMARK   F-Fold reconstruction. Coordinates are CB positions derived",
        "REMARK   from the predicted CB-CB distogram by MDS + distance-geometry",
        "REMARK   refinement; glycine uses CA. No backbone was modelled.",
        "REMARK   B-factor column holds predicted pLDDT (0-100).",
    ]
    for i, (aa, (x, y, z), b) in enumerate(zip(seq, xyz, bfac), start=1):
        name = "CA" if aa == "G" else "CB"
        lines.append(
            f"ATOM  {i:5d}  {name:<3s}{ONE_TO_THREE.get(aa,'UNK'):>4s} A{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}           C")
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../runs/ffold_mini/ffold_mini_best.pt")
    ap.add_argument("--data", default="../data/teacher/teacher.npz")
    ap.add_argument("--meta", default="../data/teacher/teacher_meta.json")
    ap.add_argument("--acc", default=None)
    ap.add_argument("--out", default="../results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(Path(args.meta).read_text())
    acc = args.acc or meta["holdout"]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = FFold(ck["config"])
    model.load_state_dict(ck["model"])
    model.eval()

    z = np.load(args.data)
    seq_idx = z[f"{acc}|seq"].astype(np.int64)
    cb_true = z[f"{acc}|cb"].astype(np.float32)
    plddt_true = z[f"{acc}|plddt"].astype(np.float32)
    seq = "".join(RESTYPES[i] for i in seq_idx)
    L = len(seq_idx)

    with torch.no_grad():
        o = model(torch.from_numpy(seq_idx)[None],
                  torch.arange(L)[None], mask=torch.ones(1, L))
    logits = o["distogram_logits"][0]
    prob = logits.softmax(-1).numpy()
    centres = model.distogram_bin_centres().numpy()
    d_pred = (prob * centres).sum(-1)
    plddt_pred = model.plddt(o)[0].numpy()

    # teacher reference
    d_true = np.linalg.norm(cb_true[:, None] - cb_true[None], axis=-1)

    # distogram cross-entropy against teacher labels
    edges = np.linspace(2.0, 22.0, 63)
    lab = np.clip((d_true[..., None] > edges).sum(-1), 0, 63)
    ce = float(F.cross_entropy(logits.permute(2, 0, 1)[None],
                               torch.from_numpy(lab)[None]).item())

    # contacts (< 8 A) from the distogram, long and medium range
    contact_prob = prob[..., :int(np.searchsorted(centres, 8.0))].sum(-1)
    metrics = {"accession": acc, "length": int(L),
               "distogram_cross_entropy": ce}
    for sep, tag in [(12, "medium+long"), (24, "long")]:
        for div, dtag in [(5, "L_5"), (2, "L_2"), (1, "L")]:
            p, k = contact_precision(contact_prob, d_true, sep, div)
            metrics[f"contact_P@{dtag}_sep{sep}"] = round(p, 4)
            metrics[f"contact_topk@{dtag}_sep{sep}"] = k

    # pLDDT agreement
    metrics["plddt_pearson"] = round(float(np.corrcoef(plddt_pred, plddt_true)[0, 1]), 4)
    rp = np.argsort(np.argsort(plddt_pred)); rt = np.argsort(np.argsort(plddt_true))
    metrics["plddt_spearman"] = round(float(np.corrcoef(rp, rt)[0, 1]), 4)
    metrics["plddt_pred_mean"] = round(float(plddt_pred.mean()), 2)
    metrics["plddt_teacher_mean"] = round(float(plddt_true.mean()), 2)

    # 3D reconstruction from the predicted distances
    var = (prob * (centres[None, None] - d_pred[..., None]) ** 2).sum(-1)
    weight = 1.0 / (1.0 + var)
    x0 = classical_mds(d_pred)
    x = refine(x0, d_pred, weight)
    x_sup, rmsd = kabsch(x, cb_true)
    # mirror image is an equally valid MDS solution; keep the better one
    xm_sup, rmsd_m = kabsch(x * np.array([1, 1, -1]), cb_true)
    if rmsd_m < rmsd:
        x_sup, rmsd = xm_sup, rmsd_m
    metrics["rmsd_cb_A"] = round(rmsd, 3)
    metrics["tm_score"] = round(tm_score(x_sup, cb_true), 4)

    # control: reconstruct from the TEACHER distances to bound the
    # reconstruction procedure's own error
    xt = refine(classical_mds(d_true), d_true, np.ones_like(d_true))
    xt_sup, rmsd_t = kabsch(xt, cb_true)
    xtm_sup, rmsd_tm = kabsch(xt * np.array([1, 1, -1]), cb_true)
    metrics["rmsd_cb_A_teacher_distances_control"] = round(min(rmsd_t, rmsd_tm), 3)

    write_pdb(out / f"ffold_{acc}.pdb", seq, x_sup, plddt_pred)
    np.savez_compressed(out / f"ffold_{acc}_pred.npz", d_pred=d_pred,
                        d_true=d_true, prob=prob, plddt_pred=plddt_pred,
                        plddt_true=plddt_true, contact_prob=contact_prob,
                        coords=x_sup, cb_true=cb_true)
    (out / f"metrics_{acc}.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
