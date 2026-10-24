"""Distil AlphaFold teacher structures into ffold-mini.

Losses follow the AF3 label space: 64-bin Cb distogram over 2-22 A and a
50-bin pLDDT head over [0, 1]. The optimiser keeps AF3's schedule shape
(linear warmup, then stepwise exponential decay); the peak learning rate is
reduced from AF3's 1.8e-3 because ffold-mini is ~40x narrower.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import get_config  # noqa: E402
from model import FFold  # noqa: E402


class TeacherSet(Dataset):
    def __init__(self, npz_path: str, accs: list[str], crop: int):
        self.z = np.load(npz_path)
        self.accs = accs
        self.crop = crop

    def __len__(self):
        return len(self.accs)

    def __getitem__(self, i):
        a = self.accs[i]
        seq = self.z[f"{a}|seq"].astype(np.int64)
        cb = self.z[f"{a}|cb"].astype(np.float32)
        pl = self.z[f"{a}|plddt"].astype(np.float32) / 100.0
        L = len(seq)
        if L > self.crop:
            s = np.random.randint(0, L - self.crop + 1)
            seq, cb, pl = seq[s:s + self.crop], cb[s:s + self.crop], pl[s:s + self.crop]
        return (torch.from_numpy(seq), torch.from_numpy(cb),
                torch.from_numpy(pl))


def collate(batch):
    L = max(len(b[0]) for b in batch)
    B = len(batch)
    seq = torch.full((B, L), 20, dtype=torch.long)
    cb = torch.zeros(B, L, 3)
    pl = torch.zeros(B, L)
    mask = torch.zeros(B, L)
    for i, (s, c, p) in enumerate(batch):
        n = len(s)
        seq[i, :n], cb[i, :n], pl[i, :n], mask[i, :n] = s, c, p, 1.0
    return seq, cb, pl, mask


def distogram_labels(cb, n_bins, bmin, bmax):
    d = torch.cdist(cb, cb)
    edges = torch.linspace(bmin, bmax, n_bins - 1, device=cb.device)
    return (d[..., None] > edges).sum(-1).clamp(max=n_bins - 1), d


def lr_at(step, peak, warmup, decay_every, decay_factor):
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * decay_factor ** ((step - warmup) // decay_every)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/teacher/teacher.npz")
    ap.add_argument("--meta", default="../data/teacher/teacher_meta.json")
    ap.add_argument("--preset", default="ffold-mini")
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--decay-every", type=int, default=1500)
    ap.add_argument("--decay-factor", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=10.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", default="../runs/ffold_mini")
    ap.add_argument("--probe-memory", action="store_true")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-batches", type=int, default=20,
                    help="cap validation batches per eval (0 = all)")
    ap.add_argument("--auto-batch", action="store_true",
                    help="pick the largest batch that fits before training")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    meta = json.loads(Path(args.meta).read_text())
    accs = [a for a in meta["accessions"] if a != meta["holdout"]]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(accs)
    n_val = max(1, int(len(accs) * args.val_frac))
    val_accs, train_accs = accs[:n_val], accs[n_val:]

    cfg = get_config(args.preset)
    model = FFold(cfg).to(dev)
    if args.grad_checkpoint:
        model.pairformer.grad_checkpoint = True
        model.confidence.pairformer.grad_checkpoint = True
    n_par = sum(p.numel() for p in model.parameters())

    nb = cfg["loss_module"]["distogram"]["no_bins"]
    bmin = cfg["loss_module"]["distogram"]["bin_min"]
    bmax = cfg["loss_module"]["distogram"]["bin_max"]
    n_pl = cfg["loss_module"]["confidence"]["plddt"]["no_bins"]

    tr = DataLoader(TeacherSet(args.data, train_accs, args.crop),
                    batch_size=args.batch, shuffle=True, collate_fn=collate,
                    drop_last=True)
    va = DataLoader(TeacherSet(args.data, val_accs, args.crop),
                    batch_size=args.batch, shuffle=False, collate_fn=collate)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                           eps=1e-8)
    scaler = torch.amp.GradScaler(dev, enabled=args.amp and dev == "cuda")

    def step_loss(seq, cb, pl, mask):
        ri = torch.arange(seq.shape[1], device=dev)[None].expand(seq.shape[0], -1)
        out = model(seq, ri, mask=mask)
        lab, _ = distogram_labels(cb, nb, bmin, bmax)
        pm = mask[..., :, None] * mask[..., None, :]
        ld = (F.cross_entropy(out["distogram_logits"].permute(0, 3, 1, 2), lab,
                              reduction="none") * pm).sum() / pm.sum().clamp(min=1)
        pl_lab = (pl * n_pl).long().clamp(max=n_pl - 1)
        pl_logit = out["plddt_logits"].mean(dim=-2)
        lp = (F.cross_entropy(pl_logit.permute(0, 2, 1), pl_lab,
                              reduction="none") * mask).sum() / mask.sum().clamp(min=1)
        return ld + 0.3 * lp, ld.detach(), lp.detach()

    if args.probe_memory:
        model.train()
        seq, cb, pl, mask = next(iter(tr))
        seq, cb, pl, mask = seq.to(dev), cb.to(dev), pl.to(dev), mask.to(dev)
        with torch.autocast(dev, torch.float16, enabled=args.amp):
            loss, _, _ = step_loss(seq, cb, pl, mask)
        scaler.scale(loss).backward()
        if dev == "cuda":
            print(f"params={n_par:,} batch={args.batch} crop={args.crop} "
                  f"peak_vram_GB={torch.cuda.max_memory_allocated()/2**30:.2f}")
        else:
            print(f"params={n_par:,} (cpu probe, no vram reading)")
        return

    if args.auto_batch and dev == "cuda":
        chosen = 1
        for b in (8, 6, 4, 3, 2, 1):
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                seq = torch.randint(0, 20, (b, args.crop), device=dev)
                cb = torch.randn(b, args.crop, 3, device=dev) * 10
                pl = torch.rand(b, args.crop, device=dev)
                mk = torch.ones(b, args.crop, device=dev)
                model.train()
                with torch.autocast(dev, torch.float16, enabled=args.amp):
                    loss, _, _ = step_loss(seq, cb, pl, mk)
                scaler.scale(loss).backward()
                peak = torch.cuda.max_memory_allocated() / 2 ** 30
                model.zero_grad(set_to_none=True)
                del seq, cb, pl, mk, loss
                print(f"  probe batch={b} crop={args.crop} peak={peak:.2f} GB",
                      flush=True)
                chosen = b
                break
            except torch.cuda.OutOfMemoryError:
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"  probe batch={b} OOM", flush=True)
        if chosen != args.batch:
            print(f"auto_batch: {args.batch} -> {chosen}", flush=True)
            args.batch = chosen
            tr = DataLoader(TeacherSet(args.data, train_accs, args.crop),
                            batch_size=args.batch, shuffle=True,
                            collate_fn=collate, drop_last=True)
            va = DataLoader(TeacherSet(args.data, val_accs, args.crop),
                            batch_size=args.batch, shuffle=False,
                            collate_fn=collate)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    hist = []
    t0 = time.time()
    best = float("inf")
    it = iter(tr)
    model.train()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.lr, args.warmup, args.decay_every,
                            args.decay_factor)
        try:
            batch = next(it)
        except StopIteration:
            it = iter(tr)
            batch = next(it)
        seq, cb, pl, mask = [x.to(dev) for x in batch]
        opt.zero_grad(set_to_none=True)
        with torch.autocast(dev, torch.float16, enabled=args.amp):
            loss, ld, lp = step_loss(seq, cb, pl, mask)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(opt)
        scaler.update()

        if (step + 1) % args.eval_every == 0:
            model.eval()
            vd = vp = vn = 0.0
            with torch.no_grad():
                for vi, b in enumerate(va):
                    if args.val_batches and vi >= args.val_batches:
                        break
                    s2, c2, p2, m2 = [x.to(dev) for x in b]
                    with torch.autocast(dev, torch.float16, enabled=args.amp):
                        _, d_, p_ = step_loss(s2, c2, p2, m2)
                    vd += float(d_); vp += float(p_); vn += 1
            model.train()
            rec = {"step": step + 1, "lr": opt.param_groups[0]["lr"],
                   "train_disto": float(ld), "train_plddt": float(lp),
                   "val_disto": vd / vn, "val_plddt": vp / vn,
                   "elapsed_s": time.time() - t0}
            hist.append(rec)
            print(json.dumps(rec), flush=True)
            if rec["val_disto"] < best:
                best = rec["val_disto"]
                torch.save({"model": model.state_dict(), "config": cfg,
                            "preset": args.preset, "step": step + 1,
                            "val_disto": best, "args": vars(args)},
                           outdir / "ffold_mini_best.pt")

    (outdir / "history.json").write_text(json.dumps(hist, indent=2))
    (outdir / "run_meta.json").write_text(json.dumps(
        {"params": n_par, "n_train": len(train_accs), "n_val": len(val_accs),
         "device": dev, "best_val_disto": best, "args": vars(args)}, indent=2))
    print(f"done best_val_disto={best:.4f} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
