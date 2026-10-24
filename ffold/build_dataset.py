"""Build the AlphaFold-teacher distillation set.

Teacher labels come from AlphaFold DB (EBI) predictions: Cb-Cb distance
matrices discretised into the AF3 64-bin distogram (2-22 A) and per-residue
pLDDT read from the B-factor column.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

AF_URL = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v{v}.pdb"
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
RESTYPES = "ARNDCQEGHILKMFPSTWYVX"
RESTYPE_IDX = {a: i for i, a in enumerate(RESTYPES)}


def kmers(seq: str, k: int = 5) -> set[str]:
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def dedupe(df: pd.DataFrame, n_target: int, max_shared: float = 0.30,
           seed: int = 0) -> pd.DataFrame:
    """Greedy k-mer redundancy filter with an inverted index over accepted."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    index: dict[str, list[int]] = {}
    kept_kmers: list[set[str]] = []
    kept_rows: list[int] = []
    for pos in order:
        seq = df.iloc[pos]["Sequence"]
        ks = kmers(seq)
        if not ks:
            continue
        hits: dict[int, int] = {}
        for km in ks:
            for j in index.get(km, ()):
                hits[j] = hits.get(j, 0) + 1
        redundant = any(
            n / min(len(ks), len(kept_kmers[j])) > max_shared
            for j, n in hits.items())
        if redundant:
            continue
        j = len(kept_kmers)
        kept_kmers.append(ks)
        kept_rows.append(pos)
        for km in ks:
            index.setdefault(km, []).append(j)
        if len(kept_rows) >= n_target:
            break
    return df.iloc[kept_rows].reset_index(drop=True)


# AlphaFold DB model versions, newest first. v6 is the 2025-08-01 release;
# older ones are kept as a fallback for entries not yet rebuilt.
AF_VERSIONS = (6, 5, 4)


def fetch_af(acc: str, timeout: int = 30, retries: int = 3):
    """Fetch an AlphaFold DB PDB; returns (text, version) or (None, None).

    Any transport error is retried with backoff. A 404 moves on to the next
    model version; 404 on all of them means the entry has no prediction.
    """
    for v in AF_VERSIONS:
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(AF_URL.format(acc=acc, v=v),
                                            timeout=timeout) as r:
                    return r.read().decode(), v
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    break           # try the next model version
                time.sleep(1.5 * (attempt + 1))
            except Exception:       # URLError, RemoteDisconnected, timeouts
                time.sleep(1.5 * (attempt + 1))
    return None, None


def parse_pdb(text: str):
    """Return (seq, ca[N,3], cb[N,3], plddt[N]) from an AlphaFold PDB."""
    res: dict[int, dict] = {}
    for line in text.splitlines():
        if not line.startswith("ATOM"):
            continue
        name = line[12:16].strip()
        if name not in ("CA", "CB"):
            continue
        resn = line[17:20].strip()
        resi = int(line[22:26])
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        b = float(line[60:66])
        r = res.setdefault(resi, {"aa": THREE_TO_ONE.get(resn, "X")})
        r[name] = xyz
        if name == "CA":
            r["plddt"] = b
    keys = sorted(res)
    if not keys:
        return None
    seq, ca, cb, pl = [], [], [], []
    for k in keys:
        r = res[k]
        if "CA" not in r:
            return None
        seq.append(r["aa"])
        ca.append(r["CA"])
        cb.append(r.get("CB", r["CA"]))   # glycine has no CB
        pl.append(r.get("plddt", 0.0))
    return ("".join(seq), np.array(ca, np.float32), np.array(cb, np.float32),
            np.array(pl, np.float32))


def one(acc: str):
    """Never raises: a failed accession is reported as None, not an exception
    (ThreadPoolExecutor.map would otherwise abort the whole build)."""
    try:
        txt, v = fetch_af(acc)
        return acc, (parse_pdb(txt) if txt is not None else None), v
    except Exception:
        return acc, None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="../data/uniprot_candidates.parquet")
    ap.add_argument("--out", default="../data/teacher")
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--holdout", default="P61626")  # human lysozyme C
    ap.add_argument("--min-plddt", type=float, default=70.0,
                    help="drop teachers whose mean pLDDT is below this")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.candidates)
    df = df[df["Length"] <= args.max_len]
    holdout_seq = None
    hrow = df[df["Entry"] == args.holdout]
    if len(hrow):
        holdout_seq = hrow.iloc[0]["Sequence"]
        df = df[df["Entry"] != args.holdout]

    # drop anything resembling the holdout before sampling
    if holdout_seq:
        hk = kmers(holdout_seq)
        keep = [len(kmers(s) & hk) / max(1, min(len(hk), len(kmers(s)))) <= 0.30
                for s in df["Sequence"]]
        df = df[np.array(keep)]

    sel = dedupe(df, args.n)
    print(f"candidates={len(df)} selected={len(sel)}")

    accs = list(sel["Entry"])
    if holdout_seq is not None:
        accs.append(args.holdout)

    records, failed, versions, low_conf = {}, [], {}, []
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for i, (acc, parsed, ver) in enumerate(ex.map(one, accs)):
            if parsed is None:
                failed.append(acc)
            else:
                seq, ca, cb, pl = parsed
                # A low mean pLDDT means AlphaFold predicted disorder, not a
                # fold; such a "structure" is not a useful teacher label.
                if float(pl.mean()) < args.min_plddt and acc != args.holdout:
                    low_conf.append(acc)
                else:
                    records[acc] = {"seq": seq, "ca": ca, "cb": cb, "plddt": pl}
                    versions[ver] = versions.get(ver, 0) + 1
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(accs)} kept={len(records)} "
                      f"lowconf={len(low_conf)} fail={len(failed)}", flush=True)

    print(f"kept={len(records)} low_confidence={len(low_conf)} "
          f"failed={len(failed)}")

    np.savez_compressed(
        out / "teacher.npz",
        **{f"{a}|seq": np.frombuffer(
            bytes([RESTYPE_IDX.get(c, 20) for c in r["seq"]]), dtype=np.uint8)
           for a, r in records.items()},
        **{f"{a}|cb": r["cb"] for a, r in records.items()},
        **{f"{a}|plddt": r["plddt"] for a, r in records.items()},
    )
    meta = {
        "source": "AlphaFold DB (alphafold.ebi.ac.uk)",
        "model_versions_used": {str(k): v for k, v in sorted(versions.items())},
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_structures": len(records),
        "holdout": args.holdout,
        "max_len": args.max_len,
        "dedupe_max_shared_kmer_frac": 0.30,
        "min_mean_plddt": args.min_plddt,
        "n_dropped_low_confidence": len(low_conf),
        "n_failed_download": len(failed),
        "accessions": sorted(records),
        "failed": failed,
    }
    (out / "teacher_meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote", out / "teacher.npz")


if __name__ == "__main__":
    main()
