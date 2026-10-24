"""F-Fold configuration.

`FFOLD_BASE` is written independently against the AlphaFold3 specification as
realised by OpenFold3-preview (openfold3==0.4.1, Apache-2.0). It is NOT derived
programmatically from the OpenFold3 config -- that would make the alignment
check vacuous. `ffold/check_alignment.py` diffs it field-by-field against
`of3_reference_config.json`, which IS extracted from the OpenFold3 source.

`FFOLD_MINI` is a width/depth-reduced variant that fits a 15 GB Tesla T4.
Fields that define the *label space* (distogram bins, pLDDT bins, PAE/PDE bins
and their edges) are deliberately held identical to FFOLD_BASE so that a mini
checkpoint's output heads remain interpretable on the same scale.
"""

from __future__ import annotations

import copy
from typing import Any

# --------------------------------------------------------------------------
# Base hidden dimensions (AF3 SI Table 6 / OpenFold3 model_config.py)
# --------------------------------------------------------------------------
C_S = 384  # single (token) representation
C_Z = 128  # pair representation
C_M = 64  # MSA representation
C_T = 64  # template representation
C_ATOM = 128
C_ATOM_PAIR = 16
C_TOKEN_EMBEDDER = 384
C_TOKEN_DIFFUSION = 768
C_S_INPUT = C_TOKEN_EMBEDDER + 65  # 449

SIGMA_DATA = 16
MAX_RELATIVE_IDX = 32
MAX_RELATIVE_CHAIN = 2
N_QUERY = 32  # local atom attention query window
N_KEY = 128  # local atom attention key window
MAX_ATOMS_PER_TOKEN = 23

BASE_DIMS = {
    "c_s": C_S,
    "c_z": C_Z,
    "c_m": C_M,
    "c_t": C_T,
    "c_atom": C_ATOM,
    "c_atom_pair": C_ATOM_PAIR,
    "c_token_embedder": C_TOKEN_EMBEDDER,
    "c_token_diffusion": C_TOKEN_DIFFUSION,
    "c_s_input": C_S_INPUT,
    "sigma_data": SIGMA_DATA,
    "max_relative_idx": MAX_RELATIVE_IDX,
    "max_relative_chain": MAX_RELATIVE_CHAIN,
    "n_query": N_QUERY,
    "n_key": N_KEY,
    "max_atoms_per_token": MAX_ATOMS_PER_TOKEN,
}


def _atom_attn_enc(c_token: int, c_atom: int, c_atom_pair: int,
                   c_s: int, c_z: int, c_hidden: int, no_heads: int,
                   no_blocks: int, n_query: int, n_key: int) -> dict[str, Any]:
    return {
        "c_atom": c_atom,
        "c_atom_pair": c_atom_pair,
        "c_atom_ref": {"element": 119, "name_chars": 256},
        "c_hidden": c_hidden,
        "c_s": c_s,
        "c_token": c_token,
        "c_z": c_z,
        "n_key": n_key,
        "n_query": n_query,
        "n_transition": 2,
        "no_blocks": no_blocks,
        "no_heads": no_heads,
        "use_ada_layer_norm": True,
    }


def _build(
    *,
    c_s: int,
    c_z: int,
    c_m: int,
    c_t: int,
    c_atom: int,
    c_atom_pair: int,
    c_token_embedder: int,
    c_token_diffusion: int,
    c_s_input: int,
    n_query: int,
    n_key: int,
    max_atoms_per_token: int,
    num_recycles: int,
    pairformer_blocks: int,
    pairformer_heads_pair_bias: int,
    pairformer_c_hidden_pair_bias: int,
    pairformer_c_hidden_mul: int,
    pairformer_c_hidden_pair_att: int,
    pairformer_no_heads_pair: int,
    pairformer_transition_n: int,
    msa_blocks: int,
    msa_c_hidden_msa_att: int,
    msa_c_hidden_opm: int,
    msa_c_hidden_mul: int,
    msa_c_hidden_pair_att: int,
    msa_no_heads_msa: int,
    msa_no_heads_pair: int,
    msa_transition_n: int,
    msa_subsample: int,
    template_blocks: int,
    template_c_hidden_tri_att: int,
    template_c_hidden_tri_mul: int,
    template_no_heads: int,
    atom_enc_c_hidden: int,
    atom_enc_no_heads: int,
    atom_enc_no_blocks: int,
    dit_blocks: int,
    dit_no_heads: int,
    dit_c_hidden: int,
    conf_pairformer_blocks: int,
    no_samples: int,
) -> dict[str, Any]:
    """Assemble a full F-Fold config. Key paths mirror OpenFold3 exactly."""
    return {
        "shared": {
            "c_s": c_s,
            "c_s_input": c_s_input,
            "c_z": c_z,
            "num_recycles": num_recycles,
            "sync_seed": 0,
            "use_confidence_emb_prob": 1.0,
            "diffusion": {
                "no_samples": no_samples,
                "no_mini_rollout_samples": 1,
                "no_full_rollout_samples": 5,
                "sigma_data": SIGMA_DATA,
                "use_conditioning_prob": 1.0,
            },
        },
        "input_embedder": {
            "c_s": c_s,
            "c_s_input": c_s_input,
            "c_z": c_z,
            "max_relative_idx": MAX_RELATIVE_IDX,
            "max_relative_chain": MAX_RELATIVE_CHAIN,
            "atom_attn_enc": _atom_attn_enc(
                c_token_embedder, c_atom, c_atom_pair, c_s, c_z,
                atom_enc_c_hidden, atom_enc_no_heads, atom_enc_no_blocks,
                n_query, n_key,
            ),
        },
        "msa": {
            "msa_module_embedder": {
                "c_m": c_m,
                "c_m_feats": 34,
                "c_s_input": c_s_input,
                "subsample_main_msa": False,
                "subsample_all_msa": True,
                "min_subsampled_all_msa": msa_subsample,
                "max_subsampled_all_msa": msa_subsample,
            },
            "msa_module": {
                "c_m": c_m,
                "c_z": c_z,
                "c_hidden_msa_att": msa_c_hidden_msa_att,
                "c_hidden_opm": msa_c_hidden_opm,
                "c_hidden_mul": msa_c_hidden_mul,
                "c_hidden_pair_att": msa_c_hidden_pair_att,
                "no_heads_msa": msa_no_heads_msa,
                "no_heads_pair": msa_no_heads_pair,
                "no_blocks": msa_blocks,
                "transition_n": msa_transition_n,
                "transition_type": "swiglu",
                "msa_dropout": 0.15,
                "pair_dropout": 0.25,
                "opm_first": True,
            },
        },
        "template": {
            "c_t": c_t,
            "c_z": c_z,
            "template_pair_embedder": {
                "c_in": c_z,
                "c_dgram": 39,
                "c_aatype": 32,
                "c_out": c_t,
            },
            "template_pair_stack": {
                "c_t": c_t,
                "c_hidden_tri_att": template_c_hidden_tri_att,
                "c_hidden_tri_mul": template_c_hidden_tri_mul,
                "no_blocks": template_blocks,
                "no_heads": template_no_heads,
                "pair_transition_n": 2,
                "transition_type": "swiglu",
                "dropout_rate": 0.25,
                "tri_mul_first": True,
            },
        },
        "pairformer": {
            "c_s": c_s,
            "c_z": c_z,
            "c_hidden_pair_bias": pairformer_c_hidden_pair_bias,
            "no_heads_pair_bias": pairformer_heads_pair_bias,
            "c_hidden_mul": pairformer_c_hidden_mul,
            "c_hidden_pair_att": pairformer_c_hidden_pair_att,
            "no_heads_pair": pairformer_no_heads_pair,
            "no_blocks": pairformer_blocks,
            "transition_n": pairformer_transition_n,
            "transition_type": "swiglu",
            "pair_dropout": 0.25,
        },
        "diffusion_module": {
            "diffusion_module": {
                "c_s": c_s,
                "c_token": c_token_diffusion,
                "sigma_data": SIGMA_DATA,
            },
            "diffusion_conditioning": {
                "c_s": c_s,
                "c_s_input": c_s_input,
                "c_z": c_z,
                "c_fourier_emb": 256,
                "max_relative_idx": MAX_RELATIVE_IDX,
                "max_relative_chain": MAX_RELATIVE_CHAIN,
                "seed_fourier_emb": 42,
                "sigma_data": SIGMA_DATA,
            },
            "atom_attn_enc": _atom_attn_enc(
                c_token_diffusion, c_atom, c_atom_pair, c_s, c_z,
                atom_enc_c_hidden, atom_enc_no_heads, atom_enc_no_blocks,
                n_query, n_key,
            ),
            "diffusion_transformer": {
                "c_a": c_token_diffusion,
                "c_s": c_s,
                "c_z": c_z,
                "c_hidden": dit_c_hidden,
                "no_heads": dit_no_heads,
                "no_blocks": dit_blocks,
                "n_transition": 2,
                "n_query": None,
                "n_key": None,
                "use_ada_layer_norm": True,
            },
            "atom_attn_dec": {
                "c_atom": c_atom,
                "c_atom_pair": c_atom_pair,
                "c_hidden": atom_enc_c_hidden,
                "c_token": c_token_diffusion,
                "n_key": n_key,
                "n_query": n_query,
                "n_transition": 2,
                "no_blocks": atom_enc_no_blocks,
                "no_heads": atom_enc_no_heads,
                "use_ada_layer_norm": True,
            },
        },
        # EDM noise schedule (Karras et al. 2022) as parameterised by AF3
        "noise_schedule": {
            "s_max": 160.0,
            "s_min": 4e-4,
            "p": 7,
            "sigma_data": SIGMA_DATA,
        },
        "sample_diffusion": {
            "gamma_0": 0.8,
            "gamma_min": 1.0,
            "noise_scale": 1.003,
            "step_scale": 1.5,
        },
        "heads": {
            "max_atoms_per_token": max_atoms_per_token,
            "per_sample_token_cutoff": 750,
            "pairformer_embedding": {
                "c_s_input": c_s_input,
                "c_z": c_z,
                "min_bin": 3.25,
                "max_bin": 50.75,
                "no_bin": 39,
                "pairformer": {
                    "c_s": c_s,
                    "c_z": c_z,
                    "c_hidden_pair_bias": pairformer_c_hidden_pair_bias,
                    "no_heads_pair_bias": pairformer_heads_pair_bias,
                    "c_hidden_mul": pairformer_c_hidden_mul,
                    "c_hidden_pair_att": pairformer_c_hidden_pair_att,
                    "no_heads_pair": pairformer_no_heads_pair,
                    "no_blocks": conf_pairformer_blocks,
                    "transition_n": pairformer_transition_n,
                    "transition_type": "swiglu",
                    "pair_dropout": 0.25,
                },
            },
            "distogram": {"c_z": c_z, "c_out": 64, "enabled": True},
            "pae": {"c_z": c_z, "c_out": 64, "enabled": True},
            "pde": {"c_z": c_z, "c_out": 64},
            "lddt": {"c_s": c_s, "c_out": 50,
                     "max_atoms_per_token": max_atoms_per_token},
            "experimentally_resolved": {
                "c_s": c_s, "c_out": 2,
                "max_atoms_per_token": max_atoms_per_token},
        },
        "loss_module": {
            "train_confidence_only": False,
            "low_mem_validation": False,
            "per_sample_atom_cutoff": 10000,
            "diffusion_loss_names": "['bond', 'smooth_lddt', 'mse']",
            "confidence_loss_names": (
                "['plddt', 'pde', 'experimentally_resolved', 'pae']"),
            "diffusion": {
                "sigma_data": SIGMA_DATA,
                "dna_weight": 5.0,
                "rna_weight": 5.0,
                "ligand_weight": 10.0,
                "use_sparse_loss": False,
            },
            "distogram": {"no_bins": 64, "bin_min": 2.0, "bin_max": 22.0},
            "confidence": {
                "plddt": {"no_bins": 50, "bin_min": 0.0, "bin_max": 1.0},
                "pae": {"no_bins": 64, "bin_min": 0.0, "bin_max": 32.0,
                        "angle_threshold": 25.0, "enabled": True},
                "pde": {"no_bins": 64, "bin_min": 0.0, "bin_max": 32.0},
                "experimentally_resolved": {"no_bins": 2},
                "per_sample_atom_cutoff": 10000,
            },
        },
    }


FFOLD_BASE: dict[str, Any] = _build(
    c_s=C_S, c_z=C_Z, c_m=C_M, c_t=C_T,
    c_atom=C_ATOM, c_atom_pair=C_ATOM_PAIR,
    c_token_embedder=C_TOKEN_EMBEDDER, c_token_diffusion=C_TOKEN_DIFFUSION,
    c_s_input=C_S_INPUT, n_query=N_QUERY, n_key=N_KEY,
    max_atoms_per_token=MAX_ATOMS_PER_TOKEN,
    num_recycles=3,
    pairformer_blocks=48,
    pairformer_heads_pair_bias=16,
    pairformer_c_hidden_pair_bias=24,   # c_s / no_heads_pair_bias
    pairformer_c_hidden_mul=128,
    pairformer_c_hidden_pair_att=32,
    pairformer_no_heads_pair=4,
    pairformer_transition_n=4,
    msa_blocks=4,
    msa_c_hidden_msa_att=8,
    msa_c_hidden_opm=32,
    msa_c_hidden_mul=128,
    msa_c_hidden_pair_att=32,
    msa_no_heads_msa=8,
    msa_no_heads_pair=4,
    msa_transition_n=4,
    msa_subsample=1024,
    template_blocks=2,
    template_c_hidden_tri_att=16,
    template_c_hidden_tri_mul=64,
    template_no_heads=4,
    atom_enc_c_hidden=32,
    atom_enc_no_heads=4,
    atom_enc_no_blocks=3,
    dit_blocks=24,
    dit_no_heads=16,
    dit_c_hidden=48,                    # c_token_diffusion / no_heads
    conf_pairformer_blocks=4,
    no_samples=48,
)

# T4-sized variant. Widths cut ~3x, depths cut ~6x; bin counts untouched.
_MINI_C_S = 128
FFOLD_MINI: dict[str, Any] = _build(
    c_s=_MINI_C_S, c_z=48, c_m=32, c_t=32,
    c_atom=32, c_atom_pair=8,
    c_token_embedder=_MINI_C_S, c_token_diffusion=256,
    c_s_input=_MINI_C_S + 65, n_query=N_QUERY, n_key=N_KEY,
    max_atoms_per_token=MAX_ATOMS_PER_TOKEN,
    num_recycles=1,
    pairformer_blocks=8,
    pairformer_heads_pair_bias=8,
    pairformer_c_hidden_pair_bias=16,   # c_s / no_heads_pair_bias
    pairformer_c_hidden_mul=48,
    pairformer_c_hidden_pair_att=16,
    pairformer_no_heads_pair=4,
    pairformer_transition_n=2,
    msa_blocks=2,
    msa_c_hidden_msa_att=8,
    msa_c_hidden_opm=16,
    msa_c_hidden_mul=48,
    msa_c_hidden_pair_att=16,
    msa_no_heads_msa=4,
    msa_no_heads_pair=4,
    msa_transition_n=2,
    msa_subsample=256,
    template_blocks=1,
    template_c_hidden_tri_att=16,
    template_c_hidden_tri_mul=32,
    template_no_heads=4,
    atom_enc_c_hidden=16,
    atom_enc_no_heads=2,
    atom_enc_no_blocks=2,
    dit_blocks=6,
    dit_no_heads=8,
    dit_c_hidden=32,
    conf_pairformer_blocks=2,
    no_samples=8,
)

PRESETS = {"ffold-base": FFOLD_BASE, "ffold-mini": FFOLD_MINI}


def get_config(name: str = "ffold-base") -> dict[str, Any]:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return copy.deepcopy(PRESETS[name])


def flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(d, dict):
        return {prefix: d}
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, p))
        else:
            out[p] = v
    return out
