"""F-Fold model.

Assembles the AlphaFold3 trunk at the hyperparameters verified in
`check_alignment.py`: input embedder, MSA module, template pair stack,
Pairformer, diffusion module (conditioning + atom encoder/decoder +
diffusion transformer) and the five output heads.

Scope note: the *data* pipeline AF3 needs (CCD ligand graphs, MSA search,
template search) is not part of this package. `FFold.forward` runs the
protein-only, token-level path: sequence -> trunk -> distogram / pLDDT /
PAE / PDE / experimentally-resolved. The atom-level encoder and decoder are
instantiated at full config shape so parameter counts are meaningful, and
are exercised by the diffusion path when atom features are supplied.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # not auto-imported by `import torch`

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layers import (  # noqa: E402
    AdaLayerNorm, AttentionPairBias, ConditionedTransition, LayerNorm, Linear,
    Transition, TriangleAttention, TriangleMultiplication,
)

RESTYPES = "ARNDCQEGHILKMFPSTWYVX"
RESTYPE_IDX = {a: i for i, a in enumerate(RESTYPES)}
N_RESTYPE = len(RESTYPES)


# ---------------------------------------------------------------- Pairformer
class PairformerBlock(nn.Module):
    def __init__(self, c: dict[str, Any], with_single: bool = True):
        super().__init__()
        c_s, c_z = c["c_s"], c["c_z"]
        self.with_single = with_single
        self.tri_mul_out = TriangleMultiplication(c_z, c["c_hidden_mul"], True)
        self.tri_mul_in = TriangleMultiplication(c_z, c["c_hidden_mul"], False)
        self.tri_att_start = TriangleAttention(
            c_z, c["c_hidden_pair_att"], c["no_heads_pair"], starting=True)
        self.tri_att_end = TriangleAttention(
            c_z, c["c_hidden_pair_att"], c["no_heads_pair"], starting=False)
        self.pair_transition = Transition(c_z, c["transition_n"])
        self.dropout = nn.Dropout(c["pair_dropout"])
        if with_single:
            self.attn_pair_bias = AttentionPairBias(
                c_s, c_z, c["c_hidden_pair_bias"], c["no_heads_pair_bias"])
            self.single_transition = Transition(c_s, c["transition_n"])

    def forward(self, s, z, mask=None, pair_mask=None):
        z = z + self.dropout(self.tri_mul_out(z, pair_mask))
        z = z + self.dropout(self.tri_mul_in(z, pair_mask))
        z = z + self.dropout(self.tri_att_start(z, mask))
        z = z + self.dropout(self.tri_att_end(z, mask))
        z = z + self.pair_transition(z)
        if self.with_single:
            s = s + self.attn_pair_bias(s, z, mask=mask)
            s = s + self.single_transition(s)
        return s, z


class PairformerStack(nn.Module):
    def __init__(self, c: dict[str, Any], with_single: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList(
            [PairformerBlock(c, with_single) for _ in range(c["no_blocks"])])
        self.grad_checkpoint = False

    def forward(self, s, z, mask=None, pair_mask=None):
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                s, z = torch.utils.checkpoint.checkpoint(
                    blk, s, z, mask, pair_mask, use_reentrant=False)
            else:
                s, z = blk(s, z, mask, pair_mask)
        return s, z


# ------------------------------------------------------------- MSA / template
class OuterProductMean(nn.Module):
    def __init__(self, c_m: int, c_z: int, c_hidden: int):
        super().__init__()
        self.ln = LayerNorm(c_m)
        self.linear_a = Linear(c_m, c_hidden, bias=False)
        self.linear_b = Linear(c_m, c_hidden, bias=False)
        self.linear_out = Linear(c_hidden * c_hidden, c_z, init="final")

    def forward(self, m):                       # m: [..., S, L, c_m]
        m = self.ln(m)
        a, b = self.linear_a(m), self.linear_b(m)
        o = torch.einsum("...sic,...sjd->...ijcd", a, b) / m.shape[-3]
        return self.linear_out(o.flatten(start_dim=-2))


class MSABlock(nn.Module):
    def __init__(self, c: dict[str, Any]):
        super().__init__()
        c_m, c_z = c["c_m"], c["c_z"]
        self.opm = OuterProductMean(c_m, c_z, c["c_hidden_opm"])
        self.msa_ln = LayerNorm(c_m)
        self.linear_msa_v = Linear(c_m, c["c_hidden_msa_att"] * c["no_heads_msa"],
                                   bias=False)
        self.linear_msa_b = Linear(c_z, c["no_heads_msa"], bias=False)
        self.linear_msa_g = Linear(c_m, c["c_hidden_msa_att"] * c["no_heads_msa"],
                                   bias=False, init="gating")
        self.linear_msa_o = Linear(c["c_hidden_msa_att"] * c["no_heads_msa"],
                                   c_m, bias=False, init="final")
        self.ln_z_for_msa = LayerNorm(c_z, create_offset=False)
        self.msa_transition = Transition(c_m, c["transition_n"])
        self.tri_mul_out = TriangleMultiplication(c_z, c["c_hidden_mul"], True)
        self.tri_mul_in = TriangleMultiplication(c_z, c["c_hidden_mul"], False)
        self.tri_att_start = TriangleAttention(
            c_z, c["c_hidden_pair_att"], c["no_heads_pair"], starting=True)
        self.tri_att_end = TriangleAttention(
            c_z, c["c_hidden_pair_att"], c["no_heads_pair"], starting=False)
        self.pair_transition = Transition(c_z, c["transition_n"])
        self.h, self.c = c["no_heads_msa"], c["c_hidden_msa_att"]

    def forward(self, m, z, mask=None, pair_mask=None):
        z = z + self.opm(m)
        mn = self.msa_ln(m)
        *B, S, L, _ = mn.shape
        v = self.linear_msa_v(mn).view(*B, S, L, self.h, self.c)
        w = self.linear_msa_b(self.ln_z_for_msa(z))            # ...LLh
        if mask is not None:
            w = w + (1.0 - mask[..., None, :, None]) * -1e9
        w = torch.softmax(w, dim=-2)
        out = torch.einsum("...ijh,...sjhc->...sihc", w, v)
        g = torch.sigmoid(self.linear_msa_g(mn)).view(*B, S, L, self.h, self.c)
        m = m + self.linear_msa_o((out * g).reshape(*B, S, L, self.h * self.c))
        m = m + self.msa_transition(m)
        z = z + self.tri_mul_out(z, pair_mask)
        z = z + self.tri_mul_in(z, pair_mask)
        z = z + self.tri_att_start(z, mask)
        z = z + self.tri_att_end(z, mask)
        z = z + self.pair_transition(z)
        return m, z


class MSAModule(nn.Module):
    def __init__(self, c: dict[str, Any]):
        super().__init__()
        emb, mod = c["msa_module_embedder"], c["msa_module"]
        self.linear_m = Linear(emb["c_m_feats"], emb["c_m"], bias=False)
        self.linear_s = Linear(emb["c_s_input"], emb["c_m"], bias=False)
        self.blocks = nn.ModuleList([MSABlock(mod) for _ in range(mod["no_blocks"])])

    def forward(self, msa_feat, s_input, z, mask=None, pair_mask=None):
        m = self.linear_m(msa_feat) + self.linear_s(s_input)[..., None, :, :]
        for blk in self.blocks:
            m, z = blk(m, z, mask, pair_mask)
        return z


class TemplateBlock(nn.Module):
    def __init__(self, c: dict[str, Any]):
        super().__init__()
        c_t = c["c_t"]
        self.tri_mul_out = TriangleMultiplication(c_t, c["c_hidden_tri_mul"], True)
        self.tri_mul_in = TriangleMultiplication(c_t, c["c_hidden_tri_mul"], False)
        self.tri_att_start = TriangleAttention(
            c_t, c["c_hidden_tri_att"], c["no_heads"], starting=True)
        self.tri_att_end = TriangleAttention(
            c_t, c["c_hidden_tri_att"], c["no_heads"], starting=False)
        self.transition = Transition(c_t, c["pair_transition_n"])
        self.dropout = nn.Dropout(c["dropout_rate"])

    def forward(self, t, mask=None, pair_mask=None):
        t = t + self.dropout(self.tri_mul_out(t, pair_mask))
        t = t + self.dropout(self.tri_mul_in(t, pair_mask))
        t = t + self.dropout(self.tri_att_start(t, mask))
        t = t + self.dropout(self.tri_att_end(t, mask))
        return t + self.transition(t)


class TemplateModule(nn.Module):
    def __init__(self, c: dict[str, Any]):
        super().__init__()
        emb, stack = c["template_pair_embedder"], c["template_pair_stack"]
        self.linear_z = Linear(emb["c_in"], emb["c_out"], bias=False)
        self.linear_dgram = Linear(emb["c_dgram"], emb["c_out"], bias=False)
        self.linear_aatype = Linear(emb["c_aatype"] * 2, emb["c_out"], bias=False)
        self.blocks = nn.ModuleList(
            [TemplateBlock(stack) for _ in range(stack["no_blocks"])])
        self.ln = LayerNorm(emb["c_out"])
        self.linear_out = Linear(emb["c_out"], emb["c_in"], bias=False,
                                 init="final")

    def forward(self, z, tmpl_dgram, tmpl_aatype, mask=None, pair_mask=None):
        t = self.linear_z(z) + self.linear_dgram(tmpl_dgram) + \
            self.linear_aatype(tmpl_aatype)
        for blk in self.blocks:
            t = blk(t, mask, pair_mask)
        return self.linear_out(F.relu(self.ln(t)))


# ------------------------------------------------------------ atom attention
class AtomTransformerBlock(nn.Module):
    def __init__(self, c_atom, c_atom_pair, c_hidden, no_heads, n_transition,
                 c_s):
        super().__init__()
        self.att = AttentionPairBias(c_atom, c_atom_pair, c_hidden, no_heads,
                                     c_s=c_s, use_ada_layer_norm=True)
        self.trans = ConditionedTransition(c_atom, c_s, n_transition)

    def forward(self, q, p, c):
        q = q + self.att(q, p, s=c)
        return q + self.trans(q, c)


class AtomAttentionEncoder(nn.Module):
    """Sequence-local atom attention (AF3 Alg. 5). Shapes follow the config."""

    def __init__(self, c: dict[str, Any]):
        super().__init__()
        self.c_atom, self.c_atom_pair = c["c_atom"], c["c_atom_pair"]
        self.c_token, self.n_query, self.n_key = c["c_token"], c["n_query"], c["n_key"]
        ref = c["c_atom_ref"]
        self.linear_ref_pos = Linear(3, c["c_atom"], bias=False)
        self.linear_ref_element = Linear(ref["element"], c["c_atom"], bias=False)
        self.linear_ref_charge = Linear(1, c["c_atom"], bias=False)
        self.linear_ref_name = Linear(ref["name_chars"], c["c_atom"], bias=False)
        self.linear_pair_dist = Linear(3, c["c_atom_pair"], bias=False)
        self.linear_pair_inv = Linear(1, c["c_atom_pair"], bias=False)
        self.linear_pair_c = Linear(c["c_atom"], c["c_atom_pair"], bias=False)
        self.linear_s_trunk = Linear(c["c_s"], c["c_atom"], bias=False)
        self.linear_z_trunk = Linear(c["c_z"], c["c_atom_pair"], bias=False)
        self.linear_noisy = Linear(3, c["c_atom"], bias=False)
        self.pair_mlp = nn.Sequential(
            nn.ReLU(), Linear(c["c_atom_pair"], c["c_atom_pair"], bias=False),
            nn.ReLU(), Linear(c["c_atom_pair"], c["c_atom_pair"], bias=False,
                              init="final"))
        self.blocks = nn.ModuleList([
            AtomTransformerBlock(c["c_atom"], c["c_atom_pair"], c["c_hidden"],
                                 c["no_heads"], c["n_transition"], c["c_atom"])
            for _ in range(c["no_blocks"])])
        self.linear_token = Linear(c["c_atom"], c["c_token"], bias=False)

    def forward(self, ref_pos, ref_element, ref_charge, ref_name,
                atom_to_token, noisy_pos=None, s_trunk=None, z_trunk=None):
        c = (self.linear_ref_pos(ref_pos)
             + self.linear_ref_element(ref_element)
             + self.linear_ref_charge(ref_charge)
             + self.linear_ref_name(ref_name))
        q = c.clone()
        if noisy_pos is not None:
            q = q + self.linear_noisy(noisy_pos)
        if s_trunk is not None:
            idx_s = atom_to_token[..., None].expand(*atom_to_token.shape,
                                                    s_trunk.shape[-1])
            q = q + self.linear_s_trunk(s_trunk.gather(-2, idx_s))
        d = ref_pos[..., :, None, :] - ref_pos[..., None, :, :]
        p = self.linear_pair_dist(d) + \
            self.linear_pair_inv(1.0 / (1.0 + d.pow(2).sum(-1, keepdim=True)))
        p = p + self.linear_pair_c(c)[..., :, None, :] + \
            self.linear_pair_c(c)[..., None, :, :]
        if z_trunk is not None:
            # zt[b, i, j] = z_trunk[b, a2t[b, i], a2t[b, j]]
            n_at = atom_to_token.shape[-1]
            ti = atom_to_token[..., :, None].expand(*atom_to_token.shape, n_at)
            tj = atom_to_token[..., None, :].expand_as(ti)
            if z_trunk.dim() == 4:                       # batched [B, T, T, C]
                b = torch.arange(z_trunk.shape[0], device=z_trunk.device)
                b = b[:, None, None].expand_as(ti)
                zt = z_trunk[b, ti, tj]
            else:                                        # unbatched [T, T, C]
                zt = z_trunk[ti, tj]
            p = p + self.linear_z_trunk(zt)
        p = p + self.pair_mlp(p)
        for blk in self.blocks:
            q = blk(q, p, c)
        a = F.relu(self.linear_token(q))
        n_tok = int(atom_to_token.max().item()) + 1
        out = a.new_zeros(*a.shape[:-2], n_tok, a.shape[-1])
        idx = atom_to_token[..., None].expand(*atom_to_token.shape, a.shape[-1])
        out = out.scatter_add(-2, idx, a)
        cnt = a.new_zeros(*a.shape[:-2], n_tok, 1).scatter_add(
            -2, atom_to_token[..., None], torch.ones_like(a[..., :1]))
        return out / cnt.clamp(min=1.0), q, c, p


class AtomAttentionDecoder(nn.Module):
    def __init__(self, c: dict[str, Any]):
        super().__init__()
        self.linear_a = Linear(c["c_token"], c["c_atom"], bias=False)
        self.blocks = nn.ModuleList([
            AtomTransformerBlock(c["c_atom"], c["c_atom_pair"], c["c_hidden"],
                                 c["no_heads"], c["n_transition"], c["c_atom"])
            for _ in range(c["no_blocks"])])
        self.ln = LayerNorm(c["c_atom"])
        self.linear_out = Linear(c["c_atom"], 3, bias=False, init="final")

    def forward(self, a_token, q_skip, c_skip, p, atom_to_token):
        idx = atom_to_token[..., None].expand(*atom_to_token.shape,
                                              a_token.shape[-1])
        q = self.linear_a(a_token.gather(-2, idx)) + q_skip
        for blk in self.blocks:
            q = blk(q, p, c_skip)
        return self.linear_out(self.ln(q))


# ------------------------------------------------------------------ diffusion
class FourierEmbedding(nn.Module):
    def __init__(self, c: int, seed: int = 42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("w", torch.randn(c, generator=g))
        self.register_buffer("b", torch.rand(c, generator=g))

    def forward(self, t):
        return torch.cos(2 * math.pi * (t[..., None] * self.w + self.b))


class RelativePositionEncoding(nn.Module):
    def __init__(self, c_z: int, max_rel_idx: int, max_rel_chain: int):
        super().__init__()
        self.max_rel_idx, self.max_rel_chain = max_rel_idx, max_rel_chain
        n = (2 * max_rel_idx + 2) + (2 * max_rel_idx + 2) + 1 + \
            (2 * max_rel_chain + 2)
        self.linear = Linear(n, c_z, bias=False)

    def forward(self, residue_index, asym_id=None):
        L = residue_index.shape[-1]
        if asym_id is None:
            asym_id = torch.zeros_like(residue_index)
        same = (asym_id[..., :, None] == asym_id[..., None, :])
        d = residue_index[..., :, None] - residue_index[..., None, :]
        d = torch.where(same, d.clamp(-self.max_rel_idx, self.max_rel_idx)
                        + self.max_rel_idx, torch.full_like(d, 2 * self.max_rel_idx + 1))
        a_rel_pos = F.one_hot(d, 2 * self.max_rel_idx + 2).float()
        a_rel_tok = a_rel_pos.clone()
        dc = (asym_id[..., :, None] - asym_id[..., None, :]).clamp(
            -self.max_rel_chain, self.max_rel_chain) + self.max_rel_chain
        a_rel_chain = F.one_hot(dc, 2 * self.max_rel_chain + 2).float()
        feat = torch.cat([a_rel_pos, a_rel_tok, same[..., None].float(),
                          a_rel_chain], dim=-1)
        return self.linear(feat)


class DiffusionConditioning(nn.Module):
    def __init__(self, c: dict[str, Any]):
        super().__init__()
        c_s, c_z = c["c_s"], c["c_z"]
        self.sigma_data = c["sigma_data"]
        self.rel_pos = RelativePositionEncoding(
            c_z, c["max_relative_idx"], c["max_relative_chain"])
        self.ln_z = LayerNorm(2 * c_z)
        self.linear_z = Linear(2 * c_z, c_z, bias=False)
        self.z_trans = nn.ModuleList([Transition(c_z, 2) for _ in range(2)])
        self.ln_s = LayerNorm(c_s + c["c_s_input"])
        self.linear_s = Linear(c_s + c["c_s_input"], c_s, bias=False)
        self.fourier = FourierEmbedding(c["c_fourier_emb"],
                                        c["seed_fourier_emb"])
        self.ln_n = LayerNorm(c["c_fourier_emb"])
        self.linear_n = Linear(c["c_fourier_emb"], c_s, bias=False)
        self.s_trans = nn.ModuleList([Transition(c_s, 2) for _ in range(2)])

    def forward(self, t, s_trunk, s_inputs, z_trunk, residue_index, asym_id=None):
        z = self.linear_z(self.ln_z(torch.cat(
            [z_trunk, self.rel_pos(residue_index, asym_id)], dim=-1)))
        for tr in self.z_trans:
            z = z + tr(z)
        s = self.linear_s(self.ln_s(torch.cat([s_trunk, s_inputs], dim=-1)))
        n = self.fourier(0.25 * torch.log(t / self.sigma_data))
        s = s + self.linear_n(self.ln_n(n))[..., None, :]
        for tr in self.s_trans:
            s = s + tr(s)
        return s, z


class DiffusionTransformer(nn.Module):
    def __init__(self, c: dict[str, Any]):
        super().__init__()
        self.blocks = nn.ModuleList([
            AtomTransformerBlock(c["c_a"], c["c_z"], c["c_hidden"],
                                 c["no_heads"], c["n_transition"], c["c_s"])
            for _ in range(c["no_blocks"])])

    def forward(self, a, s, z):
        for blk in self.blocks:
            a = blk(a, z, s)
        return a


# ---------------------------------------------------------------------- heads
class ConfidenceHeads(nn.Module):
    def __init__(self, c: dict[str, Any]):
        super().__init__()
        emb = c["pairformer_embedding"]
        self.no_bin = emb["no_bin"]
        self.min_bin, self.max_bin = emb["min_bin"], emb["max_bin"]
        self.linear_s_i = Linear(emb["c_s_input"], emb["c_z"], bias=False)
        self.linear_s_j = Linear(emb["c_s_input"], emb["c_z"], bias=False)
        self.linear_dgram = Linear(emb["no_bin"], emb["c_z"], bias=False)
        self.pairformer = PairformerStack(emb["pairformer"], with_single=True)
        self.linear_pae = Linear(c["pae"]["c_z"], c["pae"]["c_out"], init="final")
        self.linear_pde = Linear(c["pde"]["c_z"], c["pde"]["c_out"], init="final")
        self.linear_plddt = Linear(
            c["lddt"]["c_s"], c["lddt"]["c_out"] * c["lddt"]["max_atoms_per_token"],
            init="final")
        self.linear_resolved = Linear(
            c["experimentally_resolved"]["c_s"],
            c["experimentally_resolved"]["c_out"]
            * c["experimentally_resolved"]["max_atoms_per_token"], init="final")
        self.n_atom = c["lddt"]["max_atoms_per_token"]
        self.n_plddt = c["lddt"]["c_out"]

    def forward(self, s, z, s_inputs, pred_dist, mask=None):
        edges = torch.linspace(self.min_bin, self.max_bin, self.no_bin,
                               device=z.device)
        d = (pred_dist[..., None] > edges).sum(-1).clamp(max=self.no_bin - 1)
        z = z + self.linear_s_i(s_inputs)[..., :, None, :] \
              + self.linear_s_j(s_inputs)[..., None, :, :] \
              + self.linear_dgram(F.one_hot(d, self.no_bin).float())
        s, z = self.pairformer(s, z, mask)
        *B, L, _ = s.shape
        return {
            "pae_logits": self.linear_pae(z),
            "pde_logits": self.linear_pde(z + z.transpose(-3, -2)),
            "plddt_logits": self.linear_plddt(s).view(*B, L, self.n_atom,
                                                      self.n_plddt),
            "resolved_logits": self.linear_resolved(s).view(*B, L, self.n_atom, 2),
        }


# ----------------------------------------------------------------------- model
class FFold(nn.Module):
    def __init__(self, cfg: dict[str, Any], use_msa: bool = False,
                 use_template: bool = False):
        super().__init__()
        self.cfg = cfg
        self.use_msa, self.use_template = use_msa, use_template
        ie = cfg["input_embedder"]
        c_s, c_z, c_s_in = ie["c_s"], ie["c_z"], ie["c_s_input"]
        self.num_recycles = cfg["shared"]["num_recycles"]

        self.linear_restype = Linear(N_RESTYPE, c_s_in, bias=False)
        self.atom_attn_enc = AtomAttentionEncoder(ie["atom_attn_enc"])
        self.linear_atom_token = Linear(ie["atom_attn_enc"]["c_token"], c_s_in,
                                        bias=False)
        self.linear_s_init = Linear(c_s_in, c_s, bias=False)
        self.linear_z_i = Linear(c_s_in, c_z, bias=False)
        self.linear_z_j = Linear(c_s_in, c_z, bias=False)
        self.rel_pos = RelativePositionEncoding(
            c_z, ie["max_relative_idx"], ie["max_relative_chain"])

        self.ln_recycle_s = LayerNorm(c_s)
        self.linear_recycle_s = Linear(c_s, c_s, bias=False, init="final")
        self.ln_recycle_z = LayerNorm(c_z)
        self.linear_recycle_z = Linear(c_z, c_z, bias=False, init="final")

        if use_msa:
            self.msa_module = MSAModule(cfg["msa"])
        if use_template:
            self.template_module = TemplateModule(cfg["template"])
        self.pairformer = PairformerStack(cfg["pairformer"], with_single=True)

        dm = cfg["diffusion_module"]
        self.diffusion_conditioning = DiffusionConditioning(
            dm["diffusion_conditioning"])
        self.diffusion_atom_enc = AtomAttentionEncoder(dm["atom_attn_enc"])
        self.diffusion_transformer = DiffusionTransformer(
            dm["diffusion_transformer"])
        self.diffusion_atom_dec = AtomAttentionDecoder(dm["atom_attn_dec"])

        lm = cfg["loss_module"]["distogram"]
        self.n_dist_bins = lm["no_bins"]
        self.dist_bin_min, self.dist_bin_max = lm["bin_min"], lm["bin_max"]
        self.linear_distogram = Linear(cfg["heads"]["distogram"]["c_z"],
                                       cfg["heads"]["distogram"]["c_out"],
                                       init="final")
        self.confidence = ConfidenceHeads(cfg["heads"])

    # -- trunk ------------------------------------------------------------
    def embed(self, aatype, residue_index, asym_id=None):
        s_inputs = self.linear_restype(F.one_hot(aatype, N_RESTYPE).float())
        s = self.linear_s_init(s_inputs)
        z = self.linear_z_i(s_inputs)[..., :, None, :] + \
            self.linear_z_j(s_inputs)[..., None, :, :] + \
            self.rel_pos(residue_index, asym_id)
        return s_inputs, s, z

    def forward(self, aatype, residue_index, asym_id=None, mask=None,
                num_recycles: int | None = None):
        s_inputs, s_init, z_init = self.embed(aatype, residue_index, asym_id)
        pair_mask = None if mask is None else mask[..., :, None] * mask[..., None, :]
        s_prev = torch.zeros_like(s_init)
        z_prev = torch.zeros_like(z_init)
        n_cycle = self.num_recycles if num_recycles is None else num_recycles
        for _ in range(max(1, n_cycle)):
            s = s_init + self.linear_recycle_s(self.ln_recycle_s(s_prev))
            z = z_init + self.linear_recycle_z(self.ln_recycle_z(z_prev))
            s, z = self.pairformer(s, z, mask, pair_mask)
            s_prev, z_prev = s.detach(), z.detach()
        dist_logits = self.linear_distogram(z + z.transpose(-3, -2))
        exp_dist = (dist_logits.softmax(-1)
                    * self.distogram_bin_centres(z.device)).sum(-1)
        conf = self.confidence(s, z, s_inputs, exp_dist, mask)
        return {"single": s, "pair": z, "distogram_logits": dist_logits,
                "expected_distance": exp_dist, **conf}

    def distogram_bin_centres(self, device=None) -> torch.Tensor:
        """Centres matching the `n_bins - 1` edges used to build the labels.

        Edges are linspace(bin_min, bin_max, n_bins - 1), so bin i spans
        (e_{i-1}, e_i] with width w; the two outer bins are open-ended and
        are represented by their adjacent half-bin.
        """
        nb = self.n_dist_bins
        w = (self.dist_bin_max - self.dist_bin_min) / (nb - 2)
        i = torch.arange(nb, device=device, dtype=torch.float32)
        return self.dist_bin_min + (i - 0.5) * w

    def plddt(self, out) -> torch.Tensor:
        """Per-residue pLDDT in [0, 100] from the 50-bin head."""
        p = out["plddt_logits"].softmax(-1)
        n = p.shape[-1]
        centres = torch.linspace(1 / (2 * n), 1 - 1 / (2 * n), n, device=p.device)
        return (p * centres).sum(-1).mean(-1) * 100.0


def param_table(model: nn.Module) -> dict[str, int]:
    top: dict[str, int] = {}
    for name, p in model.named_parameters():
        top[name.split(".")[0]] = top.get(name.split(".")[0], 0) + p.numel()
    top["TOTAL"] = sum(top.values())
    return top
