"""F-Fold 2.0 research architecture.

This module extends the static F-Fold trunk without changing its public API.
It adds experimental-density conditioning, endpoint-conditioned conformational
path generation, differentiable physics diagnostics and active-learning scores.

The implementation is an executable architecture scaffold. It does not ship
trained dynamic weights and does not claim molecular-dynamics accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .model import FFold
except ImportError:  # pragma: no cover
    from model import FFold


@dataclass(frozen=True)
class DynamicsConfig:
    density_channels: int = 32
    path_hidden: int = 128
    path_frames: int = 16
    max_step: float = 4.0
    bond_target: float = 3.8
    clash_cutoff: float = 2.0


class DensityEncoder3D(nn.Module):
    """Encode a cryo-EM/cryo-ET density crop into a global condition vector."""

    def __init__(self, out_dim: int, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, width, 3, padding=1), nn.SiLU(),
            nn.Conv3d(width, width, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv3d(width, width * 2, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.proj = nn.Linear(width * 2, out_dim)

    def forward(self, density: torch.Tensor) -> torch.Tensor:
        if density.ndim != 5 or density.shape[1] != 1:
            raise ValueError("density must have shape [batch, 1, depth, height, width]")
        return self.proj(self.net(density).flatten(1))


class ConformationalPathDecoder(nn.Module):
    """Generate a smooth path while exactly preserving supplied endpoints."""

    def __init__(self, token_dim: int, condition_dim: int, hidden: int = 128,
                 max_step: float = 4.0):
        super().__init__()
        self.max_step = max_step
        self.net = nn.Sequential(
            nn.Linear(token_dim + condition_dim + 7, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3),
        )

    def forward(self, token: torch.Tensor, condition: torch.Tensor,
                start: torch.Tensor, end: torch.Tensor,
                n_frames: int) -> torch.Tensor:
        if n_frames < 2:
            raise ValueError("n_frames must be at least 2")
        if start.shape != end.shape or start.shape[-1] != 3:
            raise ValueError("start and end must have the same [..., residues, 3] shape")
        b, length, _ = start.shape
        t = torch.linspace(0, 1, n_frames, device=start.device, dtype=start.dtype)
        tv = t.view(1, n_frames, 1, 1).expand(b, -1, length, -1)
        base = start[:, None] * (1 - tv) + end[:, None] * tv
        tok = token[:, None].expand(-1, n_frames, -1, -1)
        cond = condition[:, None, None].expand(-1, n_frames, length, -1)
        geom = torch.cat([start[:, None].expand_as(base),
                          end[:, None].expand_as(base), tv], dim=-1)
        delta = torch.tanh(self.net(torch.cat([tok, cond, geom], dim=-1)))
        envelope = 4.0 * tv * (1.0 - tv)
        return base + self.max_step * envelope * delta


class EnergyForceHead(nn.Module):
    """Predict frame energy and per-residue forces from latent tokens and paths."""

    def __init__(self, token_dim: int, hidden: int = 128):
        super().__init__()
        self.local = nn.Sequential(
            nn.Linear(token_dim + 3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.energy = nn.Linear(hidden, 1)
        self.force = nn.Linear(hidden, 3)

    def forward(self, token: torch.Tensor, path: torch.Tensor,
                mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([token[:, None].expand(-1, path.shape[1], -1, -1), path], -1)
        h = self.local(x)
        local_e = self.energy(h).squeeze(-1)
        forces = self.force(h)
        if mask is None:
            energy = local_e.mean(-1)
        else:
            energy = ((local_e * mask[:, None]).sum(-1)
                      / mask.sum(-1, keepdim=True).clamp_min(1))
            forces = forces * mask[:, None, :, None]
        return energy, forces


class FFold2(nn.Module):
    """Multimodal dynamics wrapper around an existing :class:`FFold` trunk."""

    def __init__(self, cfg: dict[str, Any], dynamics: DynamicsConfig | None = None,
                 trunk: FFold | None = None):
        super().__init__()
        self.dynamics = dynamics or DynamicsConfig()
        self.trunk = trunk or FFold(cfg)
        c_s = cfg["shared"]["c_s"]
        self.density_encoder = DensityEncoder3D(c_s, self.dynamics.density_channels)
        self.no_density = nn.Parameter(torch.zeros(c_s))
        self.path_decoder = ConformationalPathDecoder(
            c_s, c_s, self.dynamics.path_hidden, self.dynamics.max_step)
        self.energy_force = EnergyForceHead(c_s, self.dynamics.path_hidden)
        self.state_logit = nn.Linear(c_s, 1)
        self.uncertainty = nn.Sequential(
            nn.Linear(c_s, c_s), nn.SiLU(), nn.Linear(c_s, 1))

    def forward(self, aatype: torch.Tensor, residue_index: torch.Tensor,
                start_coords: torch.Tensor, end_coords: torch.Tensor,
                density: torch.Tensor | None = None,
                mask: torch.Tensor | None = None,
                n_frames: int | None = None) -> dict[str, torch.Tensor]:
        static = self.trunk(aatype, residue_index, mask=mask)
        token = static["single"]
        condition = (self.density_encoder(density) if density is not None
                     else self.no_density[None].expand(token.shape[0], -1))
        frames = n_frames or self.dynamics.path_frames
        path = self.path_decoder(token, condition, start_coords, end_coords, frames)
        energy, forces = self.energy_force(token, path, mask)
        pooled = token.mean(-2) if mask is None else (
            (token * mask[..., None]).sum(-2)
            / mask.sum(-1, keepdim=True).clamp_min(1))
        return {
            **static, "density_condition": condition, "path": path,
            "frame_energy": energy, "forces": forces,
            "state_logits": self.state_logit(token).squeeze(-1),
            "epistemic_uncertainty": F.softplus(
                self.uncertainty(pooled).squeeze(-1)),
        }


def physics_losses(path: torch.Tensor, forces: torch.Tensor | None = None,
                   mask: torch.Tensor | None = None, bond_target: float = 3.8,
                   clash_cutoff: float = 2.0) -> dict[str, torch.Tensor]:
    """Differentiable geometry diagnostics for token-level C-alpha paths."""
    if path.ndim != 4 or path.shape[-1] != 3:
        raise ValueError("path must have shape [batch, frames, residues, 3]")
    bond = torch.linalg.vector_norm(path[..., 1:, :] - path[..., :-1, :], dim=-1)
    bond_mask = (torch.ones_like(bond) if mask is None
                 else mask[:, None, 1:] * mask[:, None, :-1])
    bond_loss = (((bond - bond_target) ** 2) * bond_mask).sum() \
        / bond_mask.sum().clamp_min(1)

    distances = torch.cdist(path, path)
    length = path.shape[-2]
    separated = torch.ones(length, length, device=path.device, dtype=path.dtype)
    separated.fill_diagonal_(0)
    if length > 1:
        separated -= torch.diag(torch.ones(length - 1, device=path.device), 1)
        separated -= torch.diag(torch.ones(length - 1, device=path.device), -1)
    pair_mask = separated.clamp_min(0)[None, None]
    if mask is not None:
        pair_mask = pair_mask * mask[:, None, :, None] * mask[:, None, None, :]
    clash = (F.relu(clash_cutoff - distances) ** 2 * pair_mask).sum() \
        / pair_mask.sum().clamp_min(1)

    velocity = path[:, 1:] - path[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    smoothness = acceleration.square().mean() if acceleration.numel() else path.new_zeros(())
    result = {"bond": bond_loss, "clash": clash, "smoothness": smoothness}
    if forces is not None:
        if forces.shape != path.shape:
            raise ValueError("forces must match path shape")
        result["force_smoothness"] = (forces[:, 1:] - forces[:, :-1]).square().mean()
    return result


def acquisition_score(uncertainty: torch.Tensor, state_logits: torch.Tensor,
                      novelty: torch.Tensor | None = None) -> torch.Tensor:
    """Rank experiments by uncertainty, state ambiguity and optional novelty."""
    probability = state_logits.sigmoid()
    ambiguity = 1.0 - (probability - 0.5).abs() * 2.0
    score = uncertainty + ambiguity.mean(-1)
    return score if novelty is None else score + novelty

