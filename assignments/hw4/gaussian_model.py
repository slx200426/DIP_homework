import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class GaussianParameters:
    positions: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor
    covariance: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor


class GaussianModel(nn.Module):

    def __init__(self, points3D_xyz: torch.Tensor, points3D_rgb: torch.Tensor):

        super().__init__()

        downsample_rate = 5
        points3D_xyz = torch.as_tensor(points3D_xyz)[::downsample_rate]
        points3D_rgb = torch.as_tensor(points3D_rgb)[::downsample_rate]

        self.n_points = len(points3D_xyz)

        self._init_positions(points3D_xyz)
        self._init_rotations()
        self._init_scales(points3D_xyz)
        self._init_colors(points3D_rgb)
        self._init_opacities()

    def _init_positions(self, points3D_xyz: torch.Tensor) -> None:
        self.positions = nn.Parameter(
            torch.as_tensor(points3D_xyz, dtype=torch.float32))

    def _init_rotations(self) -> None:
        initial_rotations = torch.zeros((self.n_points, 4))
        initial_rotations[:, 0] = 1.0
        self.rotations = nn.Parameter(initial_rotations)

    def _init_scales(self, points3D_xyz: torch.Tensor) -> None:
        K = min(50, self.n_points - 1)

        dist_matrix = torch.cdist(points3D_xyz, points3D_xyz)

        val, _ = torch.topk(dist_matrix, k=K, largest=False, dim=1)

        mean_dists = torch.mean(val, dim=1, keepdim=True) * 2.
        mean_dists = mean_dists.clamp(0.2 * torch.median(mean_dists),
                                      3.0 * torch.median(mean_dists))
        print('init_scales', torch.min(mean_dists), torch.max(mean_dists))

        log_scales = torch.log(mean_dists)
        self.scales = nn.Parameter(log_scales.repeat(1, 3))

    def _init_colors(self, points3D_rgb: torch.Tensor) -> None:
        colors = torch.as_tensor(points3D_rgb, dtype=torch.float32) / 255.0
        colors = colors.clamp(0.001, 0.999)
        self.colors = nn.Parameter(torch.logit(colors))

    def _init_opacities(self) -> None:
        self.opacities = nn.Parameter(8.0 * torch.ones(
            (self.n_points, 1), dtype=torch.float32))

    def _compute_rotation_matrices(self) -> torch.Tensor:
        eps = 1e-8
        norm = torch.sqrt(
            torch.sum(self.rotations**2, dim=-1, keepdim=True) + eps)
        q = self.rotations / norm
        w, x, y, z = q.unbind(-1)

        R00 = 1 - 2 * y * y - 2 * z * z
        R01 = 2 * x * y - 2 * w * z
        R02 = 2 * x * z + 2 * w * y
        R10 = 2 * x * y + 2 * w * z
        R11 = 1 - 2 * x * x - 2 * z * z
        R12 = 2 * y * z - 2 * w * x
        R20 = 2 * x * z - 2 * w * y
        R21 = 2 * y * z + 2 * w * x
        R22 = 1 - 2 * x * x - 2 * y * y

        return torch.stack([R00, R01, R02, R10, R11, R12, R20, R21, R22],
                           dim=-1).reshape(-1, 3, 3)

    def compute_covariance(self) -> torch.Tensor:
        R = self._compute_rotation_matrices()

        scales = torch.exp(torch.clamp(self.scales, max=2.0))
        S = torch.diag_embed(scales)

        RS = R @ S
        Covs3d = RS @ RS.transpose(-1, -2)

        return Covs3d

    def get_gaussian_params(self) -> GaussianParameters:
        eps = 1e-8
        norm = torch.sqrt(
            torch.sum(self.rotations**2, dim=-1, keepdim=True) + eps)
        normalized_rotations = self.rotations / norm
        return GaussianParameters(positions=self.positions,
                                  colors=torch.sigmoid(self.colors),
                                  opacities=torch.sigmoid(self.opacities),
                                  covariance=self.compute_covariance(),
                                  rotations=normalized_rotations,
                                  scales=torch.exp(
                                      torch.clamp(self.scales, max=2.0)))

    def forward(self) -> Dict[str, torch.Tensor]:
        params = self.get_gaussian_params()
        return {
            'positions': params.positions,
            'covariance': params.covariance,
            'colors': params.colors,
            'opacities': params.opacities
        }
