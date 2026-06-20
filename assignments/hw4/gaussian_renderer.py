import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np


class GaussianRenderer(nn.Module):

    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.H = image_height
        self.W = image_width

        y, x = torch.meshgrid(torch.arange(image_height, dtype=torch.float32),
                              torch.arange(image_width, dtype=torch.float32),
                              indexing='ij')
        self.register_buffer('pixels', torch.stack([x, y], dim=-1))

    def compute_projection(
            self, means3D: torch.Tensor, covs3d: torch.Tensor, K: torch.Tensor,
            R: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = means3D.shape[0]

        cam_points = means3D @ R.T + t.unsqueeze(0)

        depths = cam_points[:, 2]

        screen_points = cam_points @ K.T
        means2D = screen_points[..., :2] / screen_points[...,
                                                         2:3].clamp(min=1e-4)

        J_proj = torch.zeros((N, 2, 3), device=means3D.device)
        fx = K[0, 0]
        fy = K[1, 1]
        x = cam_points[:, 0]
        y = cam_points[:, 1]
        z = cam_points[:, 2].clamp(min=1e-4)

        J_proj[:, 0, 0] = fx / z
        J_proj[:, 0, 2] = -fx * x / (z * z)
        J_proj[:, 1, 1] = fy / z
        J_proj[:, 1, 2] = -fy * y / (z * z)

        R_expand = R.unsqueeze(0)
        covs_cam = torch.matmul(
            R_expand, torch.matmul(covs3d, R_expand.transpose(-1, -2)))

        covs2D = torch.bmm(J_proj, torch.bmm(covs_cam, J_proj.permute(0, 2,
                                                                      1)))

        return means2D, covs2D, depths

    def compute_gaussian_values(self, means2D: torch.Tensor,
                                covs2D: torch.Tensor,
                                pixels: torch.Tensor) -> torch.Tensor:
        N = means2D.shape[0]
        H, W = pixels.shape[:2]

        dx = pixels.unsqueeze(0) - means2D.reshape(N, 1, 1, 2)

        eps = 1e-4
        covs2D = covs2D + eps * torch.eye(2, device=covs2D.device).unsqueeze(0)

        det = covs2D[:, 0, 0] * covs2D[:, 1, 1] - covs2D[:, 0,
                                                         1] * covs2D[:, 1, 0]
        det = torch.clamp(det, min=1e-6)

        inv_covs2D = torch.zeros_like(covs2D)
        inv_covs2D[:, 0, 0] = covs2D[:, 1, 1] / det
        inv_covs2D[:, 0, 1] = -covs2D[:, 0, 1] / det
        inv_covs2D[:, 1, 0] = -covs2D[:, 1, 0] / det
        inv_covs2D[:, 1, 1] = covs2D[:, 0, 0] / det

        vx = dx[..., 0]
        vy = dx[..., 1]
        a = inv_covs2D[:, 0, 0].view(N, 1, 1)
        b = inv_covs2D[:, 0, 1].view(N, 1, 1)
        d = inv_covs2D[:, 1, 1].view(N, 1, 1)

        power = -0.5 * (vx * vx * a + 2.0 * vx * vy * b + vy * vy * d)
        power = torch.clamp(power, max=0.0)

        norm = 1.0 / (2.0 * np.pi * torch.sqrt(det)).view(N, 1, 1)
        gaussian = norm * torch.exp(power)

        return gaussian

    def forward(self, means3D: torch.Tensor, covs3d: torch.Tensor,
                colors: torch.Tensor, opacities: torch.Tensor, K: torch.Tensor,
                R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        cam_points = means3D @ R.T + t.unsqueeze(0)
        depths = cam_points[:, 2]

        valid_mask = (depths > 0.1) & (depths < 50.0)

        if not valid_mask.any():
            return torch.zeros(self.H, self.W, 3, device=means3D.device)

        means3D = means3D[valid_mask]
        covs3d = covs3d[valid_mask]
        colors = colors[valid_mask]
        opacities = opacities[valid_mask]

        N = means3D.shape[0]

        means2D, covs2D, depths = self.compute_projection(
            means3D, covs3d, K, R, t)

        indices = torch.argsort(depths, dim=0, descending=False)
        means2D = means2D[indices]
        covs2D = covs2D[indices]
        colors = colors[indices]
        opacities = opacities[indices]

        gaussian_values = self.compute_gaussian_values(means2D, covs2D,
                                                       self.pixels)

        alphas = opacities.view(N, 1, 1) * gaussian_values
        colors = colors.view(N, 3, 1, 1).expand(-1, -1, self.H, self.W)
        colors = colors.permute(0, 2, 3, 1)

        alphas_clamped = torch.clamp(alphas, min=0.0, max=0.999)
        one_minus_alpha = 1.0 - alphas_clamped
        cumprod = torch.cumprod(one_minus_alpha, dim=0)

        T = torch.cat([
            torch.ones(1, self.H, self.W, device=alphas.device), cumprod[:-1]
        ],
                      dim=0)
        weights = alphas * T

        rendered = (weights.unsqueeze(-1) * colors).sum(dim=0)

        rendered = torch.nan_to_num(rendered, nan=0.0, posinf=1.0, neginf=0.0)

        return rendered
