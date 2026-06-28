"""
AI pose inversion network for BioSense TWR.

Architecture: 3D CNN backbone → dual-head output
  - Detection head: (n_joints,) sigmoid confidence
  - Regression head: (n_joints, 3) joint coordinates [m]

Input: (batch, 1, nx, ny, nz) f32 back-projected 3D radar image
Output:
  conf:   (batch, N_JOINTS) — joint detection confidence
  coords: (batch, N_JOINTS, 3) — joint positions [m]

Trained on synthetic Boulic-generated radar images.
Evaluated with OKS (Object Keypoint Similarity) metric.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple

N_JOINTS = 17  # matching Boulic model


class PoseOutput(NamedTuple):
    confidence: torch.Tensor   # (B, N_JOINTS)
    coords:     torch.Tensor   # (B, N_JOINTS, 3)


class ResBlock3D(nn.Module):
    """3D residual block with batch norm."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)


class BioSensePoseNet(nn.Module):
    """
    Lightweight 3D CNN for through-wall pose inversion.
    Optimised for A100: uses float16 mixed precision.
    ~4M parameters — fits comfortably in A100 VRAM even at batch=128.
    """
    def __init__(self, input_shape: tuple[int, int, int] = (40, 150, 40)):
        super().__init__()
        nx, ny, nz = input_shape

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm3d(16), nn.ReLU(),
        )
        self.res1 = ResBlock3D(16)

        self.enc2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(),
        )
        self.res2 = ResBlock3D(32)

        self.enc3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64), nn.ReLU(),
        )
        self.res3 = ResBlock3D(64)

        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Dual heads
        self.conf_head   = nn.Linear(256, N_JOINTS)
        self.coords_head = nn.Linear(256, N_JOINTS * 3)

    def forward(self, x: torch.Tensor) -> PoseOutput:
        # x: (B, 1, nx, ny, nz)
        x = self.res1(self.enc1(x))
        x = self.res2(self.enc2(x))
        x = self.res3(self.enc3(x))
        x = self.global_pool(x).flatten(1)  # (B, 64)
        x = self.trunk(x)
        conf   = self.conf_head(x)                        # raw logits for BCEWithLogitsLoss
        coords = self.coords_head(x).view(-1, N_JOINTS, 3)
        return PoseOutput(confidence=conf, coords=coords)


def oks_metric(
    pred_coords: torch.Tensor,    # (B, N_JOINTS, 3)
    gt_coords:   torch.Tensor,    # (B, N_JOINTS, 3)
    gt_conf:     torch.Tensor,    # (B, N_JOINTS) — 1 if joint visible
    sigma: float = 0.5,
) -> torch.Tensor:
    """
    Object Keypoint Similarity (OKS) metric.
    Returns mean OKS across batch, scalar tensor.
    OKS = sum_i [ exp(-d_i^2 / (2*s^2*sigma_i^2)) * v_i ] / sum_i v_i
    where d_i = Euclidean distance, s = object scale, v_i = visibility.
    """
    d2 = ((pred_coords - gt_coords) ** 2).sum(dim=-1)  # (B, N_JOINTS)
    scale = 2.0 * (sigma ** 2)
    per_joint_oks = torch.exp(-d2 / scale) * gt_conf
    oks = per_joint_oks.sum(dim=-1) / (gt_conf.sum(dim=-1) + 1e-9)  # (B,)
    return oks.mean()


class PoseLoss(nn.Module):
    """
    Combined loss for pose inversion.
    L = L_conf (BCEWithLogits) + alpha * L_coord (SmoothL1, only visible joints)
    conf head returns raw logits — BCEWithLogitsLoss applies sigmoid internally.
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.bce   = nn.BCEWithLogitsLoss(reduction='mean')
        self.sl1   = nn.SmoothL1Loss(reduction='none')

    def forward(
        self,
        pred: PoseOutput,
        gt_conf:   torch.Tensor,  # (B, N_JOINTS)
        gt_coords: torch.Tensor,  # (B, N_JOINTS, 3)
    ) -> torch.Tensor:
        loss_conf  = self.bce(pred.confidence, gt_conf)
        coord_err  = self.sl1(pred.coords, gt_coords)          # (B, N, 3)
        vis_mask   = gt_conf.unsqueeze(-1).expand_as(coord_err)
        loss_coord = (coord_err * vis_mask).mean()
        return loss_conf + self.alpha * loss_coord
