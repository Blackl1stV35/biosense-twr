"""
MIMO antenna array geometry — BioSense foveal design.

Implements the star-nosed mole inspired foveal array:
  - Central high-density fovea (8 tx, 8 rx)
  - Sparse surround elements (4 tx, 4 rx offset)
  - Barn-owl asymmetric vertical offset on rx side

Also computes synthetic aperture positions for
SAR back-projection.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class MIMOArray:
    """
    BioSense foveal MIMO array.

    Coordinate system:
      x = horizontal (cross-range)
      y = along range (depth into wall)
      z = vertical
    """
    fovea_tx_spacing_m: float = 0.03     # 3 cm between foveal Tx
    fovea_rx_spacing_m: float = 0.03
    surround_tx_spacing_m: float = 0.12  # 12 cm surround Tx
    surround_rx_spacing_m: float = 0.12
    owl_offset_z_m: float = 0.15         # barn-owl vertical Rx offset

    def tx_positions(self) -> np.ndarray:
        """Returns (n_tx, 3) array of transmitter positions."""
        # 8 foveal Tx along x-axis
        fovea_x = np.arange(-3.5, 4.0) * self.fovea_tx_spacing_m
        fovea   = np.column_stack([fovea_x, np.zeros(8), np.zeros(8)])

        # 4 surround Tx, wider spacing
        surr_x  = np.array([-1.5, -0.5, 0.5, 1.5]) * self.surround_tx_spacing_m
        surround = np.column_stack([surr_x, np.zeros(4), np.full(4, 0.06)])

        return np.vstack([fovea, surround]).astype(np.float32)

    def rx_positions(self) -> np.ndarray:
        """
        Returns (n_rx, 3) array of receiver positions.
        Rx array is offset +owl_offset_z_m in z for 3D localisation
        (barn owl asymmetric ear principle).
        """
        # 8 foveal Rx, same x-spacing but shifted vertically
        fovea_x = np.arange(-3.5, 4.0) * self.fovea_rx_spacing_m
        fovea   = np.column_stack([
            fovea_x,
            np.zeros(8),
            np.full(8, self.owl_offset_z_m)
        ])

        # 4 surround Rx
        surr_x  = np.array([-1.5, -0.5, 0.5, 1.5]) * self.surround_rx_spacing_m
        surround = np.column_stack([
            surr_x,
            np.zeros(4),
            np.full(4, self.owl_offset_z_m + 0.04)
        ])

        return np.vstack([fovea, surround]).astype(np.float32)

    def n_tx(self) -> int:
        return len(self.tx_positions())

    def n_rx(self) -> int:
        return len(self.rx_positions())

    def foveal_tx_mask(self) -> np.ndarray:
        """Boolean mask: True = foveal element (first 8 Tx)."""
        n = self.n_tx()
        m = np.zeros(n, dtype=bool)
        m[:8] = True
        return m

    def foveal_rx_mask(self) -> np.ndarray:
        n = self.n_rx()
        m = np.zeros(n, dtype=bool)
        m[:8] = True
        return m

    def build_voxel_grid(
        self,
        x_range: tuple[float, float] = (-2.0, 2.0),
        y_range: tuple[float, float] = (0.5, 25.0),
        z_range: tuple[float, float] = (-1.5, 2.5),
        resolution_m: float = 0.10,
    ) -> np.ndarray:
        """
        Build 3D voxel grid for back-projection.
        Returns (nx, ny, nz, 3) f32 array of voxel centres.
        """
        xs = np.arange(x_range[0], x_range[1], resolution_m)
        ys = np.arange(y_range[0], y_range[1], resolution_m)
        zs = np.arange(z_range[0], z_range[1], resolution_m)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
        voxels = np.stack([gx, gy, gz], axis=-1).astype(np.float32)
        return voxels
