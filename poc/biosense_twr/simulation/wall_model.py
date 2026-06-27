"""
Wall electromagnetic propagation model.

Encapsulates wall material properties and applies
2D FDTD-style propagation loss for each frequency band.
Used to generate realistic through-wall channel responses
for the synthetic radar echo simulation.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Literal

WallMaterial = Literal["drywall", "brick", "concrete", "reinforced_concrete", "wood"]

# Material electromagnetic properties at S-band (2-4 GHz)
# (eps_r, tan_delta, attn_db_per_cm)
MATERIAL_PARAMS: dict[WallMaterial, tuple[float, float, float]] = {
    "drywall":              (2.5,  0.03,  0.4),
    "wood":                 (2.0,  0.04,  0.6),
    "brick":                (4.5,  0.05,  1.2),
    "concrete":             (6.5,  0.10,  2.8),
    "reinforced_concrete":  (8.0,  0.15,  4.5),
}


@dataclass
class WallConfig:
    material: WallMaterial = "concrete"
    thickness_m: float = 0.20   # metres
    num_walls: int = 1

    @property
    def eps_r(self) -> float:
        return MATERIAL_PARAMS[self.material][0]

    @property
    def tan_delta(self) -> float:
        return MATERIAL_PARAMS[self.material][1]

    @property
    def attn_db_per_cm(self) -> float:
        return MATERIAL_PARAMS[self.material][2]

    def total_attenuation_db(self, freq_hz: float) -> float:
        """Total two-way amplitude attenuation in dB for all walls."""
        attn_per_wall = self.attn_db_per_cm * self.thickness_m * 100.0
        return 2.0 * attn_per_wall * self.num_walls  # two-way

    def effective_velocity(self) -> float:
        """EM velocity inside wall material [m/s]."""
        c0 = 3e8
        return c0 / np.sqrt(self.eps_r)

    def phase_shift_rad(self, freq_hz: float) -> float:
        """Two-way phase shift through all walls [rad]."""
        c0 = 3e8
        k = 2 * np.pi * freq_hz / c0 * np.sqrt(self.eps_r)
        return 2.0 * k * self.thickness_m * self.num_walls

    def apply_to_signal(
        self,
        s: np.ndarray,     # (..., n_fast_time) complex
        freq_hz: float,
        fs: float,
    ) -> np.ndarray:
        """
        Apply wall attenuation + phase correction to a complex signal array.
        Pure Python fallback — Rust kernel preferred for large arrays.
        """
        atten_lin = 10.0 ** (-self.total_attenuation_db(freq_hz) / 20.0)
        phi = self.phase_shift_rad(freq_hz)
        correction = atten_lin * np.exp(-1j * phi)
        return s * correction
