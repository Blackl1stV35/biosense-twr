"""
Synthetic muon track generator for PoCA density reconstruction.

Simulates cosmic-ray muon tracks through a scene containing a human body
and an intervening wall. Each muon is modelled as a straight incoming track
that scatters at the body and exits on a slightly deflected outgoing track.

The scattering angle is drawn from a Highland formula approximation:
  theta_0 = (13.6 MeV / (beta*c*p)) * z * sqrt(x/X0) * (1 + 0.038*ln(x/X0))

where for our purposes we use a simplified empirical model tuned to tissue/bone.

Outputs are (in_pos, in_dir, out_pos, out_dir) arrays ready for
rust_kernels.reconstruct_density_map().
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

# Radiation length of tissue [g/cm^2] / density → X0 ≈ 37 cm for soft tissue
_X0_TISSUE_CM = 37.0
# Approximate muon momentum at sea level [MeV/c]
_P_MUON_MEV = 3000.0
# beta ≈ 1 for GeV muons
_BETA = 1.0

# Typical muon flux at sea level: ~1 muon/(cm^2·min)
_FLUX_PER_CM2_PER_MIN = 1.0
# Detector area assumed for scan (1 m^2 active area)
_DETECTOR_AREA_CM2 = 1e4


@dataclass
class MuonSimConfig:
    scan_duration_min: float = 1.0       # scan time [minutes]
    detector_area_cm2: float = _DETECTOR_AREA_CM2
    muon_momentum_mev: float = _P_MUON_MEV
    angular_spread_rad: float = 0.5      # half-angle of incoming zenith cone [rad]
    body_scatter_scale: float = 1.0      # multiplier on scattering angle (tuning knob)
    wall_scatter_scale: float = 0.3      # wall contribution to scatter (lower than body)


def simulate_muon_tracks(
    cfg: MuonSimConfig,
    body_centre_xyz: np.ndarray,   # (3,) float32 — body centre [m]
    body_mass_kg: float,
    wall_thickness_m: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic muon tracks for one scenario.

    Returns:
      in_pos:  (n_muons, 3) f32 — incoming track positions [m]
      in_dir:  (n_muons, 3) f32 — incoming track direction unit vectors
      out_pos: (n_muons, 3) f32 — outgoing track positions [m]
      out_dir: (n_muons, 3) f32 — outgoing track direction unit vectors

    Muons travel top-down (−z), scattered by body + wall.
    """
    n_muons = max(1, int(cfg.scan_duration_min * cfg.detector_area_cm2 * _FLUX_PER_CM2_PER_MIN))

    bx, by, bz = float(body_centre_xyz[0]), float(body_centre_xyz[1]), float(body_centre_xyz[2])

    # Spread incoming positions around the body in x-y plane
    spread_m = 1.5   # ±1.5 m horizontal spread around body centre
    in_x = rng.uniform(bx - spread_m, bx + spread_m, n_muons).astype(np.float32)
    in_y = rng.uniform(by - spread_m, by + spread_m, n_muons).astype(np.float32)
    in_z = np.full(n_muons, bz + 5.0, dtype=np.float32)   # start 5 m above body centre

    in_pos = np.stack([in_x, in_y, in_z], axis=1)   # (n_muons, 3)

    # Incoming directions: mostly downward with small zenith spread
    theta_in = rng.uniform(0, cfg.angular_spread_rad, n_muons).astype(np.float32)
    phi_in   = rng.uniform(0, 2 * np.pi, n_muons).astype(np.float32)
    in_dir = np.stack([
        np.sin(theta_in) * np.cos(phi_in),
        np.sin(theta_in) * np.sin(phi_in),
        -np.cos(theta_in),
    ], axis=1).astype(np.float32)   # (n_muons, 3)

    # Highland scattering angle: body contribution
    # Approximate body thickness traversed as 30 cm soft tissue
    body_thickness_cm = 30.0 * (body_mass_kg / 75.0) ** (1/3)
    x_over_X0 = body_thickness_cm / _X0_TISSUE_CM
    theta0_body = (13.6 / (cfg.muon_momentum_mev * _BETA)) * np.sqrt(x_over_X0) * (
        1.0 + 0.038 * np.log(x_over_X0)
    )
    theta0_body *= cfg.body_scatter_scale

    # Wall contribution
    wall_thickness_cm = wall_thickness_m * 100.0
    x_wall_X0 = wall_thickness_cm / 12.0   # concrete X0 ≈ 12 cm
    theta0_wall = (13.6 / (cfg.muon_momentum_mev * _BETA)) * np.sqrt(x_wall_X0) * (
        1.0 + 0.038 * np.log(max(x_wall_X0, 1e-6))
    )
    theta0_wall *= cfg.wall_scatter_scale

    theta0_total = float(np.sqrt(theta0_body**2 + theta0_wall**2))

    # Sample scatter angles (Gaussian in each transverse component)
    d_theta_x = rng.normal(0, theta0_total, n_muons).astype(np.float32)
    d_theta_y = rng.normal(0, theta0_total, n_muons).astype(np.float32)

    # Build outgoing direction by rotating in_dir by scatter angles
    out_dir = _rotate_directions(in_dir, d_theta_x, d_theta_y)

    # Outgoing positions: propagate from scattering vertex inside body
    # PoCA vertex is near body centre; propagate ~2 m below
    out_pos = np.stack([
        in_x + in_dir[:, 0] * (in_z - bz),
        in_y + in_dir[:, 1] * (in_z - bz),
        np.full(n_muons, bz - 2.0, dtype=np.float32),
    ], axis=1).astype(np.float32)

    return in_pos, in_dir, out_pos, out_dir


def _rotate_directions(
    dirs: np.ndarray,        # (N, 3)
    d_theta_x: np.ndarray,  # (N,)
    d_theta_y: np.ndarray,  # (N,)
) -> np.ndarray:
    """Apply small-angle rotation to direction vectors."""
    # Small-angle: out ≈ in + perturbation projected onto sphere
    out = dirs.copy()
    out[:, 0] += d_theta_x
    out[:, 1] += d_theta_y
    # Re-normalise
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return (out / norms).astype(np.float32)
