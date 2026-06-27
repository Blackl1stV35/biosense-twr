"""
Synthetic gravimeter reading generator for body-mass inversion.

Models a MEMS quantum gravimeter array placed on the floor/wall surface.
Each sensor measures the vertical component of gravitational anomaly
caused by the body's mass distribution above background.

Forward model: point-mass Bouguer approximation
  delta_g = G * M_segment / r^2 * cos(theta)

where theta is the angle from vertical between sensor and mass element.

Outputs are (sensor_pos, delta_g) arrays ready for
rust_kernels.estimate_mass_anomaly().
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

G_CONST = 6.674e-11   # gravitational constant [m^3 kg^-1 s^-2]

# Body segment mass fractions (Winter 1990 biomechanics reference)
_SEGMENT_MASS_FRACTIONS: dict[str, float] = {
    "head":        0.0810,
    "trunk":       0.4970,
    "upper_arm":   0.0280,   # per arm × 2
    "forearm":     0.0160,   # per arm × 2
    "hand":        0.0060,   # per hand × 2
    "thigh":       0.1000,   # per leg × 2
    "shank":       0.0465,   # per leg × 2
    "foot":        0.0145,   # per foot × 2
}

# Map from Boulic joint pairs to segment names + bilateral multiplier
_JOINT_SEGMENT_MAP: list[tuple[str, float]] = [
    ("head",      1.0),
    ("trunk",     1.0),
    ("upper_arm", 2.0),
    ("forearm",   2.0),
    ("hand",      2.0),
    ("thigh",     2.0),
    ("shank",     2.0),
    ("foot",      2.0),
]


@dataclass
class GravimeterConfig:
    n_sensors: int = 16              # number of sensor positions
    sensor_grid_m: float = 2.0      # half-width of sensor grid [m]
    sensor_height_m: float = 0.0    # sensor elevation (floor level) [m]
    noise_eotvos: float = 10.0      # sensor noise [Eötvös = 1e-9 m/s^2]
    sensor_seed: int = 0            # fixed seed for sensor placement reproducibility


def body_segment_masses(total_mass_kg: float) -> np.ndarray:
    """
    Return per-joint mass estimates for the 17 Boulic joints.

    Uses Winter (1990) body segment mass fractions distributed
    across the N_JOINTS=17 joint positions as proxy mass centres.

    Returns: (17,) float32 array of masses [kg]
    """
    # Map fraction × total to 17 joints (pelvis→spine→...→ankles)
    # Distribute trunk mass across spine joints, limb masses to distal joints
    fracs = np.zeros(17, dtype=np.float32)

    # Joint index mapping from boulic_model.py
    # pelvis=0, spine=1, chest=2, neck=3, head=4
    # l_shoulder=5, l_elbow=6, l_wrist=7
    # r_shoulder=8, r_elbow=9, r_wrist=10
    # l_hip=11, l_knee=12, l_ankle=13
    # r_hip=14, r_knee=15, r_ankle=16

    trunk_frac = _SEGMENT_MASS_FRACTIONS["trunk"]
    fracs[0] = trunk_frac * 0.30   # pelvis  — 30% of trunk
    fracs[1] = trunk_frac * 0.35   # spine   — 35% of trunk
    fracs[2] = trunk_frac * 0.25   # chest   — 25% of trunk
    fracs[3] = trunk_frac * 0.05   # neck    — 5% of trunk
    fracs[4] = _SEGMENT_MASS_FRACTIONS["head"]

    # Arms (bilateral already in fractions × 2, split per side)
    ua  = _SEGMENT_MASS_FRACTIONS["upper_arm"] * 2
    fa  = _SEGMENT_MASS_FRACTIONS["forearm"]   * 2
    han = _SEGMENT_MASS_FRACTIONS["hand"]      * 2
    fracs[5]  = ua  / 2   # l_shoulder (upper arm mass)
    fracs[6]  = fa  / 2   # l_elbow    (forearm mass)
    fracs[7]  = han / 2   # l_wrist    (hand mass)
    fracs[8]  = ua  / 2   # r_shoulder
    fracs[9]  = fa  / 2   # r_elbow
    fracs[10] = han / 2   # r_wrist

    # Legs
    th  = _SEGMENT_MASS_FRACTIONS["thigh"] * 2
    sh  = _SEGMENT_MASS_FRACTIONS["shank"] * 2
    ft  = _SEGMENT_MASS_FRACTIONS["foot"]  * 2
    fracs[11] = th  / 2   # l_hip
    fracs[12] = sh  / 2   # l_knee
    fracs[13] = ft  / 2   # l_ankle
    fracs[14] = th  / 2   # r_hip
    fracs[15] = sh  / 2   # r_knee
    fracs[16] = ft  / 2   # r_ankle

    return (fracs * total_mass_kg).astype(np.float32)


def simulate_gravimeter_readings(
    cfg: GravimeterConfig,
    pose_gt: np.ndarray,      # (N_JOINTS, 3) f32 — joint positions [m]
    seg_masses: np.ndarray,   # (N_JOINTS,) f32 — mass per joint [kg]
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate gravity anomaly readings at sensor positions.

    Returns:
      sensor_pos: (n_sensors, 3) f32 — sensor positions [m]
      delta_g:    (n_sensors,) f32   — vertical gravity anomaly [m/s^2]
    """
    # Place sensors on a grid at floor level, centred around body x-y footprint
    body_x = float(pose_gt[:, 0].mean())
    body_y = float(pose_gt[:, 1].mean())

    seed_rng = np.random.default_rng(cfg.sensor_seed)
    sx = seed_rng.uniform(body_x - cfg.sensor_grid_m, body_x + cfg.sensor_grid_m, cfg.n_sensors)
    sy = seed_rng.uniform(body_y - cfg.sensor_grid_m, body_y + cfg.sensor_grid_m, cfg.n_sensors)
    sz = np.full(cfg.n_sensors, cfg.sensor_height_m)

    sensor_pos = np.stack([sx, sy, sz], axis=1).astype(np.float32)   # (n_s, 3)

    # Forward model: sum gravity contribution from each joint mass
    delta_g = np.zeros(cfg.n_sensors, dtype=np.float64)

    for j in range(pose_gt.shape[0]):
        mass = float(seg_masses[j])
        if mass < 1e-9:
            continue
        for s in range(cfg.n_sensors):
            dx = sensor_pos[s, 0] - float(pose_gt[j, 0])
            dy = sensor_pos[s, 1] - float(pose_gt[j, 1])
            dz = sensor_pos[s, 2] - float(pose_gt[j, 2])
            r2 = dx*dx + dy*dy + dz*dz
            if r2 < 1e-6:
                continue
            r = np.sqrt(r2)
            # Vertical component (z-direction)
            delta_g[s] += G_CONST * mass * dz / (r * r2)

    # Add instrument noise (Eötvös → m/s^2: 1 E = 1e-9 m/s^2)
    noise_std = cfg.noise_eotvos * 1e-9
    delta_g += rng.normal(0, noise_std, cfg.n_sensors)

    return sensor_pos, delta_g.astype(np.float32)
