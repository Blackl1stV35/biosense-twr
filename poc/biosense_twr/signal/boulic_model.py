"""
Boulic-Sinusoidal Pendulum human body motion model.

Generates time-series of 3D joint positions for a walking/
running/crouching human from the Boulic (1990) kinematic model.

Each body segment is modelled as an ellipsoidal scatterer.
The radar echo is synthesised as the sum of returns from
all segments, weighted by RCS (radar cross-section) and
attenuated by range.

Reference:
  Boulic, R. et al. "A global human walking model..."
  The Visual Computer, 6(6):265-279, 1990.
  Yang, X. et al. IEEE TMT&T 2024 (Boulic-Sinusoidal pendulum extension).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Literal

ActivityType = Literal["walk", "run", "crouch", "stand", "sitdown", "wave_arm"]

# Joint indices
JOINTS = {
    "pelvis": 0, "spine": 1, "chest": 2, "neck": 3, "head": 4,
    "l_shoulder": 5, "l_elbow": 6, "l_wrist": 7,
    "r_shoulder": 8, "r_elbow": 9, "r_wrist": 10,
    "l_hip": 11, "l_knee": 12, "l_ankle": 13,
    "r_hip": 14, "r_knee": 15, "r_ankle": 16,
}
N_JOINTS = len(JOINTS)

# Ellipsoidal scatterer parameters (a, b, c) semi-axes [m], RCS weight
SEGMENT_RCS: dict[str, tuple[float, float, float, float]] = {
    "torso":      (0.20, 0.12, 0.30, 1.0),
    "head":       (0.10, 0.10, 0.12, 0.6),
    "upper_arm":  (0.04, 0.04, 0.14, 0.3),
    "forearm":    (0.03, 0.03, 0.12, 0.25),
    "thigh":      (0.06, 0.06, 0.20, 0.5),
    "shin":       (0.04, 0.04, 0.18, 0.4),
}


@dataclass
class BoulicBody:
    height_m: float = 1.75
    mass_kg: float = 75.0
    activity: ActivityType = "walk"
    walking_speed_mps: float = 1.4
    fs_body: float = 100.0        # body pose sample rate [Hz]
    body_position: np.ndarray = field(
        default_factory=lambda: np.array([5.0, 0.0, 0.0], dtype=np.float32)
    )  # global position [range, cross-x, height]

    def generate_poses(self, duration_s: float) -> np.ndarray:
        """
        Generate joint positions over time.
        Returns (n_frames, N_JOINTS, 3) float32 array.
        n_frames = ceil(duration_s * fs_body)
        """
        n = int(np.ceil(duration_s * self.fs_body))
        t = np.linspace(0, duration_s, n, dtype=np.float32)
        poses = np.zeros((n, N_JOINTS, 3), dtype=np.float32)

        # Stride frequency (Boulic model: f_stride = 0.35 * v + 0.73 for v in m/s)
        v = self.walking_speed_mps
        f_stride = 0.35 * v + 0.73 if self.activity in ("walk", "run") else 0.5

        omega = 2 * np.pi * f_stride

        # Pelvis: translates along range, sinusoidal vertical bob
        poses[:, JOINTS["pelvis"], 0] = self.body_position[0] + v * t
        poses[:, JOINTS["pelvis"], 1] = self.body_position[1]
        poses[:, JOINTS["pelvis"], 2] = 0.95 * self.height_m + 0.03 * np.sin(2 * omega * t)

        # Spine / chest / neck / head — stack above pelvis
        for seg, frac in [("spine", 1.05), ("chest", 1.15), ("neck", 1.25), ("head", 1.35)]:
            poses[:, JOINTS[seg], :] = poses[:, JOINTS["pelvis"], :].copy()
            poses[:, JOINTS[seg], 2] = 0.95 * self.height_m * frac / 1.35

        # Shoulders
        shoulder_width = 0.20 * self.height_m
        for side, sign, ls, le, lw, lh, lk, la in [
            ("l", -1, "l_shoulder", "l_elbow", "l_wrist", "l_hip", "l_knee", "l_ankle"),
            ("r", +1, "r_shoulder", "r_elbow", "r_wrist", "r_hip", "r_knee", "r_ankle"),
        ]:
            # Shoulder
            poses[:, JOINTS[ls], :] = poses[:, JOINTS["chest"], :].copy()
            poses[:, JOINTS[ls], 1] += sign * shoulder_width

            # Arm swing (antiphase with ipsilateral leg)
            arm_swing = 0.3 * np.sin(omega * t + sign * np.pi)
            poses[:, JOINTS[le], :] = poses[:, JOINTS[ls], :].copy()
            poses[:, JOINTS[le], 0] += 0.15 * arm_swing
            poses[:, JOINTS[le], 2] -= 0.15 * self.height_m

            poses[:, JOINTS[lw], :] = poses[:, JOINTS[le], :].copy()
            poses[:, JOINTS[lw], 0] += 0.12 * arm_swing
            poses[:, JOINTS[lw], 2] -= 0.12 * self.height_m

            # Hip
            hip_width = 0.12 * self.height_m
            poses[:, JOINTS[lh], :] = poses[:, JOINTS["pelvis"], :].copy()
            poses[:, JOINTS[lh], 1] += sign * hip_width

            # Leg swing
            leg_swing = 0.4 * np.sin(omega * t - sign * np.pi / 2)
            thigh_len = 0.24 * self.height_m
            shin_len  = 0.23 * self.height_m

            poses[:, JOINTS[lk], :] = poses[:, JOINTS[lh], :].copy()
            poses[:, JOINTS[lk], 0] += thigh_len * np.sin(leg_swing)
            poses[:, JOINTS[lk], 2] -= thigh_len * np.cos(leg_swing)

            poses[:, JOINTS[la], :] = poses[:, JOINTS[lk], :].copy()
            poses[:, JOINTS[la], 0] += shin_len * np.sin(leg_swing * 0.6)
            poses[:, JOINTS[la], 2] -= shin_len

        return poses

    def poses_to_scatterers(
        self, poses: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert joint positions to ellipsoidal scatterer centres and RCS weights.
        Returns:
          centres: (n_frames, n_segments, 3) f32
          rcs:     (n_segments,) f32
        """
        j = JOINTS
        # Define segments as midpoints between joints
        segment_pairs = [
            ("torso",     j["pelvis"],    j["chest"]),
            ("head",      j["neck"],      j["head"]),
            ("upper_arm", j["l_shoulder"],j["l_elbow"]),
            ("forearm",   j["l_elbow"],   j["l_wrist"]),
            ("upper_arm", j["r_shoulder"],j["r_elbow"]),
            ("forearm",   j["r_elbow"],   j["r_wrist"]),
            ("thigh",     j["l_hip"],     j["l_knee"]),
            ("shin",      j["l_knee"],    j["l_ankle"]),
            ("thigh",     j["r_hip"],     j["r_knee"]),
            ("shin",      j["r_knee"],    j["r_ankle"]),
        ]
        n_frames = poses.shape[0]
        n_seg    = len(segment_pairs)
        centres  = np.zeros((n_frames, n_seg, 3), dtype=np.float32)
        rcs_vals = np.zeros(n_seg, dtype=np.float32)

        for si, (seg_name, j1, j2) in enumerate(segment_pairs):
            centres[:, si, :] = (poses[:, j1, :] + poses[:, j2, :]) * 0.5
            rcs_vals[si] = SEGMENT_RCS[seg_name][3]

        return centres, rcs_vals
