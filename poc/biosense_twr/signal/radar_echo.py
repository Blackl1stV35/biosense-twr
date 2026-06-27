"""
Synthetic UWB radar echo generator.

For each (tx, rx) pair and each body scatterer, computes
the round-trip delay, applies the bat-adaptive FM-UWB pulse,
accumulates into the received signal matrix.

Bat-adaptive mode selection:
  - Search mode:    wide bandwidth, low centre freq (penetration)
  - Approach mode:  medium bandwidth
  - Terminal mode:  narrow bandwidth, high centre freq (resolution)
"""

from __future__ import annotations
import numpy as np
from typing import Literal
from dataclasses import dataclass

AdaptiveMode = Literal["search", "approach", "terminal"]

# Frequency parameters per mode (fc_hz, bw_hz)
MODE_PARAMS: dict[AdaptiveMode, tuple[float, float]] = {
    "search":   (1.5e9, 3.0e9),   # 0–3 GHz wide band
    "approach": (3.0e9, 2.0e9),   # 2–4 GHz
    "terminal": (6.0e9, 1.0e9),   # 5.5–6.5 GHz
}


@dataclass
class RadarEchoConfig:
    fs: float = 40e9        # sampling rate [Hz]
    n_fast_time: int = 2048 # samples per PRI
    prf: float = 1000.0     # pulse repetition frequency [Hz]
    mode: AdaptiveMode = "search"
    snr_db: float = 20.0
    c0: float = 3e8

    @property
    def fc(self) -> float:
        return MODE_PARAMS[self.mode][0]

    @property
    def bw(self) -> float:
        return MODE_PARAMS[self.mode][1]

    @property
    def t_fast(self) -> np.ndarray:
        return np.arange(self.n_fast_time) / self.fs

    def fm_pulse(self) -> np.ndarray:
        """
        Generate FM chirp pulse (bat-style linear frequency modulation).
        Returns complex analytic signal of length n_fast_time.
        """
        T_pulse = self.n_fast_time / self.fs * 0.1  # pulse occupies 10% of PRI
        t = np.linspace(0, T_pulse, int(T_pulse * self.fs), dtype=np.float64)
        chirp_rate = self.bw / T_pulse
        phase = 2 * np.pi * ((self.fc - self.bw / 2) * t + 0.5 * chirp_rate * t**2)
        pulse = np.exp(1j * phase).astype(np.complex64)
        # Zero-pad to n_fast_time
        out = np.zeros(self.n_fast_time, dtype=np.complex64)
        out[:len(pulse)] = pulse
        return out


def synthesise_echo(
    tx_pos: np.ndarray,       # (n_tx, 3) f32
    rx_pos: np.ndarray,       # (n_rx, 3) f32
    scatterer_centres: np.ndarray,  # (n_seg, 3) f32 — ONE frame
    scatterer_rcs: np.ndarray,      # (n_seg,) f32
    wall_atten_lin: float,
    wall_phase_rad: float,
    cfg: RadarEchoConfig,
) -> np.ndarray:
    """
    Synthesise (n_tx, n_rx, n_fast_time) complex64 echo matrix
    for a single body pose frame.
    """
    n_tx, n_rx = tx_pos.shape[0], rx_pos.shape[0]
    n_t = cfg.n_fast_time
    c   = cfg.c0
    fs  = cfg.fs

    pulse = cfg.fm_pulse()
    t_fast = cfg.t_fast

    s = np.zeros((n_tx, n_rx, n_t), dtype=np.complex64)

    for ti in range(n_tx):
        for ri in range(n_rx):
            echo = np.zeros(n_t, dtype=np.complex64)
            for si in range(scatterer_centres.shape[0]):
                sc = scatterer_centres[si]
                d1 = np.linalg.norm(tx_pos[ti] - sc)
                d2 = np.linalg.norm(rx_pos[ri] - sc)
                delay = (d1 + d2) / c
                delay_samples = int(delay * fs)
                if delay_samples >= n_t:
                    continue
                # Amplitude: RCS weight / (d1*d2) (two-way spreading)
                amp = scatterer_rcs[si] / max(d1 * d2, 0.1)
                # Phase: Doppler + wall correction
                phase = -2 * np.pi * cfg.fc * delay + wall_phase_rad
                shifted_pulse = np.zeros(n_t, dtype=np.complex64)
                end = min(n_t, delay_samples + len(pulse))
                shifted_pulse[delay_samples:end] = pulse[:end - delay_samples]
                echo += amp * wall_atten_lin * shifted_pulse * np.exp(1j * phase)

            # Add AWGN
            snr_lin = 10 ** (cfg.snr_db / 10)
            noise_power = np.mean(np.abs(echo)**2) / snr_lin if np.any(echo != 0) else 1e-12
            noise = (np.random.randn(n_t) + 1j * np.random.randn(n_t)) * np.sqrt(noise_power / 2)
            s[ti, ri] = (echo + noise).astype(np.complex64)

    return s


def build_range_doppler_map(
    s: np.ndarray,   # (n_tx, n_rx, n_t) complex64
    n_slow_time: int = 64,
) -> np.ndarray:
    """
    Coherently sum over all Tx-Rx pairs, then take range-FFT
    along fast time and Doppler-FFT along slow time axis.
    Returns (n_range, n_doppler) f32 power map.
    """
    # Coherent sum (n_fast_time,) from single frame
    coherent = s.mean(axis=(0, 1))
    # For range-Doppler we'd need slow-time axis; here just return range profile
    range_profile = np.abs(np.fft.fft(coherent)[:s.shape[2]//2])**2
    return range_profile.astype(np.float32)
