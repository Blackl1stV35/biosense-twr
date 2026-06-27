# BioSense TWR — Claude Code Agent Prompt

## Context

You are working inside the `biosense-twr` repository. The proof-of-concept simulation lives under `poc/`. Your role is to implement, debug, extend, and optimise the through-wall radar simulation pipeline targeting an A100 GPU on Colab Pro.

## Repository structure

```
biosense-twr/
├── poc/
│   ├── pyproject.toml              ← maturin build (PyO3 + Python package)
│   ├── BENCHMARK.md                ← technology comparison table (read-only reference)
│   ├── biosense_twr/               ← Python package
│   │   ├── checkpoint_manager.py   ← chunked Drive I/O, zstd compression
│   │   ├── simulation/
│   │   │   ├── wall_model.py       ← dielectric wall configs + attenuation
│   │   │   └── antenna_array.py    ← foveal MIMO geometry + 3D voxel grid
│   │   ├── signal/
│   │   │   ├── boulic_model.py     ← Boulic human body kinematics → scatterers
│   │   │   └── radar_echo.py       ← bat-adaptive FM-UWB echo synthesis
│   │   └── inversion/
│   │       └── pose_net.py         ← 3D CNN pose inversion + OKS metric
│   ├── rust_kernels/               ← PyO3 Rust crate
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs              ← PyO3 exports (PoC pipeline only)
│   │       ├── backprojection.rs   ← 3D SAR delay-and-sum (Rayon parallel)
│   │       ├── cfar.rs             ← 2D CA-CFAR detector (Rayon parallel)
│   │       ├── wall_correction.rs  ← complex wall phase/amplitude correction
│   │       ├── muon_density.rs     ← benchmark ref: PoCA muon reconstruction
│   │       └── gravity_inversion.rs← benchmark ref: CG gravity inversion
│   └── notebooks/
│       └── A100_sim_run.ipynb      ← Colab A100 orchestration notebook
```

## Bio-inspired architecture principles

Every module maps to a biological model. Keep this mapping intact when extending:

| Module | Biological principle | Key mechanism |
|---|---|---|
| `antenna_array.py` | Star-nosed mole (foveal MIMO) + Barn owl (asymmetric Rx offset) | High-density centre + 15 cm vertical Rx shift |
| `radar_echo.py` | Bat echolocation (adaptive FM-UWB) | search / approach / terminal mode switching |
| `wall_model.py` | Weakly electric fish (field-distortion inversion) | Dielectric eps_r encodes medium perturbation |
| `pose_net.py` | Mantis shrimp (parallel multi-channel) | Multi-head CNN fusing 3D spatial features |

## Compute constraints — A100 Colab Pro

- GPU: A100 40 GB VRAM. Use `torch.compile()` and `torch.cuda.amp.autocast()`.
- RAM: ~83 GB. Keep in-memory simulation buffer ≤ 500 scenarios (~200 MB).
- Drive: 5 GB budget. Flush every `CHUNK_SIZE=200` scenarios with zstd compression.
- Session: ~12 hr. Pipeline must be resumable — `CheckpointManager` handles this.
- CPU: 2–4 cores for Rust kernels. Rayon auto-detects thread count.

## Rust / PyO3 rules

- Rebuild with `maturin develop --release` from `poc/` after any `.rs` change.
- Only three kernels are exported to Python (see `lib.rs`):
  - `backproject_3d` — call after wall correction
  - `ca_cfar_2d` — call on range-Doppler map to get detection mask
  - `apply_wall_correction` — call on raw echo before back-projection
- `muon_density.rs` and `gravity_inversion.rs` are **benchmark reference only** — do not add them to the PyO3 exports unless explicitly asked.
- All Rust arrays are `f32`. Never pass `f64` from Python — cast with `.astype(np.float32)` first.

## Pipeline order (one scenario)

```
BoulicBody.generate_poses()
    → synthesise_echo()              [Python — FM-UWB echo synthesis]
    → apply_wall_correction()        [Rust  — phase + amplitude correction]
    → backproject_3d()               [Rust  — 3D SAR image]
    → BioSensePoseNet.forward()      [PyTorch — pose inversion]
    → oks_metric()                   [PyTorch — OKS evaluation]
    → CheckpointManager.add()        [Python — chunk to Drive]
```

## Primary metric

**OKS (Object Keypoint Similarity)** — defined in `pose_net.py`.

| OKS threshold | Meaning |
|---|---|
| > 0.3 | Presence detected |
| > 0.5 | Coarse pose (standing vs crouching) |
| > 0.7 | Full skeletal pose — publication grade |
| > 0.85 | Matches optical pose estimation — research frontier |

## When modifying code

1. Never change the `poc/` directory structure — the Colab notebook uses hardcoded imports.
2. When adding a new wall material, add it to `MATERIAL_PARAMS` in `wall_model.py`.
3. When changing voxel grid resolution, update `resolution_m` in both `antenna_array.py` and any call to `build_voxel_grid()` in the notebook.
4. Keep `BENCHMARK.md` read-only — it documents external technology baselines.
5. All new Python files go under the appropriate `biosense_twr/` subpackage with a corresponding `__init__.py` import.

## Benchmark reference

See `BENCHMARK.md` for a full comparison of BioSense TWR against existing
fielded and research systems, and against the two physics-limit technologies
(muon tomography, quantum gravimetry) that define the theoretical ceiling.
