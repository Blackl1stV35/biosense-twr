"""
BioSense TWR — Bio-inspired through-wall radar simulation.

Architecture layers:
  1. simulation   - FDTD EM propagation + wall models
  2. signal       - Radar echo synthesis (Boulic body model)
  3. inversion    - AI pose inversion (PyTorch CNN)
  4. muon_gravity - Muon tomography + quantum gravimetry fusion
  5. rust_kernels - PyO3 Rust compute kernels (loaded after maturin build)
"""

__version__ = "0.1.0"
