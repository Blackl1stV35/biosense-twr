use pyo3::prelude::*;

mod backprojection;
mod cfar;
mod wall_correction;
mod muon_density;
mod gravity_inversion;

/// BioSense TWR Rust kernels — compute-critical inner loops for PoC pipeline.
/// Exposed to Python via PyO3.
///
/// PoC pipeline kernels:
///   backproject_3d        — 3D delay-and-sum SAR back-projection (MIMO aperture, Rayon)
///   ca_cfar_2d            — Cell-Averaging CFAR detector (Rayon)
///   apply_wall_correction — Dielectric wall phase + amplitude correction (Rayon)
///   reconstruct_density_map — Muon PoCA density reconstruction (Rayon)
///   estimate_mass_anomaly   — Gravity anomaly CG inversion (Rayon)
#[pymodule]
fn rust_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(backprojection::backproject_3d, m)?)?;
    m.add_function(wrap_pyfunction!(cfar::ca_cfar_2d, m)?)?;
    m.add_function(wrap_pyfunction!(wall_correction::apply_wall_correction, m)?)?;
    m.add_function(wrap_pyfunction!(muon_density::reconstruct_density_map, m)?)?;
    m.add_function(wrap_pyfunction!(gravity_inversion::estimate_mass_anomaly, m)?)?;
    Ok(())
}
