/// Wall dielectric correction.
/// Applies phase shift and amplitude attenuation for each
/// (tx, rx) pair based on wall permittivity and thickness.
///
/// Two-way correction: signal passes wall twice per round-trip.
///
/// Args (Python):
///   s_real, s_imag: (n_tx, n_rx, n_t) f32 - complex signal
///   eps_r:          f32  relative permittivity of wall (concrete ~6-8)
///   thickness:      f32  wall thickness [m]
///   freq_hz:        f32  centre frequency [Hz]
///   attn_db_per_cm: f32  attenuation coefficient [dB/cm] for wall material
///
/// Returns:
///   (corrected_real, corrected_imag): (n_tx, n_rx, n_t) f32 each

use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use rayon::prelude::*;
use std::f32::consts::PI;

#[pyfunction]
pub fn apply_wall_correction<'py>(
    py: Python<'py>,
    s_real: PyReadonlyArray3<'py, f32>,
    s_imag: PyReadonlyArray3<'py, f32>,
    eps_r: f32,
    thickness: f32,
    freq_hz: f32,
    attn_db_per_cm: f32,
) -> PyResult<(Bound<'py, PyArray3<f32>>, Bound<'py, PyArray3<f32>>)> {
    let sr = s_real.as_array();
    let si = s_imag.as_array();
    let shape = sr.shape();
    let (n_tx, n_rx, n_t) = (shape[0], shape[1], shape[2]);

    // Two-way phase shift: phi = 2 * (2*pi*f/c) * sqrt(eps_r) * d * 2
    let c0: f32 = 3e8;
    let k_wall = 2.0 * PI * freq_hz / c0 * eps_r.sqrt();
    let phase_shift = 2.0 * k_wall * thickness; // two-way

    let cos_phi = phase_shift.cos();
    let sin_phi = phase_shift.sin();

    // Two-way amplitude attenuation
    let attn_per_m = attn_db_per_cm * 100.0; // convert to dB/m
    let amplitude_factor = 10.0_f32.powf(-attn_per_m * thickness * 2.0 / 20.0);

    let total = n_tx * n_rx * n_t;
    let sr_flat: Vec<f32> = sr.iter().cloned().collect();
    let si_flat: Vec<f32> = si.iter().cloned().collect();

    let (cr_flat, ci_flat): (Vec<f32>, Vec<f32>) = (0..total).into_par_iter().map(|idx| {
        let r = sr_flat[idx];
        let i = si_flat[idx];
        // Rotate by -phase_shift and scale
        let cr = amplitude_factor * (r * cos_phi + i * sin_phi);
        let ci = amplitude_factor * (-r * sin_phi + i * cos_phi);
        (cr, ci)
    }).unzip();

    // Reshape back to (n_tx, n_rx, n_t)
    use ndarray::Array3;
    let cr_arr = Array3::from_shape_vec((n_tx, n_rx, n_t), cr_flat)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e}")))?;
    let ci_arr = Array3::from_shape_vec((n_tx, n_rx, n_t), ci_flat)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e}")))?;

    Ok((cr_arr.into_pyarray_bound(py), ci_arr.into_pyarray_bound(py)))
}
