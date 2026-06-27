/// Gravitational anomaly body-mass estimation.
///
/// Given a set of gravity sensor readings and their positions,
/// estimates the mass distribution of an anomalous body
/// using a forward model + least-squares inversion.
///
/// Forward model: point-mass Bouguer approximation
///   delta_g(r) = G * M / r^2 * cos(theta)
/// where theta is the angle from vertical.
///
/// Inversion: iterative weighted least-squares over a voxel grid.
/// Each voxel has an unknown mass; we minimise ||G*m - delta_g||^2
/// with Tikhonov regularisation lambda*||m||^2 for stability.
///
/// Args (Python):
///   sensor_pos:   (n_sensors, 3) f32 - sensor positions [m]
///   delta_g_obs:  (n_sensors,) f32   - observed gravity anomaly [m/s^2]
///   vox_centers:  (n_vox, 3) f32     - voxel centre positions [m]
///   lambda_reg:   f32                - Tikhonov regularisation weight
///   max_iter:     usize              - conjugate gradient iterations
///
/// Returns:
///   mass_map: (n_vox,) f32 - estimated mass per voxel [kg]

use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2, PyReadonlyArray1};
use rayon::prelude::*;
use std::f32::consts::PI;

const G: f32 = 6.674e-11;

#[pyfunction]
pub fn estimate_mass_anomaly<'py>(
    py: Python<'py>,
    sensor_pos:  PyReadonlyArray2<'py, f32>,
    delta_g_obs: PyReadonlyArray1<'py, f32>,
    vox_centers: PyReadonlyArray2<'py, f32>,
    lambda_reg:  f32,
    max_iter:    usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let sp  = sensor_pos.as_array();
    let dg  = delta_g_obs.as_array();
    let vc  = vox_centers.as_array();

    let n_s = sp.shape()[0];
    let n_v = vc.shape()[0];

    // Build sensitivity matrix G_mat: (n_s, n_v)
    // G_mat[i,j] = vertical component of gravity from unit mass at vox j to sensor i
    let g_flat: Vec<f32> = (0..n_s).into_par_iter().flat_map(|i| {
        (0..n_v).map(move |j| {
            let dx = sp[[i,0]] - vc[[j,0]];
            let dy = sp[[i,1]] - vc[[j,1]];
            let dz = sp[[i,2]] - vc[[j,2]];
            let r2 = dx*dx + dy*dy + dz*dz;
            if r2 < 1e-6 { return 0.0f32; }
            let r  = r2.sqrt();
            // Vertical (z) component contribution
            G * dz / (r * r2)  // = G * cos(theta) / r^2
        }).collect::<Vec<_>>()
    }).collect();

    // Conjugate gradient solver for:
    // (G^T G + lambda I) m = G^T delta_g
    let dg_vec: Vec<f32> = dg.iter().cloned().collect();

    // G^T * delta_g  ->  (n_v,)
    let mut rhs: Vec<f32> = (0..n_v).into_par_iter().map(|j| {
        (0..n_s).map(|i| g_flat[i*n_v + j] * dg_vec[i]).sum::<f32>()
    }).collect();

    // CG solve: A = G^T G + lambda I
    let mut m = vec![0.0f32; n_v];
    let mut r = rhs.clone();
    let mut p = r.clone();
    let mut r_dot = dot_vec(&r, &r);

    for _ in 0..max_iter {
        // A * p = G^T(G * p) + lambda * p
        let gp: Vec<f32> = mat_vec_g(&g_flat, &p, n_s, n_v);
        let gtgp: Vec<f32> = mat_vec_gt(&g_flat, &gp, n_s, n_v);
        let ap: Vec<f32> = (0..n_v).map(|j| gtgp[j] + lambda_reg * p[j]).collect();

        let alpha = r_dot / dot_vec(&p, &ap).max(1e-30);

        for j in 0..n_v {
            m[j] += alpha * p[j];
            r[j] -= alpha * ap[j];
        }

        let r_dot_new = dot_vec(&r, &r);
        if r_dot_new < 1e-20 { break; }

        let beta = r_dot_new / r_dot;
        for j in 0..n_v { p[j] = r[j] + beta * p[j]; }
        r_dot = r_dot_new;
    }

    let result = numpy::ndarray::Array1::from_vec(m);
    Ok(result.into_pyarray_bound(py))
}

fn mat_vec_g(g: &[f32], v: &[f32], n_s: usize, n_v: usize) -> Vec<f32> {
    (0..n_s).into_par_iter().map(|i| {
        (0..n_v).map(|j| g[i*n_v+j] * v[j]).sum::<f32>()
    }).collect()
}

fn mat_vec_gt(g: &[f32], v: &[f32], n_s: usize, n_v: usize) -> Vec<f32> {
    (0..n_v).into_par_iter().map(|j| {
        (0..n_s).map(|i| g[i*n_v+j] * v[i]).sum::<f32>()
    }).collect()
}

fn dot_vec(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x,y)| x*y).sum()
}
