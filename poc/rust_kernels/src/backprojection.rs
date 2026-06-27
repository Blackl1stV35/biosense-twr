/// Delay-and-sum 3D back-projection for MIMO UWB through-wall radar.
/// Parallelised over voxels via Rayon.
///
/// Args (Python):
///   s_matrix: (n_tx, n_rx, n_fast_time) complex64 received signal matrix
///   tx_pos:   (n_tx, 3) f32 transmitter positions [m]
///   rx_pos:   (n_rx, 3) f32 receiver positions [m]
///   voxels:   (nx, ny, nz, 3) f32 3D voxel grid [m]
///   fs:       f32 sampling frequency [Hz]
///   c:        f32 propagation velocity [m/s] (adjusted for wall)
///
/// Returns:
///   image: (nx, ny, nz) f32 back-projected image intensity

use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2, PyReadonlyArray4, PyReadonlyArray3};
use num_complex::Complex32;
use rayon::prelude::*;
use ndarray::{Array3, Axis};

#[pyfunction]
pub fn backproject_3d<'py>(
    py: Python<'py>,
    s_real: PyReadonlyArray3<'py, f32>,
    s_imag: PyReadonlyArray3<'py, f32>,
    tx_pos: PyReadonlyArray2<'py, f32>,
    rx_pos: PyReadonlyArray2<'py, f32>,
    voxels: PyReadonlyArray4<'py, f32>,
    fs: f32,
    c: f32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let s_r = s_real.as_array();
    let s_i = s_imag.as_array();
    let tx = tx_pos.as_array();
    let rx = rx_pos.as_array();
    let vox = voxels.as_array();

    let n_tx = tx.shape()[0];
    let n_rx = rx.shape()[0];
    let n_t  = s_r.shape()[2];
    let nx   = vox.shape()[0];
    let ny   = vox.shape()[1];
    let nz   = vox.shape()[2];

    // Build complex signal buffer (owned, thread-safe)
    let mut s_complex: Vec<Complex32> = Vec::with_capacity(n_tx * n_rx * n_t);
    for ti in 0..n_tx {
        for ri in 0..n_rx {
            for t in 0..n_t {
                s_complex.push(Complex32::new(
                    s_r[[ti, ri, t]],
                    s_i[[ti, ri, t]],
                ));
            }
        }
    }

    // Copy voxel grid and antenna positions to owned Vec for Rayon
    let vox_flat: Vec<[f32; 3]> = (0..nx).flat_map(|ix| {
        (0..ny).flat_map(move |iy| {
            (0..nz).map(move |iz| {
                [vox[[ix, iy, iz, 0]], vox[[ix, iy, iz, 1]], vox[[ix, iy, iz, 2]]]
            })
        })
    }).collect();

    let tx_flat: Vec<[f32; 3]> = (0..n_tx).map(|i| [tx[[i,0]], tx[[i,1]], tx[[i,2]]]).collect();
    let rx_flat: Vec<[f32; 3]> = (0..n_rx).map(|i| [rx[[i,0]], rx[[i,1]], rx[[i,2]]]).collect();

    // Parallel back-projection over voxels
    let image_flat: Vec<f32> = vox_flat.par_iter().map(|vp| {
        let mut acc = Complex32::new(0.0, 0.0);
        for ti in 0..n_tx {
            for ri in 0..n_rx {
                let d_tx = dist3(vp, &tx_flat[ti]);
                let d_rx = dist3(vp, &rx_flat[ri]);
                let delay = (d_tx + d_rx) / c;
                let sample_idx = (delay * fs) as isize;
                if sample_idx >= 0 && (sample_idx as usize) < n_t {
                    let idx = ti * n_rx * n_t + ri * n_t + sample_idx as usize;
                    acc += s_complex[idx];
                }
            }
        }
        acc.norm()
    }).collect();

    // Reshape flat Vec -> Array3
    let mut image = Array3::<f32>::zeros((nx, ny, nz));
    let mut flat_idx = 0;
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                image[[ix, iy, iz]] = image_flat[flat_idx];
                flat_idx += 1;
            }
        }
    }

    Ok(image.into_pyarray_bound(py))
}

#[inline(always)]
fn dist3(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx*dx + dy*dy + dz*dz).sqrt()
}
