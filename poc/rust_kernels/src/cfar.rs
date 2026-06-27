/// 2D Cell-Averaging CFAR (CA-CFAR) detector.
/// Parallelised over range bins via Rayon.
///
/// Args (Python):
///   data:       (n_range, n_doppler) f32 power map
///   guard_r:    usize  guard cells in range
///   guard_d:    usize  guard cells in Doppler
///   train_r:    usize  training cells in range
///   train_d:    usize  training cells in Doppler
///   pfa:        f32    probability of false alarm (e.g. 1e-4)
///
/// Returns:
///   mask: (n_range, n_doppler) bool detection mask

use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use rayon::prelude::*;
use ndarray::Array2;

#[pyfunction]
pub fn ca_cfar_2d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    guard_r: usize,
    guard_d: usize,
    train_r: usize,
    train_d: usize,
    pfa: f32,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let d = data.as_array();
    let (nr, nd) = (d.shape()[0], d.shape()[1]);

    // Alpha threshold factor from Pfa for CA-CFAR:
    // alpha = N * (Pfa^(-1/N) - 1)  where N = number of training cells
    let n_train = (2 * train_r * (2 * (guard_d + train_d) + 1)
                 + 2 * train_d * (2 * guard_r + 1)) as f32;
    let alpha = n_train * (pfa.powf(-1.0 / n_train) - 1.0);

    // Precompute owned row of data for Rayon
    let d_owned: Vec<Vec<f32>> = (0..nr).map(|r| {
        (0..nd).map(|c| d[[r, c]]).collect()
    }).collect();

    let mask_flat: Vec<bool> = (0..nr).into_par_iter().flat_map(|r| {
        let row: Vec<bool> = (0..nd).map(|c| {
            let mut noise_sum = 0.0f32;
            let mut cell_count = 0u32;

            let r_start = r.saturating_sub(train_r + guard_r);
            let r_end   = (r + train_r + guard_r + 1).min(nr);
            let c_start = c.saturating_sub(train_d + guard_d);
            let c_end   = (c + train_d + guard_d + 1).min(nd);

            for ri in r_start..r_end {
                for ci in c_start..c_end {
                    // Skip guard and CUT
                    let in_guard_r = ri >= r.saturating_sub(guard_r) && ri <= r + guard_r;
                    let in_guard_c = ci >= c.saturating_sub(guard_d) && ci <= c + guard_d;
                    if in_guard_r && in_guard_c { continue; }
                    noise_sum += d_owned[ri][ci];
                    cell_count += 1;
                }
            }

            if cell_count == 0 { return false; }
            let threshold = alpha * noise_sum / cell_count as f32;
            d_owned[r][c] > threshold
        }).collect();
        row
    }).collect();

    let mut mask = Array2::<bool>::from_elem((nr, nd), false);
    for r in 0..nr {
        for c in 0..nd {
            mask[[r, c]] = mask_flat[r * nd + c];
        }
    }

    Ok(mask.into_pyarray_bound(py))
}
