/// Muon scattering tomography density reconstruction.
///
/// Uses the PoCA (Point of Closest Approach) algorithm:
/// for each muon track pair (incoming + outgoing), compute
/// the point of closest approach — this is the estimated
/// scattering vertex. Accumulate scattering angles in voxels.
/// High scattering angle → high atomic number / density (Z-material).
///
/// Args (Python):
///   in_pos:    (n_muons, 3) f32 - incoming track position [m]
///   in_dir:    (n_muons, 3) f32 - incoming track direction unit vector
///   out_pos:   (n_muons, 3) f32 - outgoing track position [m]
///   out_dir:   (n_muons, 3) f32 - outgoing track direction unit vector
///   vox_min:   (3,) f32 - voxel grid minimum corner [m]
///   vox_max:   (3,) f32 - voxel grid maximum corner [m]
///   vox_n:     (3,) usize - number of voxels per axis
///
/// Returns:
///   density_map: (nx, ny, nz) f32 - scattering angle sum per voxel
///                (proxy for material density)

use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2, PyReadonlyArray1};
use rayon::prelude::*;
use ndarray::Array3;
use std::sync::Mutex;

#[pyfunction]
pub fn reconstruct_density_map<'py>(
    py: Python<'py>,
    in_pos:  PyReadonlyArray2<'py, f32>,
    in_dir:  PyReadonlyArray2<'py, f32>,
    out_pos: PyReadonlyArray2<'py, f32>,
    out_dir: PyReadonlyArray2<'py, f32>,
    vox_min: PyReadonlyArray1<'py, f32>,
    vox_max: PyReadonlyArray1<'py, f32>,
    vox_n:   PyReadonlyArray1<'py, usize>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let ip = in_pos.as_array();
    let id = in_dir.as_array();
    let op = out_pos.as_array();
    let od = out_dir.as_array();
    let vmin = vox_min.as_array();
    let vmax = vox_max.as_array();
    let vn   = vox_n.as_array();

    let n_muons = ip.shape()[0];
    let (nx, ny, nz) = (vn[0], vn[1], vn[2]);
    let (x0, y0, z0) = (vmin[0], vmin[1], vmin[2]);
    let (x1, y1, z1) = (vmax[0], vmax[1], vmax[2]);
    let (dx, dy, dz) = (
        (x1 - x0) / nx as f32,
        (y1 - y0) / ny as f32,
        (z1 - z0) / nz as f32,
    );

    // Shared accumulator protected by Mutex
    let density = Mutex::new(Array3::<f32>::zeros((nx, ny, nz)));

    // Preload arrays into owned Vecs for Rayon
    let ip_v: Vec<[f32;3]> = (0..n_muons).map(|i| [ip[[i,0]],ip[[i,1]],ip[[i,2]]]).collect();
    let id_v: Vec<[f32;3]> = (0..n_muons).map(|i| [id[[i,0]],id[[i,1]],id[[i,2]]]).collect();
    let op_v: Vec<[f32;3]> = (0..n_muons).map(|i| [op[[i,0]],op[[i,1]],op[[i,2]]]).collect();
    let od_v: Vec<[f32;3]> = (0..n_muons).map(|i| [od[[i,0]],od[[i,1]],od[[i,2]]]).collect();

    // Per-muon thread-local results, then merge
    let contributions: Vec<([usize;3], f32)> = (0..n_muons).into_par_iter()
        .filter_map(|i| {
            let poca = point_of_closest_approach(&ip_v[i], &id_v[i], &op_v[i], &od_v[i])?;
            let scatter_angle = angle_between(&id_v[i], &od_v[i]);

            // Map PoCA to voxel index
            let vix = ((poca[0] - x0) / dx) as isize;
            let viy = ((poca[1] - y0) / dy) as isize;
            let viz = ((poca[2] - z0) / dz) as isize;

            if vix < 0 || viy < 0 || viz < 0
                || vix >= nx as isize || viy >= ny as isize || viz >= nz as isize {
                return None;
            }
            Some(([vix as usize, viy as usize, viz as usize], scatter_angle))
        }).collect();

    // Sequential merge into density map
    {
        let mut d = density.lock().unwrap();
        for ([vx, vy, vz], angle) in contributions {
            d[[vx, vy, vz]] += angle;
        }
    }

    let result = density.into_inner().unwrap();
    Ok(result.into_pyarray_bound(py))
}

fn point_of_closest_approach(
    p1: &[f32;3], d1: &[f32;3],
    p2: &[f32;3], d2: &[f32;3],
) -> Option<[f32;3]> {
    // w = p1 - p2
    let w = [p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]];
    let a = dot3(d1, d1);
    let b = dot3(d1, d2);
    let c = dot3(d2, d2);
    let d = dot3(d1, &w);
    let e = dot3(d2, &w);
    let denom = a*c - b*b;
    if denom.abs() < 1e-10 { return None; } // parallel tracks
    let t1 = (b*e - c*d) / denom;
    let t2 = (a*e - b*d) / denom;
    let q1 = [p1[0]+t1*d1[0], p1[1]+t1*d1[1], p1[2]+t1*d1[2]];
    let q2 = [p2[0]+t2*d2[0], p2[1]+t2*d2[1], p2[2]+t2*d2[2]];
    Some([(q1[0]+q2[0])*0.5, (q1[1]+q2[1])*0.5, (q1[2]+q2[2])*0.5])
}

fn angle_between(a: &[f32;3], b: &[f32;3]) -> f32 {
    let cos_a = (dot3(a,b) / (norm3(a) * norm3(b))).clamp(-1.0, 1.0);
    cos_a.acos()
}

#[inline] fn dot3(a: &[f32;3], b: &[f32;3]) -> f32 { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }
#[inline] fn norm3(a: &[f32;3]) -> f32 { dot3(a,a).sqrt() }
