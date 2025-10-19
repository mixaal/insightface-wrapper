// //use ndarray_stats::QuantileExt;

use crate::math;

// Compute 25th, 50th (median), and 75th percentiles of pairwise cosine distances
// between all embeddings in the provided slice
pub fn find_distance_statistics(embeddings: &[[f32; 512]]) -> (f32, f32, f32) {
    let mut distances = vec![];

    for i in 0..embeddings.len() {
        for j in (i + 1)..embeddings.len() {
            let dist = math::cosine_distance(&embeddings[i], &embeddings[j]);
            distances.push(dist);
        }
    }

    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = distances.len();

    (
        distances[len / 4],     // 25th percentile
        distances[len / 2],     // median
        distances[3 * len / 4], // 75th percentile
    )
}
