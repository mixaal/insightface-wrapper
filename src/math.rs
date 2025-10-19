use ndarray::{Array2, Axis};

// Normalize a set of embeddings to unit length (L2 normalization)
pub(crate) fn normalize_embeddings(embeddings: Vec<[f32; 512]>) -> Array2<f32> {
    let n_samples = embeddings.len();
    let flat: Vec<f32> = embeddings.iter().flat_map(|e| e.iter().copied()).collect();

    let mut dataset =
        Array2::from_shape_vec((n_samples, 512), flat).expect("Failed to create array");

    // L2 normalize each row (embedding)
    for mut row in dataset.axis_iter_mut(Axis(0)) {
        let norm = row.iter().map(|&x| x * x).sum::<f32>().sqrt();

        if norm > 0.0 {
            row.mapv_inplace(|x| x / norm);
        }
    }

    dataset
}

// Compute cosine distance between two 512-dimensional embeddings
pub fn cosine_distance(emb1: &[f32; 512], emb2: &[f32; 512]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { cosine_distance_avx2(emb1, emb2) };
        }
    }

    // Fallback to scalar version
    cosine_distance_scalar(emb1, emb2)
}

fn cosine_distance_scalar(emb1: &[f32; 512], emb2: &[f32; 512]) -> f32 {
    let mut dot_product = 0.0_f32;
    let mut mag1_sq = 0.0_f32;
    let mut mag2_sq = 0.0_f32;

    for i in 0..512 {
        let a = emb1[i];
        let b = emb2[i];
        dot_product += a * b;
        mag1_sq += a * a;
        mag2_sq += b * b;
    }

    let magnitude1 = mag1_sq.sqrt();
    let magnitude2 = mag2_sq.sqrt();

    if magnitude1 == 0.0 || magnitude2 == 0.0 {
        return 1.0;
    }

    1.0 - (dot_product / (magnitude1 * magnitude2)).clamp(-1.0, 1.0)
}

#[cfg(all(target_arch = "x86_64"))]
unsafe fn cosine_distance_avx2(emb1: &[f32; 512], emb2: &[f32; 512]) -> f32 {
    use std::arch::x86_64::*;

    // AVX2 registers hold 8 f32 values (256 bits)
    // We'll process 8 elements at a time, so 512 / 8 = 64 iterations

    let mut dot_acc = _mm256_setzero_ps();
    let mut mag1_acc = _mm256_setzero_ps();
    let mut mag2_acc = _mm256_setzero_ps();

    // Process 8 elements at a time
    for i in (0..512).step_by(8) {
        // Load 8 f32 values from each embedding
        let a = _mm256_loadu_ps(emb1.as_ptr().add(i));
        let b = _mm256_loadu_ps(emb2.as_ptr().add(i));

        // Compute dot product: a * b
        dot_acc = _mm256_fmadd_ps(a, b, dot_acc);

        // Compute magnitude squared for emb1: a * a
        mag1_acc = _mm256_fmadd_ps(a, a, mag1_acc);

        // Compute magnitude squared for emb2: b * b
        mag2_acc = _mm256_fmadd_ps(b, b, mag2_acc);
    }

    // Horizontal sum of the accumulated vectors
    let dot_product = horizontal_sum(dot_acc);
    let mag1_sq = horizontal_sum(mag1_acc);
    let mag2_sq = horizontal_sum(mag2_acc);

    let magnitude1 = mag1_sq.sqrt();
    let magnitude2 = mag2_sq.sqrt();

    if magnitude1 == 0.0 || magnitude2 == 0.0 {
        return 1.0;
    }

    let cosine_similarity = (dot_product / (magnitude1 * magnitude2)).clamp(-1.0, 1.0);

    1.0 - cosine_similarity
}

#[cfg(all(target_arch = "x86_64"))]
#[inline]
unsafe fn horizontal_sum(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;

    // Sum all 8 lanes of the __m256 vector
    // Step 1: Add high 128 bits to low 128 bits
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(hi, lo);

    // Step 2: Horizontal add within 128 bits
    let sum64 = _mm_hadd_ps(sum128, sum128);
    let sum32 = _mm_hadd_ps(sum64, sum64);

    // Extract the final sum
    _mm_cvtss_f32(sum32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_distance() {
        let emb1 = [0.5_f32; 512];
        let emb2 = [0.7_f32; 512];
        cosine_distance(&emb1, &emb2);

        let emb1 = [0.0_f32; 512];
        let emb2 = [1.0_f32; 512];
        cosine_distance(&emb1, &emb2);

        let emb1 = [0.0_f32; 512];
        let emb2 = [0.0_f32; 512];
        cosine_distance(&emb1, &emb2);

        let emb1 = [1.0_f32; 512];
        let emb2 = [1.0_f32; 512];
        cosine_distance(&emb1, &emb2);

        let emb1 = [1.0_f32; 512];
        let emb2 = [-1.0_f32; 512];
        cosine_distance(&emb1, &emb2);
    }

    fn cosine_distance(emb1: &[f32; 512], emb2: &[f32; 512]) {
        let scalar_result = cosine_distance_scalar(emb1, emb2);
        let avx2_result = unsafe { cosine_distance_avx2(emb1, emb2) };

        assert!((scalar_result - avx2_result).abs() < 1e-6);
    }
}
