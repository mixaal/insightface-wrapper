use image::Rgba32FImage;
use ndarray::Array;

use crate::image::ImageTensor;

pub(crate) fn to_tensor(image: &Rgba32FImage) -> ImageTensor {
    let shape = image.dimensions();

    let input = Array::from_shape_fn(
        (1_usize, 3_usize, shape.0 as usize, shape.1 as usize),
        |(_, c, i, j)| ((image[(j as _, i as _)][c] as f32) - 0.5f32) / 0.5f32,
    );

    return input;
}

// Compute cosine similarity between two embeddings
//  similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
pub fn similarity(emb1: &ndarray::Array1<f32>, emb2: &ndarray::Array1<f32>) -> f32 {
    let dot_product = emb1.dot(emb2);
    let norm1 = emb1.dot(emb1).sqrt();
    let norm2 = emb2.dot(emb2).sqrt();
    dot_product / (norm1 * norm2)
}
