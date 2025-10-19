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
