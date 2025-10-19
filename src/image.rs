use std::time::Instant;

use image::{Rgba32FImage, imageops::FilterType};

use crate::{error::InsightFaceError, utils};

pub struct ImageBatchLoader {
    pub(crate) batch: Vec<ImageWrapper>,
}

pub type ImageTensor = ndarray::Array<f32, ndarray::Dim<[usize; 4]>>;

impl ImageBatchLoader {
    // read images from file paths and resize to specified dimensions
    pub fn read_from_path(
        image_path: &Vec<&str>,
        dimensions: (u32, u32),
    ) -> Result<Self, InsightFaceError> {
        let mut batch = Vec::new();
        for path in image_path {
            batch.push(ImageWrapper::from_path(path, dimensions)?);
        }
        Ok(Self { batch })
    }

    // read images from memory buffers and resize to specified dimensions
    pub fn read_from_memory(
        images_in_memory: Vec<&[u8]>,
        dimensions: (u32, u32),
    ) -> Result<Self, InsightFaceError> {
        let mut batch = Vec::new();
        for img in images_in_memory {
            batch.push(ImageWrapper::from_memory(img, dimensions)?);
        }
        Ok(Self { batch })
    }

    // pub fn read_rgb8(&self, image_path: Vec<&str>) -> Result<Vec<RgbImage>, InsightFaceError> {
    //     let mut images = Vec::new();
    //     for path in image_path {
    //         let image = image::open(path)
    //             .map_err(|e| InsightFaceError::new(e))?
    //             .to_rgb8();
    //         images.push(image::imageops::resize(
    //             &image,
    //             self.dimensions.0,
    //             self.dimensions.1,
    //             FilterType::CatmullRom,
    //         ));
    //     }
    //     Ok(images)
    // }
}

pub struct ImageWrapper {
    pub(crate) image: Rgba32FImage,
    orig_dimensions: (u32, u32),
    scaled_dimensions: (u32, u32),
}

impl ImageWrapper {
    // return scale factor between original and scaled dimensions
    pub(crate) fn scale_factor(&self) -> (f32, f32) {
        (
            self.orig_dimensions.0 as f32 / self.scaled_dimensions.0 as f32,
            self.orig_dimensions.1 as f32 / self.scaled_dimensions.1 as f32,
        )
    }

    pub fn from_path(path: &str, dimensions: (u32, u32)) -> Result<Self, InsightFaceError> {
        let image = image::open(path)
            .map_err(|e| InsightFaceError::new(e))?
            .to_rgba32f();
        let orig_dim = image.dimensions();
        Ok(ImageWrapper {
            image: image::imageops::resize(
                &image,
                dimensions.0,
                dimensions.1,
                FilterType::CatmullRom,
            ),
            orig_dimensions: orig_dim,
            scaled_dimensions: dimensions,
        })
    }

    pub fn from_memory(buffer: &[u8], dimensions: (u32, u32)) -> Result<Self, InsightFaceError> {
        let start = Instant::now();
        let image = image::load_from_memory(buffer).map_err(|e| InsightFaceError::new(e))?;
        tracing::info!("Loaded image from memory in {:?}", start.elapsed());
        let start = Instant::now();
        let image = image.to_rgba32f();
        let orig_dim = image.dimensions();
        tracing::info!("Converted image into rgba32f {:?}", start.elapsed());
        let start = Instant::now();
        let resized =
            image::imageops::resize(&image, dimensions.0, dimensions.1, FilterType::CatmullRom);
        tracing::info!("Resized image in {:?}", start.elapsed());
        Ok(ImageWrapper {
            image: resized,
            orig_dimensions: orig_dim,
            scaled_dimensions: dimensions,
        })
    }

    pub(crate) fn to_tensor(&self) -> ImageTensor {
        utils::to_tensor(&self.image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    static IMAGE_PATHS: [&str; 6] = [
        "images/20250711_112226.jpg",
        "images/20250711_112229.jpg",
        "images/20250716_154632.jpg",
        "images/20250716_154633.jpg",
        "images/20251018_144943.jpg",
        "images/20251018_144948.jpg",
    ];

    #[test]
    fn test_image_loader() {
        tracing_subscriber::fmt::init();
        let path_loader =
            ImageBatchLoader::read_from_path(&IMAGE_PATHS.to_vec(), (640, 640)).unwrap();
        let from_path_images = path_loader
            .batch
            .iter()
            .map(|img| img.to_tensor())
            .collect::<Vec<ImageTensor>>();
        assert_eq!(from_path_images.len(), IMAGE_PATHS.len());
        tracing::info!("Loaded {} images from path", from_path_images.len());
        let memory_images = IMAGE_PATHS
            .iter()
            .map(|path| std::fs::read(path).unwrap())
            .collect::<Vec<Vec<u8>>>();
        let mem_loader = ImageBatchLoader::read_from_memory(
            memory_images.iter().map(|v| v.as_slice()).collect(),
            (640, 640),
        )
        .unwrap();
        let from_memory_images = mem_loader
            .batch
            .iter()
            .map(|img| img.to_tensor())
            .collect::<Vec<ImageTensor>>();
        tracing::info!("Loaded {} images from memory", from_memory_images.len());
        assert_eq!(from_memory_images.len(), IMAGE_PATHS.len());

        assert_eq!(from_path_images.len(), from_memory_images.len());
        // assert_eq!(from_path_images, from_memory_images);
    }
}
