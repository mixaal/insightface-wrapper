use std::{io::Cursor, time::Instant};

use image::{DynamicImage, Rgba32FImage, imageops::FilterType};

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
    pub(crate) orig_dimensions: (u32, u32),
    pub(crate) scaled_dimensions: (u32, u32),
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
        let mut image = image::open(path).map_err(|e| InsightFaceError::new(e))?;
        image = Self::apply_exif_orientation_from_path(path, image)?;
        let image = image.to_rgba32f();
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
        let mut image = image::load_from_memory(buffer).map_err(|e| InsightFaceError::new(e))?;
        image = Self::apply_exif_orientation_from_memory(buffer, image)?;
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

    fn apply_exif_orientation_from_path(
        path: &str,
        img: DynamicImage,
    ) -> Result<DynamicImage, InsightFaceError> {
        let file = std::fs::File::open(path).map_err(|e| InsightFaceError::new(e))?;
        let mut bufreader = std::io::BufReader::new(&file);

        Self::apply_exif_orientation(&mut bufreader, img)
    }

    fn apply_exif_orientation_from_memory(
        buffer: &[u8],
        img: DynamicImage,
    ) -> Result<DynamicImage, InsightFaceError> {
        let mut bufreader = std::io::BufReader::new(Cursor::new(buffer));
        Self::apply_exif_orientation(&mut bufreader, img)
    }

    fn apply_exif_orientation<R: std::io::BufRead + std::io::Seek>(
        reader: &mut R,
        img: DynamicImage,
    ) -> Result<DynamicImage, InsightFaceError> {
        let exifreader = exif::Reader::new();
        let exif = match exifreader.read_from_container(reader) {
            Ok(exif) => exif,
            Err(_) => return Ok(img), // No EXIF data, return original
        };

        let orientation = match exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY) {
            Some(orientation) => match orientation.value.get_uint(0) {
                Some(v) => v,
                None => return Ok(img),
            },
            None => return Ok(img),
        };

        Ok(match orientation {
            1 => img,
            2 => img.fliph(),
            3 => img.rotate180(),
            4 => img.flipv(),
            5 => img.rotate90().fliph(),
            6 => img.rotate90(),
            7 => img.rotate270().fliph(),
            8 => img.rotate270(),
            _ => img,
        })
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
        for (i, (img1, img2)) in from_path_images
            .iter()
            .zip(from_memory_images.iter())
            .enumerate()
        {
            tracing::info!("Comparing image tensor at index {}", i);
            tracing::info!(" - Image from path shape: {:?}", img1.shape());
            tracing::info!(" - Image from memory shape: {:?}", img2.shape());
            assert_eq!(
                img1.shape(),
                img2.shape(),
                "Image tensors at index {} have different shapes: {:?} vs {:?}",
                i,
                img1.shape(),
                img2.shape()
            );
        }
        assert_eq!(from_path_images, from_memory_images);
    }
}
