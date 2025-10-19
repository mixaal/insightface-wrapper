use image::Rgba32FImage;
use ort::session::{Session, builder::GraphOptimizationLevel};

use crate::{
    error::InsightFaceError,
    image::{ImageBatchLoader, ImageWrapper},
    utils,
};

// just a simple struct to hold detected face information (same as in insightface crate)
#[derive(Debug)]
pub struct DetectedFace {
    pub score: f32,
    pub image_face: FaceProps, // to hold scaled face data (as per input image dimensions)
    pub orig_face: FaceProps,  // to hold original non-scaled face data (640,640) dimensions
    pub cropped_face: Rgba32FImage,
}

#[derive(Debug)]
pub struct FaceProps {
    bbox: (f32, f32, f32, f32),
    keypoints: [(f32, f32); 5],
}

impl FaceProps {
    pub fn new(bbox: (f32, f32, f32, f32), keypoints: [(f32, f32); 5]) -> Self {
        Self { bbox, keypoints }
    }

    pub fn eq(&self, other: &FaceProps, eps: f32) -> bool {
        (self.bbox.0 - other.bbox.0).abs() < eps
            && (self.bbox.1 - other.bbox.1).abs() < eps
            && (self.bbox.2 - other.bbox.2).abs() < eps
            && (self.bbox.3 - other.bbox.3).abs() < eps
            && self
                .keypoints
                .iter()
                .zip(other.keypoints.iter())
                .all(|((x1, y1), (x2, y2))| (x1 - x2).abs() < eps && (y1 - y2).abs() < eps)
    }
}

impl DetectedFace {
    fn from_face(face: &insightface::Face, img: &ImageWrapper) -> Self {
        let cropped_face = insightface::crop_face(&img.image, &face.keypoints, 112);
        let orig_face = FaceProps {
            bbox: face.bbox,
            keypoints: face.keypoints,
        };
        let (scale_x, scale_y) = img.scale_factor();

        let keypoints: [(f32, f32); 5] = [
            (face.keypoints[0].0 * scale_x, face.keypoints[0].1 * scale_y),
            (face.keypoints[1].0 * scale_x, face.keypoints[1].1 * scale_y),
            (face.keypoints[2].0 * scale_x, face.keypoints[2].1 * scale_y),
            (face.keypoints[3].0 * scale_x, face.keypoints[3].1 * scale_y),
            (face.keypoints[4].0 * scale_x, face.keypoints[4].1 * scale_y),
        ];

        let bbox = (
            face.bbox.0 * scale_x,
            face.bbox.1 * scale_y,
            face.bbox.2 * scale_x,
            face.bbox.3 * scale_y,
        );

        let image_face = FaceProps { bbox, keypoints };
        DetectedFace {
            score: face.score,
            image_face,
            orig_face,
            cropped_face,
        }
    }

    // equality check with epsilon
    pub fn eq(&self, other: &DetectedFace, eps: f32) -> bool {
        (self.score - other.score).abs() < eps
            && self.image_face.eq(&other.image_face, eps)
            && self.orig_face.eq(&other.orig_face, eps)
    }
}

pub struct FaceDetectionModel {
    model: OnnyxModel,
    detection_threshold: f32, // detection confidence threshold
    nms_threshold: f32,       // non-maximum suppression threshold
}

impl Default for FaceDetectionModel {
    fn default() -> Self {
        FaceDetectionModel::new("buffalo_l/det_10g.onnx", 0.5, 0.4)
    }
}

impl FaceDetectionModel {
    pub fn new(model: &str, detection_threshold: f32, nms_threshold: f32) -> Self {
        FaceDetectionModel {
            model: OnnyxModel::new(model),
            detection_threshold,
            nms_threshold,
        }
    }

    // detect faces in the given image batch loader
    pub fn detect_faces(
        &self,
        loader: &ImageBatchLoader,
    ) -> Result<Vec<Vec<DetectedFace>>, InsightFaceError> {
        let mut detected_faces = Vec::new();
        let mut session = self.model.create_session()?;
        for img in &loader.batch {
            let input = img.to_tensor();
            let faces = insightface::detect_faces(&mut session, input, self.detection_threshold);
            let faces = insightface::non_maximum_suppression(faces, self.nms_threshold)
                .iter()
                .map(|f| DetectedFace::from_face(f, img))
                .collect();
            detected_faces.push(faces);
        }

        return Ok(detected_faces);
    }
}

#[derive(Debug)]
pub struct FaceEmbedding {
    pub vector: [f32; 512],
}

pub struct FaceEmbeddingModel {
    model: OnnyxModel,
}

impl FaceEmbeddingModel {
    pub fn new(model: &str) -> Self {
        FaceEmbeddingModel {
            model: OnnyxModel::new(model),
        }
    }

    pub fn embedding(
        &self,
        faces: Vec<DetectedFace>,
    ) -> Result<Vec<FaceEmbedding>, InsightFaceError> {
        let mut face_embedding = self.model.create_session()?;

        let mut embeddings = Vec::new();
        for face in faces {
            let face_tensor = utils::to_tensor(&face.cropped_face);
            let embedding = insightface::calculate_embedding(&mut face_embedding, face_tensor);
            embeddings.push(FaceEmbedding { vector: embedding });
        }

        Ok(embeddings)
    }
}

impl Default for FaceEmbeddingModel {
    fn default() -> Self {
        FaceEmbeddingModel::new("buffalo_l/w600k_r50.onnx")
    }
}

struct OnnyxModel {
    model: String,
}

impl OnnyxModel {
    fn new(model: &str) -> Self {
        OnnyxModel {
            model: model.to_string(),
        }
    }

    fn get_model_path(&self) -> String {
        let home = std::env::var("HOME").unwrap_or(".".to_string());
        let model_path =
            std::env::var("MODEL_PATH").unwrap_or(format!("{}/.insightface/models", home));
        format!("{}/{}", model_path, self.model)
    }

    fn create_session(&self) -> Result<Session, InsightFaceError> {
        let model_filepath = self.get_model_path();
        let session = Session::builder()
            .map_err(|e| InsightFaceError::new(e))?
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .map_err(|e| InsightFaceError::new(e))?
            .commit_from_file(model_filepath)
            .map_err(|e| InsightFaceError::new(e))?;

        return Ok(session);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_faces() {
        let img_loader =
            ImageBatchLoader::read_from_path(vec!["images/20250711_112226.jpg"], (640, 640))
                .unwrap();
        let detector = FaceDetectionModel::default();
        let faces = detector.detect_faces(&img_loader).unwrap();
        assert_eq!(faces.len(), 1);
        assert_eq!(faces[0].len(), 2); // expecting 2 faces in the image
        println!("Detected faces: {faces:#?}");
        let edita = FaceProps::new(
            (113.546074, 274.89148, 255.50471, 508.63586),
            [
                (146.16258, 344.20395),
                (218.30405, 342.42096),
                (182.87538, 399.1779),
                (153.32144, 434.45605),
                (218.3465, 431.72583),
            ],
        );
        let michal = FaceProps::new(
            (379.1525, 193.72905, 556.0934, 489.22778),
            [
                (409.0408, 290.9723),
                (500.8356, 285.2258),
                (449.75168, 344.5245),
                (419.3767, 394.81952),
                (505.5358, 389.23486),
            ],
        );
        // one way or another, the order of detected faces may vary
        assert!(faces[0][0].orig_face.eq(&edita, 1e-2) || faces[0][1].orig_face.eq(&edita, 1e-2));
        assert!(faces[0][0].orig_face.eq(&michal, 1e-2) || faces[0][1].orig_face.eq(&michal, 1e-2));
    }
}
