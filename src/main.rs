use std::collections::HashMap;

use insightface_wrapper::{
    cluster,
    error::InsightFaceError,
    face::{FaceDetectionModel, FaceEmbeddingModel},
    image::ImageBatchLoader,
};

const DIMENSIONS: (u32, u32) = (640, 640);

fn main() -> Result<(), InsightFaceError> {
    let mut face_lookup = HashMap::new();
    tracing_subscriber::fmt::init();
    let images = vec![
        "images/20250711_112226.jpg",
        "images/20250711_112229.jpg",
        "images/20250716_154632.jpg",
        "images/20250716_154633.jpg",
        "images/20251018_144943.jpg",
        "images/20251018_144948.jpg",
    ];
    let img = ImageBatchLoader::read_from_path(&images, DIMENSIONS)?;
    let detector = FaceDetectionModel::default();

    let faces = detector.detect_faces(&img)?;
    let mut picno = 0;

    for faces_inside_pic in &faces {
        let img_name = &images[picno];
        for f in faces_inside_pic {
            face_lookup.insert(f.face_id, (img_name, f.score, f.shape));
        }
        picno += 1;
    }
    // flatten the Vec<Vec<DetectedFace>> into Vec<DetectedFace>
    let faces = faces.into_iter().flatten().collect::<Vec<_>>();
    tracing::info!("Detected faces: {:#?}", faces.len());
    tracing::info!("Detected faces: {:#?}", face_lookup);

    let embedding_model = FaceEmbeddingModel::default();

    let embeddings = embedding_model.embedding(faces)?;

    let labels = cluster::cluster_and_organize(embeddings, 0.65, 2)?;
    for cluster_id in 0..labels.len() {
        tracing::info!("Cluster {cluster_id}:");
        for face in &labels[cluster_id] {
            let (img_name, score, shape) = face_lookup.get(&face.face_id).unwrap();
            tracing::info!(
                " - Image: {}, Score: {}, shape: {:?}",
                img_name,
                score,
                shape
            );
        }
    }
    Ok(())
}
