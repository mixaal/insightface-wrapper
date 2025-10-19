use insightface_wrapper::{
    error::InsightFaceError,
    face::{FaceDetectionModel, FaceEmbeddingModel},
    image::ImageBatchLoader,
};

const DIMENSIONS: (u32, u32) = (640, 640);

fn main() -> Result<(), InsightFaceError> {
    tracing_subscriber::fmt::init();
    let img = ImageBatchLoader::read_from_path(vec!["images/20250711_112226.jpg"], DIMENSIONS)?;
    let detector = FaceDetectionModel::default();
    let faces = detector.detect_faces(&img)?;
    // flatten the Vec<Vec<DetectedFace>> into Vec<DetectedFace>
    let faces = faces.into_iter().flatten().collect::<Vec<_>>();
    tracing::info!("Detected faces: {faces:#?}");

    let embedding_model = FaceEmbeddingModel::default();

    let embeddings = embedding_model.embedding(faces)?;
    tracing::info!("Face embeddings: {embeddings:#?}");

    Ok(())
}
