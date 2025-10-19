use linfa::traits::Transformer;
use linfa_clustering::Dbscan;

use crate::{error::InsightFaceError, face::FaceEmbedding, math::normalize_embeddings};
use std::collections::HashMap;

pub fn cluster_faces(
    embeddings: Vec<[f32; 512]>,
    tolerance: f32,
    min_points: usize,
) -> Result<Vec<Option<usize>>, InsightFaceError> {
    // Normalize first!
    let dataset = normalize_embeddings(embeddings);

    // Now Euclidean distance ≈ cosine distance
    let model = Dbscan::params(min_points).tolerance(tolerance); // Adjust based on normalized space

    // Get cluster memberships for each record
    let cluster_memberships = model
        .transform(&dataset)
        .map_err(|e| InsightFaceError::new(e))?;

    let result = cluster_memberships.to_vec();

    // Count labels
    let mut label_count: HashMap<Option<usize>, usize> = HashMap::new();
    for &label in &result {
        *label_count.entry(label).or_insert(0) += 1;
    }
    tracing::info!("---------------------------------------------------------");
    tracing::info!("Clustering results:");
    for (label, count) in label_count {
        match label {
            None => tracing::info!(" - {count} noise points"),
            Some(i) => tracing::info!(" - {count} points in cluster {i}"),
        }
    }
    tracing::info!("---------------------------------------------------------");

    Ok(result)
}

pub fn cluster_and_organize(
    faces: Vec<FaceEmbedding>,
    tolerance: f32,
    min_points: usize,
) -> Result<Vec<Vec<FaceEmbedding>>, InsightFaceError> {
    let embeddings: Vec<[f32; 512]> = faces.iter().map(|f| f.vector).collect();

    let labels = cluster_faces(embeddings, tolerance, min_points)?;

    let mut clusters: std::collections::HashMap<usize, Vec<FaceEmbedding>> =
        std::collections::HashMap::new();

    for (face, label) in faces.into_iter().zip(labels.into_iter()) {
        if let Some(cluster_id) = label {
            clusters
                .entry(cluster_id)
                .or_insert_with(Vec::new)
                .push(face);
        }
    }

    Ok(clusters.into_values().collect())
}
