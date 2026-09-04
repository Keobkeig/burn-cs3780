//! Dataset utilities and common datasets for machine learning examples.
//!
//! This module provides dataset loaders and utilities for the various examples
//! in the CS3780 curriculum, including classic datasets like Iris, Boston Housing,
//! and synthetic datasets for testing algorithms.

use burn::tensor::{backend::Backend, Tensor, TensorData};
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

/// A generic dataset structure that holds features and labels
#[derive(Debug, Clone)]
pub struct Dataset<B: Backend> {
    /// Feature matrix: [num_samples, num_features]
    pub features: Tensor<B, 2>,
    /// Labels: [num_samples] for regression, [num_samples, num_classes] for classification
    pub labels: Tensor<B, 2>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Class names (for classification)
    pub class_names: Vec<String>,
}

impl<B: Backend> Dataset<B> {
    /// Create a new dataset
    pub fn new(
        features: Tensor<B, 2>,
        labels: Tensor<B, 2>,
        feature_names: Vec<String>,
        class_names: Vec<String>,
    ) -> Self {
        Self {
            features,
            labels,
            feature_names,
            class_names,
        }
    }

    /// Get the number of samples
    pub fn num_samples(&self) -> usize {
        self.features.dims()[0]
    }

    /// Get the number of features
    pub fn num_features(&self) -> usize {
        self.features.dims()[1]
    }

    /// Get a subset of the dataset by indices
    pub fn subset(&self, indices: &[usize]) -> Self {
        let indices_tensor = Tensor::from_ints(
            indices
                .iter()
                .map(|&i| i as i64)
                .collect::<Vec<_>>()
                .as_slice(),
            &self.features.device(),
        );

        let features = self.features.clone().select(0, indices_tensor.clone());
        let labels = self.labels.clone().select(0, indices_tensor);

        Self {
            features,
            labels,
            feature_names: self.feature_names.clone(),
            class_names: self.class_names.clone(),
        }
    }

    /// Split dataset into training and testing sets
    pub fn train_test_split(&self, train_ratio: f32, seed: Option<u64>) -> (Self, Self) {
        let mut rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        let num_samples = self.num_samples();
        let num_train = (num_samples as f32 * train_ratio) as usize;

        // Create shuffled indices
        let mut indices: Vec<usize> = (0..num_samples).collect();
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        let train_indices = &indices[..num_train];
        let test_indices = &indices[num_train..];

        (self.subset(train_indices), self.subset(test_indices))
    }
}

/// Generate a linearly separable 2D dataset
pub fn make_linearly_separable<B: Backend>(
    n_samples: usize,
    device: &B::Device,
    seed: Option<u64>,
) -> Dataset<B> {
    let mut rng = match seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_entropy(),
    };

    let mut features_data = Vec::new();
    let mut labels_data = Vec::new();

    for _ in 0..n_samples {
        let x1: f32 = rng.sample(StandardNormal);
        let x2: f32 = rng.sample(StandardNormal);

        // Create a linear decision boundary: y = 0.5 * x1 - x2 + 0.5
        let decision_value = 0.5 * x1 - x2 + 0.5;
        let label = if decision_value > 0.0 { 1.0 } else { 0.0 };

        features_data.extend_from_slice(&[x1, x2]);
        labels_data.push(label);
    }

    // Convert to tensors
    let features = Tensor::from_data(TensorData::new(features_data, [n_samples, 2]), device);

    let labels = Tensor::from_data(TensorData::new(labels_data, [n_samples, 1]), device);

    Dataset::new(
        features,
        labels,
        vec!["x1".to_string(), "x2".to_string()],
        vec!["Class 0".to_string(), "Class 1".to_string()],
    )
}

/// Generate XOR dataset (non-linearly separable)
pub fn make_xor_dataset<B: Backend>(
    n_samples: usize,
    noise: f32,
    device: &B::Device,
    seed: Option<u64>,
) -> Dataset<B> {
    let mut rng = match seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_entropy(),
    };

    let mut features_data = Vec::new();
    let mut labels_data = Vec::new();

    let samples_per_quadrant = n_samples / 4;

    // Generate samples in each quadrant
    let quadrants = [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
    ];

    for &(x_center, y_center, label) in &quadrants {
        for _ in 0..samples_per_quadrant {
            let x1 = x_center + noise * rng.sample::<f32, _>(StandardNormal);
            let x2 = y_center + noise * rng.sample::<f32, _>(StandardNormal);

            features_data.extend_from_slice(&[x1, x2]);
            labels_data.push(label);
        }
    }

    // Convert to tensors for XOR. The quadrants split n_samples evenly, so
    // the real count is the truncated one — using n_samples here made any
    // request that was not a multiple of four panic on the shape.
    let actual_samples = samples_per_quadrant * 4;
    let features = Tensor::from_data(TensorData::new(features_data, [actual_samples, 2]), device);

    let labels = Tensor::from_data(TensorData::new(labels_data, [actual_samples, 1]), device);

    Dataset::new(
        features,
        labels,
        vec!["x1".to_string(), "x2".to_string()],
        vec!["Class 0".to_string(), "Class 1".to_string()],
    )
}

/// Generate a regression dataset with polynomial features
pub fn make_polynomial_regression<B: Backend>(
    n_samples: usize,
    degree: usize,
    noise: f32,
    device: &B::Device,
    seed: Option<u64>,
) -> Dataset<B> {
    let mut rng = match seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_entropy(),
    };

    let mut features_data = Vec::new();
    let mut labels_data = Vec::new();

    // Generate coefficients for polynomial
    let mut coeffs = Vec::new();
    for _ in 0..=degree {
        coeffs.push(rng.sample::<f32, _>(StandardNormal));
    }

    for _ in 0..n_samples {
        let x: f32 = rng.gen_range(-2.0..2.0);

        // Compute polynomial value
        let mut y = 0.0;
        for (i, &coeff) in coeffs.iter().enumerate() {
            y += coeff * x.powi(i as i32);
        }

        // Add noise
        y += noise * rng.sample::<f32, _>(StandardNormal);

        features_data.push(x);
        labels_data.push(y);
    }

    // Convert to tensors for polynomial
    let features = Tensor::from_data(TensorData::new(features_data, [n_samples, 1]), device);

    let labels = Tensor::from_data(TensorData::new(labels_data, [n_samples, 1]), device);

    Dataset::new(
        features,
        labels,
        vec!["x".to_string()],
        vec!["y".to_string()],
    )
}

/// Generate clustering dataset with multiple Gaussian blobs
pub fn make_blobs<B: Backend>(
    n_samples: usize,
    n_centers: usize,
    cluster_std: f32,
    device: &B::Device,
    seed: Option<u64>,
) -> Dataset<B> {
    let mut rng = match seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_entropy(),
    };

    // Generate random centers
    let mut centers = Vec::new();
    for _ in 0..n_centers {
        let x = rng.gen_range(-5.0..5.0);
        let y = rng.gen_range(-5.0..5.0);
        centers.push((x, y));
    }

    let samples_per_center = n_samples / n_centers;
    let mut features_data = Vec::new();
    let mut labels_data = Vec::new();

    for (center_idx, &(center_x, center_y)) in centers.iter().enumerate() {
        for _ in 0..samples_per_center {
            let x = center_x + cluster_std * rng.sample::<f32, _>(StandardNormal);
            let y = center_y + cluster_std * rng.sample::<f32, _>(StandardNormal);

            features_data.extend_from_slice(&[x, y]);
            labels_data.push(center_idx as f32);
        }
    }

    // Convert to tensors for clustering
    let actual_samples = samples_per_center * n_centers;
    let features = Tensor::from_data(TensorData::new(features_data, [actual_samples, 2]), device);

    let labels = Tensor::from_data(TensorData::new(labels_data, [actual_samples, 1]), device);

    let class_names: Vec<String> = (0..n_centers).map(|i| format!("Cluster {}", i)).collect();

    Dataset::new(
        features,
        labels,
        vec!["x1".to_string(), "x2".to_string()],
        class_names,
    )
}

/// Generate grayscale images of four shapes: disc, square, ring, triangle.
///
/// Each image is `size x size` with the shape at a random centre, radius and
/// (for the square and triangle) rotation, so a classifier cannot win by
/// memorizing positions — it has to learn something about form. Returned as
/// flat pixel rows in `[0, 1]`, shaped `[n_per_class * 4, size * size]`, with
/// the class index as the label.
///
/// # Arguments
/// * `n_per_class` - Images generated for each of the four shapes
/// * `size` - Side length in pixels
/// * `device` - Device to allocate the tensors on
/// * `seed` - Fixed seed for reproducibility
pub fn make_shape_images<B: Backend>(
    n_per_class: usize,
    size: usize,
    device: &B::Device,
    seed: Option<u64>,
) -> Dataset<B> {
    let mut rng = match seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_entropy(),
    };

    let mut features_data = Vec::with_capacity(n_per_class * 4 * size * size);
    let mut labels_data = Vec::with_capacity(n_per_class * 4);

    for class in 0..4 {
        for _ in 0..n_per_class {
            // In [-1, 1] coordinates, so a radius of 0.6 fills most of a
            // 16-pixel image. Smaller than this and the ring's hole closes up.
            let radius: f32 = rng.gen_range(0.5..0.75);
            let jitter = (1.0 - radius) * 0.9;
            let cx: f32 = rng.gen_range(-jitter..jitter);
            let cy: f32 = rng.gen_range(-jitter..jitter);
            let angle: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
            let (sin, cos) = angle.sin_cos();

            for row in 0..size {
                for col in 0..size {
                    // Pixel centre in [-1, 1].
                    let px = ((col as f32 + 0.5) / size as f32) * 2.0 - 1.0 - cx;
                    let py = ((row as f32 + 0.5) / size as f32) * 2.0 - 1.0 - cy;
                    // Rotate into the shape's frame.
                    let x = px * cos + py * sin;
                    let y = -px * sin + py * cos;

                    // Signed distance: positive inside.
                    let inside = match class {
                        0 => radius - (x * x + y * y).sqrt(),
                        1 => radius - x.abs().max(y.abs()),
                        2 => {
                            // Ring: inside the outer edge and outside the hole.
                            let r = (x * x + y * y).sqrt();
                            (radius - r).min(r - radius * 0.55)
                        }
                        _ => {
                            // Triangle: below the base line, and inside both
                            // slanted sides. The half-width tapers linearly
                            // from `radius` at the base to zero at the apex.
                            (radius - y).min((y + radius) * 0.5 - x.abs())
                        }
                    };

                    // Soft edges: a pixel's value is how far inside the
                    // shape it is, scaled so the transition is about one
                    // pixel wide. Hard edges alias badly enough at these
                    // sizes to read as noise.
                    let value = (inside * size as f32 * 0.5).clamp(0.0, 1.0);
                    features_data.push(value);
                }
            }

            labels_data.push(class as f32);
        }
    }

    let n_samples = n_per_class * 4;
    let features = Tensor::from_data(
        TensorData::new(features_data, [n_samples, size * size]),
        device,
    );
    let labels = Tensor::from_data(TensorData::new(labels_data, [n_samples, 1]), device);

    Dataset::new(
        features,
        labels,
        (0..size * size).map(|i| format!("px{}", i)).collect(),
        vec![
            "Disc".to_string(),
            "Square".to_string(),
            "Ring".to_string(),
            "Triangle".to_string(),
        ],
    )
}
