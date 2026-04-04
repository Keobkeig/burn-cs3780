//! k-Nearest Neighbors (k-NN) implementation using Burn.
//!
//! This module implements the k-NN algorithm for both classification and regression.
//! k-NN is a non-parametric, lazy learning algorithm that makes predictions based on
//! the k nearest neighbors in the training data.

use crate::utils::Distance;
use burn::tensor::{backend::Backend, Tensor};
use std::collections::HashMap;

/// k-Nearest Neighbors classifier/regressor
#[derive(Debug, Clone)]
pub struct KNearestNeighbors<B: Backend<FloatElem = f32>> {
    k: usize,
    distance_metric: DistanceMetric,
    weights: WeightFunction,
    // Stored training data
    x_train: Option<Tensor<B, 2>>,
    y_train: Option<Tensor<B, 1>>,
    is_classification: bool,
}

/// Available distance metrics
#[derive(Debug, Clone)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance
    Euclidean,
    /// Manhattan (L1) distance
    Manhattan,
    /// Cosine distance
    Cosine,
}

/// Weight functions for neighbors
#[derive(Debug, Clone)]
pub enum WeightFunction {
    /// All neighbors weighted equally
    Uniform,
    /// Weight by inverse distance
    Distance,
    /// Exponential decay with distance
    Exponential,
}

impl<B: Backend<FloatElem = f32>> KNearestNeighbors<B> {
    /// Create a new k-NN model
    pub fn new(k: usize) -> Self {
        Self {
            k,
            distance_metric: DistanceMetric::Euclidean,
            weights: WeightFunction::Uniform,
            x_train: None,
            y_train: None,
            is_classification: true,
        }
    }

    /// Create k-NN for regression
    pub fn new_regressor(k: usize) -> Self {
        Self {
            k,
            distance_metric: DistanceMetric::Euclidean,
            weights: WeightFunction::Uniform,
            x_train: None,
            y_train: None,
            is_classification: false,
        }
    }

    /// Set distance metric
    pub fn with_distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Set weight function
    pub fn with_weights(mut self, weights: WeightFunction) -> Self {
        self.weights = weights;
        self
    }

    /// Fit the model (store training data)
    pub fn fit(&mut self, x_train: Tensor<B, 2>, y_train: Tensor<B, 1>) {
        self.x_train = Some(x_train);
        self.y_train = Some(y_train);
    }

    /// Predict for new samples
    pub fn predict(&self, x_test: &Tensor<B, 2>) -> Tensor<B, 1> {
        let x_train = self.x_train.as_ref().expect("Model not fitted");
        let y_train = self.y_train.as_ref().expect("Model not fitted");

        let n_test = x_test.dims()[0];
        let mut predictions = Vec::with_capacity(n_test);

        for i in 0..n_test {
            let test_sample = x_test.clone().slice([i..i + 1]).squeeze::<1>();
            let prediction = self.predict_single(&test_sample, x_train, y_train);
            predictions.push(prediction);
        }

        Tensor::from_floats(predictions.as_slice(), &x_test.device())
    }

    /// Predict for a single sample
    fn predict_single(
        &self,
        test_sample: &Tensor<B, 1>,
        x_train: &Tensor<B, 2>,
        y_train: &Tensor<B, 1>,
    ) -> f32 {
        let n_train = x_train.dims()[0];

        // Compute distances to all training samples
        let mut distances_with_indices = Vec::with_capacity(n_train);

        for i in 0..n_train {
            let train_sample = x_train.clone().slice([i..i + 1]).squeeze::<1>();
            let distance = self.compute_distance(test_sample, &train_sample);
            let label: f32 = y_train
                .clone()
                .slice([i..i + 1])
                .squeeze::<1>()
                .into_scalar();
            distances_with_indices.push((distance, label, i));
        }

        // Sort by distance and take k nearest neighbors
        distances_with_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let k_neighbors = &distances_with_indices[..self.k.min(n_train)];

        if self.is_classification {
            self.predict_classification(k_neighbors)
        } else {
            self.predict_regression(k_neighbors)
        }
    }

    /// Predict for classification (majority vote)
    fn predict_classification(&self, neighbors: &[(f32, f32, usize)]) -> f32 {
        let mut class_votes: HashMap<i32, f32> = HashMap::new();

        for &(distance, label, _) in neighbors {
            let class = label as i32;
            let weight = self.compute_weight(distance);
            *class_votes.entry(class).or_insert(0.0) += weight;
        }

        // Return class with maximum vote
        class_votes
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(class, _)| class as f32)
            .unwrap_or(0.0)
    }

    /// Predict for regression (weighted average)
    fn predict_regression(&self, neighbors: &[(f32, f32, usize)]) -> f32 {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for &(distance, label, _) in neighbors {
            let weight = self.compute_weight(distance);
            weighted_sum += weight * label;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            neighbors[0].1 // Fallback to first neighbor
        }
    }

    /// Compute distance between two samples
    fn compute_distance(&self, x1: &Tensor<B, 1>, x2: &Tensor<B, 1>) -> f32 {
        match self.distance_metric {
            DistanceMetric::Euclidean => Distance::euclidean(x1, x2),
            DistanceMetric::Manhattan => Distance::manhattan(x1, x2),
            DistanceMetric::Cosine => 1.0 - Distance::cosine_similarity(x1, x2),
        }
    }

    /// Compute weight based on distance
    fn compute_weight(&self, distance: f32) -> f32 {
        match self.weights {
            WeightFunction::Uniform => 1.0,
            WeightFunction::Distance => {
                if distance == 0.0 {
                    1e6
                } else {
                    1.0 / distance
                }
            }
            WeightFunction::Exponential => (-distance).exp(),
        }
    }

    /// Get k nearest neighbors for a sample (for analysis)
    pub fn get_neighbors(&self, test_sample: &Tensor<B, 1>) -> Vec<(f32, f32, usize)> {
        let x_train = self.x_train.as_ref().expect("Model not fitted");
        let y_train = self.y_train.as_ref().expect("Model not fitted");
        let n_train = x_train.dims()[0];

        let mut distances_with_indices = Vec::with_capacity(n_train);

        for i in 0..n_train {
            let train_sample = x_train.clone().slice([i..i + 1]).squeeze::<1>();
            let distance = self.compute_distance(test_sample, &train_sample);
            let label: f32 = y_train
                .clone()
                .slice([i..i + 1])
                .squeeze::<1>()
                .into_scalar();
            distances_with_indices.push((distance, label, i));
        }

        distances_with_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        distances_with_indices
            .into_iter()
            .take(self.k.min(n_train))
            .collect()
    }

    /// Predict with confidence (classification only)
    pub fn predict_proba(&self, x_test: &Tensor<B, 2>) -> Vec<HashMap<i32, f32>> {
        if !self.is_classification {
            panic!("predict_proba only available for classification");
        }

        let _x_train = self.x_train.as_ref().expect("Model not fitted");
        let _y_train = self.y_train.as_ref().expect("Model not fitted");
        let n_test = x_test.dims()[0];

        let mut probabilities = Vec::with_capacity(n_test);

        for i in 0..n_test {
            let test_sample = x_test.clone().slice([i..i + 1]).squeeze::<1>();
            let neighbors = self.get_neighbors(&test_sample);

            let mut class_weights: HashMap<i32, f32> = HashMap::new();
            let mut total_weight = 0.0;

            for &(distance, label, _) in &neighbors {
                let class = label as i32;
                let weight = self.compute_weight(distance);
                *class_weights.entry(class).or_insert(0.0) += weight;
                total_weight += weight;
            }

            // Normalize to probabilities
            if total_weight > 0.0 {
                for (_, weight) in class_weights.iter_mut() {
                    *weight /= total_weight;
                }
            }

            probabilities.push(class_weights);
        }

        probabilities
    }
}

/// Utilities for k-NN
pub struct KNNUtils;

impl KNNUtils {
    /// Find optimal k using cross-validation
    pub fn find_optimal_k<B: Backend<FloatElem = f32>>(
        x: &Tensor<B, 2>,
        y: &Tensor<B, 1>,
        k_range: std::ops::Range<usize>,
        cv_folds: usize,
        is_classification: bool,
    ) -> (usize, Vec<f32>) {
        use crate::metrics::{ClassificationMetrics, CrossValidation, RegressionMetrics};

        let n_samples = x.dims()[0];
        let cv_indices = CrossValidation::k_fold_indices(n_samples, cv_folds, true, Some(42));

        let mut k_scores = Vec::new();
        let mut best_k = k_range.start;
        let mut best_score = f32::NEG_INFINITY;

        for k in k_range {
            let mut fold_scores = Vec::new();

            for (train_indices, test_indices) in &cv_indices {
                // Create train/test splits
                let train_indices_tensor = Tensor::from_ints(
                    train_indices
                        .iter()
                        .map(|&i| i as i64)
                        .collect::<Vec<_>>()
                        .as_slice(),
                    &x.device(),
                );
                let test_indices_tensor = Tensor::from_ints(
                    test_indices
                        .iter()
                        .map(|&i| i as i64)
                        .collect::<Vec<_>>()
                        .as_slice(),
                    &x.device(),
                );

                let x_train = x.clone().select(0, train_indices_tensor.clone());
                let y_train = y.clone().select(0, train_indices_tensor);
                let x_test = x.clone().select(0, test_indices_tensor.clone());
                let y_test = y.clone().select(0, test_indices_tensor);

                // Train and evaluate k-NN
                let mut knn = if is_classification {
                    KNearestNeighbors::new(k)
                } else {
                    KNearestNeighbors::new_regressor(k)
                };
                knn.fit(x_train, y_train);

                let y_pred = knn.predict(&x_test);

                let score = if is_classification {
                    ClassificationMetrics::accuracy(&y_test, &y_pred)
                } else {
                    -RegressionMetrics::mse(&y_test, &y_pred) // Negative MSE for maximization
                };

                fold_scores.push(score);
            }

            let avg_score = fold_scores.iter().sum::<f32>() / fold_scores.len() as f32;
            k_scores.push(avg_score);

            if avg_score > best_score {
                best_score = avg_score;
                best_k = k;
            }
        }

        (best_k, k_scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{datasets, DefaultBackend};

    #[test]
    fn test_knn_classification() {
        let device = Default::default();

        // Create simple linearly separable data
        let dataset = datasets::make_linearly_separable::<DefaultBackend>(100, &device, Some(42));
        let (train_data, test_data) = dataset.train_test_split(0.8, Some(42));

        let mut knn = KNearestNeighbors::new(3);
        knn.fit(train_data.features, train_data.labels.squeeze(1));

        let predictions = knn.predict(&test_data.features);

        // Should have reasonable accuracy on linearly separable data
        use crate::metrics::ClassificationMetrics;
        let accuracy = ClassificationMetrics::accuracy(&test_data.labels.squeeze(1), &predictions);
        assert!(
            accuracy > 0.7,
            "Accuracy should be > 70% on linearly separable data"
        );
    }

    #[test]
    fn test_knn_regression() {
        let device = Default::default();

        // Create polynomial regression data
        let dataset =
            datasets::make_polynomial_regression::<DefaultBackend>(100, 2, 0.1, &device, Some(42));
        let (train_data, test_data) = dataset.train_test_split(0.8, Some(42));

        let mut knn = KNearestNeighbors::new_regressor(5);
        knn.fit(train_data.features, train_data.labels.squeeze(1));

        let predictions = knn.predict(&test_data.features);

        // Should have reasonable MSE
        use crate::metrics::RegressionMetrics;
        let mse = RegressionMetrics::mse(&test_data.labels.squeeze(1), &predictions);
        assert!(mse < 1.0, "MSE should be reasonable for polynomial data");
    }

    #[test]
    fn test_knn_different_distance_metrics() {
        let device = Default::default();
        let dataset = datasets::make_linearly_separable::<DefaultBackend>(50, &device, Some(42));
        let (train_data, test_data) = dataset.train_test_split(0.8, Some(42));

        for metric in [
            DistanceMetric::Euclidean,
            DistanceMetric::Manhattan,
            DistanceMetric::Cosine,
        ] {
            let mut knn = KNearestNeighbors::new(3).with_distance_metric(metric);
            knn.fit(
                train_data.features.clone(),
                train_data.labels.clone().squeeze(1),
            );

            let predictions = knn.predict(&test_data.features);

            // Should produce valid predictions
            assert_eq!(predictions.dims()[0], test_data.features.dims()[0]);
        }
    }
}
