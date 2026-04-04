//! Evaluation metrics for machine learning models.
//!
//! This module provides common evaluation metrics for both classification
//! and regression tasks, following the CS3780 curriculum.

pub mod classification;
pub mod cross_validation;
pub mod regression;

pub use classification::ClassificationMetrics;
pub use cross_validation::CrossValidation;
pub use regression::RegressionMetrics;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DefaultBackend;
    use burn::tensor::Tensor;

    #[test]
    fn test_classification_accuracy() {
        let device = Default::default();
        let y_true = Tensor::<DefaultBackend, 1>::from_floats([1.0, 0.0, 1.0, 1.0], &device);
        let y_pred = Tensor::<DefaultBackend, 1>::from_floats([1.0, 0.0, 0.0, 1.0], &device);

        let accuracy = ClassificationMetrics::accuracy(&y_true, &y_pred);
        assert!((accuracy - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_regression_mse() {
        let device = Default::default();
        let y_true = Tensor::<DefaultBackend, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device);
        let y_pred = Tensor::<DefaultBackend, 1>::from_floats([1.1, 1.9, 3.1, 3.9], &device);

        let mse = RegressionMetrics::mse(&y_true, &y_pred);
        assert!(mse < 0.1);
    }

    #[test]
    fn test_cross_validation_indices() {
        let indices = CrossValidation::k_fold_indices(10, 3, false, Some(42));
        assert_eq!(indices.len(), 3);

        let mut all_test_indices: Vec<usize> = Vec::new();
        for (_, test) in &indices {
            all_test_indices.extend(test);
        }
        all_test_indices.sort();

        let expected: Vec<usize> = (0..10).collect();
        assert_eq!(all_test_indices, expected);
    }
}
