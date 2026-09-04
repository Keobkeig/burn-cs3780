//! Perceptron algorithm implementation using Burn.
//!
//! The perceptron is one of the simplest neural network architectures and forms
//! the foundation for understanding linear classification and neural networks.

use burn::tensor::{backend::Backend, Tensor, TensorData};
use std::marker::PhantomData;

/// Single-layer perceptron for binary classification
#[derive(Debug, Clone)]
pub struct Perceptron<B: Backend<FloatElem = f32>> {
    weights: Option<Tensor<B, 1>>,
    learning_rate: f32,
    max_iter: usize,
    fit_intercept: bool,
    shuffle: bool,
    tolerance: Option<f32>,
    _phantom: PhantomData<B>,
}

impl<B: Backend<FloatElem = f32>> Perceptron<B> {
    /// Create a new perceptron
    pub fn new() -> Self {
        Self {
            weights: None,
            learning_rate: 1.0,
            max_iter: 1000,
            fit_intercept: true,
            shuffle: true,
            tolerance: None,
            _phantom: PhantomData,
        }
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set whether to fit intercept
    pub fn with_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set whether to shuffle training data
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = Some(tolerance);
        self
    }

    /// Train the perceptron
    pub fn fit(&mut self, x: &Tensor<B, 2>, y: &Tensor<B, 1>) -> Vec<f32> {
        let x_processed = if self.fit_intercept {
            self.add_intercept(x)
        } else {
            x.clone()
        };

        let n_samples = x_processed.dims()[0];
        let n_features = x_processed.dims()[1];

        // Initialize weights to zeros
        self.weights = Some(Tensor::<B, 1>::zeros([n_features], &x_processed.device()));

        let mut errors_per_epoch = Vec::new();
        let mut prev_errors = n_samples;

        for epoch in 0..self.max_iter {
            let mut errors = 0;
            let mut sample_indices: Vec<usize> = (0..n_samples).collect();

            // Shuffle samples if requested
            if self.shuffle {
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                sample_indices.shuffle(&mut rng);
            }

            // Process each sample
            for &idx in &sample_indices {
                let x_sample = x_processed.clone().slice([idx..idx + 1]).squeeze::<1>();
                let y_sample: f32 = y.clone().slice([idx..idx + 1]).into_scalar();
                let y_binary = if y_sample > 0.5 { 1.0 } else { -1.0 };

                // Predict
                let prediction = self.predict_single(&x_sample);
                let predicted_class = if prediction > 0.0 { 1.0 } else { -1.0 };

                // Update if prediction is wrong
                if predicted_class != y_binary {
                    errors += 1;
                    let update = x_sample.mul_scalar(self.learning_rate * y_binary);
                    if let Some(ref mut weights) = self.weights {
                        *weights = weights.clone().add(update);
                    }
                }
            }

            errors_per_epoch.push(errors as f32 / n_samples as f32);

            // Check convergence
            if errors == 0 {
                println!("Perceptron converged after {} epochs", epoch + 1);
                break;
            }

            if let Some(tol) = self.tolerance {
                if (prev_errors as f32 - errors as f32).abs() < tol {
                    println!(
                        "Perceptron converged (tolerance) after {} epochs",
                        epoch + 1
                    );
                    break;
                }
            }

            prev_errors = errors;
        }

        errors_per_epoch
    }

    /// Predict raw scores (before applying sign)
    pub fn decision_function(&self, x: &Tensor<B, 2>) -> Tensor<B, 1> {
        let weights = self.weights.as_ref().expect("Perceptron not trained");
        let x_processed = if self.fit_intercept {
            self.add_intercept(x)
        } else {
            x.clone()
        };

        let n_samples = x_processed.dims()[0];
        x_processed
            .matmul(weights.clone().unsqueeze_dim(1))
            // reshape, not squeeze: a single row gives [1, 1], and squeeze
            // removes every unit axis, leaving nothing.
            .reshape([n_samples])
    }

    /// Predict classes (0 or 1)
    pub fn predict(&self, x: &Tensor<B, 2>) -> Tensor<B, 1> {
        let scores = self.decision_function(x);
        scores.greater_elem(0.0).float()
    }

    /// Predict for a single sample
    fn predict_single(&self, x_sample: &Tensor<B, 1>) -> f32 {
        let weights = self.weights.as_ref().expect("Perceptron not trained");
        let dot_product: f32 = weights.clone().mul(x_sample.clone()).sum().into_scalar();
        dot_product
    }

    /// Add intercept (bias) term
    fn add_intercept(&self, x: &Tensor<B, 2>) -> Tensor<B, 2> {
        let [n_samples, _] = x.dims();
        let ones = Tensor::<B, 2>::ones([n_samples, 1], &x.device());
        Tensor::cat(vec![ones, x.clone()], 1)
    }

    /// Get learned weights
    pub fn weights(&self) -> Option<Tensor<B, 1>> {
        self.weights.clone()
    }

    /// Get bias term (if intercept was fitted)
    pub fn bias(&self) -> Option<f32> {
        if self.fit_intercept {
            self.weights
                .as_ref()
                .map(|w| w.clone().slice([0..1]).into_scalar())
        } else {
            None
        }
    }

    /// Get feature weights (excluding bias)
    pub fn coef(&self) -> Option<Tensor<B, 1>> {
        self.weights.clone().map(|w| {
            if self.fit_intercept {
                w.clone().slice([1..w.dims()[0]])
            } else {
                w
            }
        })
    }
}

/// Multi-class perceptron using one-vs-rest strategy
#[derive(Debug, Clone)]
pub struct MultiClassPerceptron<B: Backend<FloatElem = f32>> {
    perceptrons: Vec<Perceptron<B>>,
    classes: Vec<i32>,
    _phantom: PhantomData<B>,
}

impl<B: Backend<FloatElem = f32>> MultiClassPerceptron<B> {
    /// Create a new multi-class perceptron
    pub fn new() -> Self {
        Self {
            perceptrons: Vec::new(),
            classes: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Set perceptron parameters for all binary classifiers
    pub fn with_params(self, _learning_rate: f32, _max_iter: usize, _fit_intercept: bool) -> Self {
        self
    }

    /// Train the multi-class perceptron
    pub fn fit(&mut self, x: &Tensor<B, 2>, y: &Tensor<B, 1>) -> Vec<Vec<f32>> {
        // Find unique classes
        let n_samples = y.dims()[0];
        let mut class_set = std::collections::HashSet::new();

        for i in 0..n_samples {
            let label: f32 = y.clone().slice([i..i + 1]).into_scalar();
            class_set.insert(label as i32);
        }

        self.classes = class_set.into_iter().collect();
        self.classes.sort();

        let mut all_errors = Vec::new();

        // Train one perceptron per class (one-vs-rest)
        for &class_label in &self.classes {
            println!("Training perceptron for class {}", class_label);

            // Create binary labels (1 for current class, -1 for others)
            let mut binary_labels = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let label: f32 = y.clone().slice([i..i + 1]).into_scalar();
                binary_labels.push(if (label as i32) == class_label {
                    1.0
                } else {
                    0.0
                });
            }

            let binary_y = Tensor::from_floats(binary_labels.as_slice(), &y.device());

            let mut perceptron = Perceptron::new()
                .with_learning_rate(1.0)
                .with_max_iter(1000);

            let errors = perceptron.fit(x, &binary_y);
            self.perceptrons.push(perceptron);
            all_errors.push(errors);
        }

        all_errors
    }

    /// Predict classes for new samples
    pub fn predict(&self, x: &Tensor<B, 2>) -> Tensor<B, 1> {
        let n_samples = x.dims()[0];
        let mut predictions = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let sample = x.clone().slice([i..i + 1]);
            let mut max_score = f32::NEG_INFINITY;
            let mut predicted_class = self.classes[0];

            for (j, perceptron) in self.perceptrons.iter().enumerate() {
                let score: f32 = perceptron.decision_function(&sample).into_scalar();
                if score > max_score {
                    max_score = score;
                    predicted_class = self.classes[j];
                }
            }

            predictions.push(predicted_class as f32);
        }

        Tensor::from_floats(predictions.as_slice(), &x.device())
    }

    /// Get decision scores for all classes
    pub fn decision_function(&self, x: &Tensor<B, 2>) -> Tensor<B, 2> {
        let n_samples = x.dims()[0];
        let n_classes = self.classes.len();
        let mut scores = Vec::with_capacity(n_samples * n_classes);

        for i in 0..n_samples {
            let sample = x.clone().slice([i..i + 1]);
            for perceptron in &self.perceptrons {
                let score: f32 = perceptron.decision_function(&sample).into_scalar();
                scores.push(score);
            }
        }

        Tensor::from_data(TensorData::new(scores, [n_samples, n_classes]), &x.device())
    }

    /// Get number of classes
    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }

    /// Get class labels
    pub fn classes(&self) -> &[i32] {
        &self.classes
    }
}

/// Perceptron learning rule demonstration
pub struct PerceptronDemo;

impl PerceptronDemo {
    /// Demonstrate perceptron learning on linearly separable data
    pub fn demonstrate_learning<B: Backend<FloatElem = f32>>(
        x: &Tensor<B, 2>,
        y: &Tensor<B, 1>,
    ) -> Vec<(Tensor<B, 1>, f32)> {
        let perceptron = Perceptron::new()
            .with_learning_rate(1.0)
            .with_shuffle(false); // For reproducible demo

        // Manually implement the learning process to track weights
        let x_with_bias = perceptron.add_intercept(x);
        let n_samples = x_with_bias.dims()[0];
        let n_features = x_with_bias.dims()[1];

        let mut weights = Tensor::<B, 1>::zeros([n_features], &x_with_bias.device());
        let mut weight_history = Vec::new();

        println!("Demonstrating perceptron learning process:");
        println!("Epoch\tErrors\tWeights");
        println!("─────────────────────────────");

        for epoch in 0..50 {
            let mut errors = 0;

            for i in 0..n_samples {
                let x_sample = x_with_bias.clone().slice([i..i + 1]).squeeze::<1>();
                let y_sample: f32 = y.clone().slice([i..i + 1]).into_scalar();
                let y_binary = if y_sample > 0.5 { 1.0 } else { -1.0 };

                // Predict
                let dot_product: f32 = weights.clone().mul(x_sample.clone()).sum().into_scalar();
                let prediction = if dot_product > 0.0 { 1.0 } else { -1.0 };

                // Update if wrong
                if prediction != y_binary {
                    errors += 1;
                    let update = x_sample.mul_scalar(y_binary);
                    weights = weights.add(update);
                }
            }

            let error_rate = errors as f32 / n_samples as f32;
            weight_history.push((weights.clone(), error_rate));

            if epoch < 10 || epoch % 10 == 0 || errors == 0 {
                let w0: f32 = weights.clone().slice([0..1]).into_scalar();
                let w1: f32 = weights.clone().slice([1..2]).into_scalar();
                let w2: f32 = weights.clone().slice([2..3]).into_scalar();
                println!(
                    "{}\t{}\t[{:.3}, {:.3}, {:.3}]",
                    epoch + 1,
                    errors,
                    w0,
                    w1,
                    w2
                );
            }

            if errors == 0 {
                println!("Converged after {} epochs!", epoch + 1);
                break;
            }
        }

        weight_history
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{datasets, DefaultBackend};

    #[test]
    fn test_perceptron_linearly_separable() {
        let device = Default::default();

        // Create linearly separable data
        let dataset = datasets::make_linearly_separable::<DefaultBackend>(100, &device, Some(42));
        let (train_data, test_data) = dataset.train_test_split(0.8, Some(42));

        let mut perceptron = Perceptron::new().with_max_iter(100);
        let errors = perceptron.fit(&train_data.features, &train_data.labels.squeeze::<1>());

        // Should converge on linearly separable data
        assert!(
            errors.last().unwrap() < &0.1,
            "Should converge to low error rate"
        );

        let predictions = perceptron.predict(&test_data.features);

        // Check accuracy
        use crate::metrics::ClassificationMetrics;
        let accuracy =
            ClassificationMetrics::accuracy(&test_data.labels.squeeze::<1>(), &predictions);
        assert!(
            accuracy > 0.8,
            "Should achieve good accuracy on linearly separable data"
        );
    }

    #[test]
    fn test_multiclass_perceptron() {
        let device = Default::default();

        // Create multi-class data using blobs
        let dataset = datasets::make_blobs::<DefaultBackend>(150, 3, 1.0, &device, Some(42));
        let (train_data, test_data) = dataset.train_test_split(0.8, Some(42));

        let mut perceptron = MultiClassPerceptron::new();
        perceptron.fit(&train_data.features, &train_data.labels.squeeze::<1>());

        assert_eq!(perceptron.n_classes(), 3);

        let predictions = perceptron.predict(&test_data.features);

        // Should produce valid predictions
        assert_eq!(predictions.dims()[0], test_data.features.dims()[0]);
    }

    #[test]
    fn test_perceptron_weights() {
        let device = Default::default();
        let dataset = datasets::make_linearly_separable::<DefaultBackend>(50, &device, Some(42));

        let mut perceptron = Perceptron::new().with_intercept(true);
        perceptron.fit(&dataset.features, &dataset.labels.squeeze::<1>());

        // Should have weights and bias
        assert!(perceptron.weights().is_some());
        assert!(perceptron.bias().is_some());
        assert!(perceptron.coef().is_some());

        // Weights should be the right size
        let weights = perceptron.weights().unwrap();
        assert_eq!(weights.dims()[0], 3); // 2 features + 1 bias
    }
}
