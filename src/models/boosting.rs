//! Boosting algorithms implementation using Burn.
//!
//! This module implements various boosting algorithms:
//! - AdaBoost (Adaptive Boosting)
//! - Gradient Boosting for classification and regression
//! - Weak learners (decision stumps, linear models)

use crate::models::decision_tree::{DecisionTree, SplitCriterion};
use burn::tensor::{backend::Backend, Tensor, TensorData};
use std::fmt;

/// Weak learner trait for boosting algorithms
pub trait WeakLearner<B: Backend<FloatElem = f32>> {
    /// Fit the weak learner to weighted training data
    fn fit(
        &mut self,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 1>,
        weights: &Tensor<B, 1>,
    ) -> Result<(), String>;

    /// Predict using the weak learner
    fn predict(&self, x: &Tensor<B, 2>) -> Result<Tensor<B, 1>, String>;

    /// Check if the weak learner is fitted
    fn is_fitted(&self) -> bool;

    /// Index of the feature this learner splits on, if it splits on exactly one.
    ///
    /// Used to aggregate feature importances across an ensemble.
    fn feature_used(&self) -> Option<usize> {
        None
    }
}

/// Decision stump: a single feature, a single threshold.
///
/// Fitted by exhaustive search for the split with the lowest *weighted*
/// error, which is what AdaBoost needs — reweighting the samples has to
/// change the stump you get, or every round returns the same learner and
/// boosting stops after one.
#[derive(Debug, Clone)]
pub struct DecisionStump<B: Backend<FloatElem = f32>> {
    feature_idx: usize,
    threshold: f32,
    /// Label predicted for `value <= threshold`.
    below_label: f32,
    /// Label predicted for `value > threshold`.
    above_label: f32,
    fitted: bool,
    device: burn::tensor::Device<B>,
}

impl<B: Backend<FloatElem = f32>> DecisionStump<B> {
    /// Create a new decision stump
    pub fn new(device: burn::tensor::Device<B>) -> Self {
        Self {
            feature_idx: 0,
            threshold: 0.0,
            below_label: 0.0,
            above_label: 0.0,
            fitted: false,
            device,
        }
    }

    /// The feature this stump splits on, and where.
    pub fn split(&self) -> (usize, f32) {
        (self.feature_idx, self.threshold)
    }
}

impl<B: Backend<FloatElem = f32>> WeakLearner<B> for DecisionStump<B> {
    fn fit(
        &mut self,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 1>,
        weights: &Tensor<B, 1>,
    ) -> Result<(), String> {
        let [n_samples, n_features] = x.dims();
        let x_data = x
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|_| "Failed to convert features to vector")?;
        let y_data = y
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|_| "Failed to convert labels to vector")?;
        let w_data = weights
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|_| "Failed to convert weights to vector")?;

        if n_samples == 0 {
            return Err("No training samples provided".to_string());
        }

        let mut classes = y_data.clone();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        classes.dedup();

        let total_weight: f32 = w_data.iter().sum();
        let mut best: Option<(f32, usize, f32, f32, f32)> = None;

        for feature in 0..n_features {
            let mut values: Vec<f32> = (0..n_samples)
                .map(|i| x_data[i * n_features + feature])
                .collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values.dedup();

            // Midpoints between adjacent distinct values, plus one split that
            // sends everything to the "above" side.
            let mut thresholds: Vec<f32> = values.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();
            if thresholds.is_empty() {
                thresholds.push(values.first().copied().unwrap_or(0.0) - 1.0);
            }

            for &threshold in &thresholds {
                // Weighted class mass on each side of the split.
                let mut below = vec![0.0f32; classes.len()];
                let mut above = vec![0.0f32; classes.len()];
                for i in 0..n_samples {
                    let class = classes
                        .iter()
                        .position(|c| (c - y_data[i]).abs() < 1e-6)
                        .unwrap_or(0);
                    if x_data[i * n_features + feature] <= threshold {
                        below[class] += w_data[i];
                    } else {
                        above[class] += w_data[i];
                    }
                }

                let pick = |mass: &[f32]| {
                    let mut best_class = 0;
                    for (i, &m) in mass.iter().enumerate() {
                        if m > mass[best_class] {
                            best_class = i;
                        }
                    }
                    (classes[best_class], mass[best_class])
                };
                let (below_label, below_correct) = pick(&below);
                let (above_label, above_correct) = pick(&above);

                // Whatever the majority side does not cover is error.
                let error = (total_weight - below_correct - above_correct).max(0.0);
                if best.map_or(true, |(b, ..)| error < b) {
                    best = Some((error, feature, threshold, below_label, above_label));
                }
            }
        }

        let (_, feature, threshold, below_label, above_label) =
            best.ok_or("No usable split found")?;
        self.feature_idx = feature;
        self.threshold = threshold;
        self.below_label = below_label;
        self.above_label = above_label;
        self.fitted = true;
        Ok(())
    }

    fn predict(&self, x: &Tensor<B, 2>) -> Result<Tensor<B, 1>, String> {
        if !self.fitted {
            return Err("Stump must be fitted before prediction".to_string());
        }

        let [n_samples, n_features] = x.dims();
        let x_data = x
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|_| "Failed to convert features to vector")?;

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                if x_data[i * n_features + self.feature_idx] <= self.threshold {
                    self.below_label
                } else {
                    self.above_label
                }
            })
            .collect();

        Ok(Tensor::from_data(
            TensorData::new(predictions, [n_samples]),
            &self.device,
        ))
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }

    fn feature_used(&self) -> Option<usize> {
        Some(self.feature_idx)
    }
}

/// AdaBoost classifier configuration
#[derive(Debug, Clone)]
pub struct AdaBoostConfig {
    /// Number of weak learners
    pub n_estimators: usize,
    /// Learning rate (step size shrinkage)
    pub learning_rate: f32,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for AdaBoostConfig {
    fn default() -> Self {
        Self {
            n_estimators: 50,
            learning_rate: 1.0,
            random_seed: None,
        }
    }
}

/// AdaBoost classifier
///
/// AdaBoost (Adaptive Boosting) is an ensemble method that combines multiple
/// weak learners by giving more weight to previously misclassified examples.
pub struct AdaBoostClassifier<B: Backend<FloatElem = f32>> {
    /// Configuration
    config: AdaBoostConfig,
    /// Weak learners
    estimators: Vec<Box<dyn WeakLearner<B>>>,
    /// Estimator weights (alpha values)
    estimator_weights: Vec<f32>,
    /// Classes seen during training
    classes: Vec<i32>,
    /// Number of classes
    n_classes: usize,
    /// Number of features seen during training
    n_features: usize,
    /// Whether the model is fitted
    is_fitted: bool,
}

impl<B: Backend<FloatElem = f32>> AdaBoostClassifier<B> {
    /// Create a new AdaBoost classifier
    pub fn new(config: AdaBoostConfig) -> Self {
        Self {
            config,
            estimators: Vec::new(),
            estimator_weights: Vec::new(),
            classes: Vec::new(),
            n_classes: 0,
            n_features: 0,
            is_fitted: false,
        }
    }

    /// Create AdaBoost with default configuration
    pub fn default() -> Self {
        Self::new(AdaBoostConfig::default())
    }

    /// Set number of estimators
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, learning_rate: f32) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Fit the AdaBoost classifier
    ///
    /// # Arguments
    /// * `x` - Training features of shape [n_samples, n_features]
    /// * `y` - Training labels of shape [n_samples]
    ///
    /// # Returns
    /// * `Result<(), String>` - Ok if training succeeded, Err with error message otherwise
    pub fn fit(&mut self, x: &Tensor<B, 2>, y: &Tensor<B, 1>) -> Result<(), String> {
        let n_samples = x.dims()[0];

        if n_samples == 0 {
            return Err("No training samples provided".to_string());
        }

        // Get unique classes
        let y_data = y
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|_| "Failed to convert labels to vector")?;

        let mut unique_classes = y_data.clone();
        unique_classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_classes.dedup();
        self.classes = unique_classes.iter().map(|&x| x as i32).collect();
        self.n_classes = self.classes.len();

        if self.n_classes < 2 {
            return Err("Need at least 2 classes for classification".to_string());
        }

        self.n_features = x.dims()[1];

        // Initialize sample weights uniformly
        let mut sample_weights = vec![1.0 / n_samples as f32; n_samples];

        // Clear previous estimators
        self.estimators.clear();
        self.estimator_weights.clear();

        for t in 0..self.config.n_estimators {
            // Create and fit weak learner
            let mut weak_learner = Box::new(DecisionStump::<B>::new(x.device().clone()));
            let weights_tensor = Tensor::from_floats(
                TensorData::new(sample_weights.clone(), [n_samples]),
                &x.device(),
            );

            weak_learner.fit(x, y, &weights_tensor)?;

            // Get predictions
            let predictions = weak_learner.predict(x)?;
            let pred_data = predictions
                .to_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .map_err(|_| "Failed to convert predictions to vector")?;

            // Calculate weighted error
            let mut weighted_error = 0.0;
            let mut total_weight = 0.0;

            for i in 0..n_samples {
                total_weight += sample_weights[i];
                if (pred_data[i] - y_data[i]).abs() > 1e-6 {
                    // Misclassification
                    weighted_error += sample_weights[i];
                }
            }

            weighted_error /= total_weight;

            // Stop if perfect classifier or worse than random
            if weighted_error <= 0.0 {
                self.estimator_weights.push(1.0);
                self.estimators.push(weak_learner);
                break;
            }

            if weighted_error >= 0.5 {
                if t == 0 {
                    return Err(
                        "First weak learner performs worse than random guessing".to_string()
                    );
                }
                break;
            }

            // Calculate alpha (estimator weight)
            let alpha =
                self.config.learning_rate * ((1.0 - weighted_error) / weighted_error).ln() * 0.5;

            self.estimator_weights.push(alpha);
            self.estimators.push(weak_learner);

            // Update sample weights
            let exp_alpha = alpha.exp();
            let exp_neg_alpha = (-alpha).exp();

            for i in 0..n_samples {
                if (pred_data[i] - y_data[i]).abs() > 1e-6 {
                    // Misclassification
                    sample_weights[i] *= exp_alpha;
                } else {
                    // Correct classification
                    sample_weights[i] *= exp_neg_alpha;
                }
            }

            // Normalize weights
            let weight_sum: f32 = sample_weights.iter().sum();
            if weight_sum > 0.0 {
                for weight in &mut sample_weights {
                    *weight /= weight_sum;
                }
            }
        }

        if self.estimators.is_empty() {
            return Err("No weak learners were successfully trained".to_string());
        }

        self.is_fitted = true;
        Ok(())
    }

    /// Predict class labels
    ///
    /// # Arguments
    /// * `x` - Input features of shape [n_samples, n_features]
    ///
    /// # Returns
    /// * `Result<Tensor<B, 1>, String>` - Predicted class labels
    pub fn predict(&self, x: &Tensor<B, 2>) -> Result<Tensor<B, 1>, String> {
        if !self.is_fitted {
            return Err("Model must be fitted before prediction".to_string());
        }

        let decision_scores = self.decision_function(x)?;

        // For binary classification, predict based on sign
        if self.n_classes == 2 {
            let predictions = decision_scores.clone().sign();
            // Convert -1, 1 to class labels
            let positive_class = self.classes[1] as f32;
            let negative_class = self.classes[0] as f32;
            let predictions = predictions.clone() * (positive_class - negative_class) / 2.0
                + (positive_class + negative_class) / 2.0;
            Ok(predictions)
        } else {
            // Multi-class: return the class with highest score
            Ok(decision_scores)
        }
    }

    /// Compute decision function scores
    ///
    /// # Arguments
    /// * `x` - Input features of shape [n_samples, n_features]
    ///
    /// # Returns
    /// * `Result<Tensor<B, 1>, String>` - Decision scores
    pub fn decision_function(&self, x: &Tensor<B, 2>) -> Result<Tensor<B, 1>, String> {
        if !self.is_fitted {
            return Err("Model must be fitted before prediction".to_string());
        }

        let n_samples = x.dims()[0];
        let mut scores = vec![0.0; n_samples];

        // Accumulate weighted predictions
        for (estimator, &weight) in self.estimators.iter().zip(self.estimator_weights.iter()) {
            let predictions = estimator.predict(x)?;
            let pred_data = predictions
                .to_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .map_err(|_| "Failed to convert predictions to vector")?;

            for i in 0..n_samples {
                // Convert predictions to -1/+1 for binary classification
                let pred_binary = if self.n_classes == 2 {
                    if pred_data[i] == self.classes[1] as f32 {
                        1.0
                    } else {
                        -1.0
                    }
                } else {
                    pred_data[i]
                };
                scores[i] += weight * pred_binary;
            }
        }

        Ok(Tensor::from_floats(
            TensorData::new(scores, [n_samples]),
            &x.device(),
        ))
    }

    /// Predict class probabilities (for binary classification)
    ///
    /// # Arguments
    /// * `x` - Input features of shape [n_samples, n_features]
    ///
    /// # Returns
    /// * `Result<Tensor<B, 2>, String>` - Class probabilities of shape [n_samples, n_classes]
    pub fn predict_proba(&self, x: &Tensor<B, 2>) -> Result<Tensor<B, 2>, String> {
        if !self.is_fitted {
            return Err("Model must be fitted before prediction".to_string());
        }

        let decision_scores = self.decision_function(x)?;
        let n_samples = x.dims()[0];

        if self.n_classes == 2 {
            // For binary classification, convert decision scores to probabilities using sigmoid
            let scores_2d: Tensor<B, 2> = decision_scores.unsqueeze_dim::<2>(1);
            let positive_probs = (scores_2d.clone().neg().exp() + 1.0).recip(); // Sigmoid
            let negative_probs = positive_probs.clone().neg() + 1.0;

            // Concatenate to get [n_samples, 2]
            let mut proba_data = Vec::with_capacity(n_samples * 2);
            let pos_data = positive_probs
                .squeeze::<1>()
                .to_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .map_err(|_| "Failed to convert probabilities to vector")?;
            let neg_data = negative_probs
                .squeeze::<1>()
                .to_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .map_err(|_| "Failed to convert probabilities to vector")?;

            for i in 0..n_samples {
                proba_data.push(neg_data[i]);
                proba_data.push(pos_data[i]);
            }

            Ok(Tensor::from_floats(
                TensorData::new(proba_data, [n_samples, 2]),
                &x.device(),
            ))
        } else {
            // Multi-class not fully implemented for this example
            Err("Multi-class probability prediction not implemented".to_string())
        }
    }

    /// Get feature importances
    ///
    /// # Returns
    /// * `Option<Vec<f32>>` - Feature importances if available
    pub fn feature_importances(&self) -> Option<Vec<f32>> {
        if !self.is_fitted || self.estimators.is_empty() {
            return None;
        }

        // Total |alpha| attributed to each feature, normalized.
        let mut importances = vec![0.0f32; self.n_features];
        for (estimator, weight) in self.estimators.iter().zip(self.estimator_weights.iter()) {
            if let Some(feature) = estimator.feature_used() {
                if feature < importances.len() {
                    importances[feature] += weight.abs();
                }
            }
        }

        let total: f32 = importances.iter().sum();
        if total > 0.0 {
            for importance in &mut importances {
                *importance /= total;
            }
        }
        Some(importances)
    }

    /// Check if the model is fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Get the number of estimators actually used
    pub fn n_estimators_used(&self) -> usize {
        self.estimators.len()
    }
}

/// Gradient Boosting configuration
#[derive(Debug, Clone)]
pub struct GradientBoostingConfig {
    /// Number of boosting stages
    pub n_estimators: usize,
    /// Learning rate (step size shrinkage)
    pub learning_rate: f32,
    /// Maximum depth of weak learners
    pub max_depth: usize,
    /// Minimum samples required to split a node
    pub min_samples_split: usize,
    /// Minimum samples required at a leaf node
    pub min_samples_leaf: usize,
    /// Fraction of samples used for fitting weak learners
    pub subsample: f32,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for GradientBoostingConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: 3,
            min_samples_split: 2,
            min_samples_leaf: 1,
            subsample: 1.0,
            random_seed: None,
        }
    }
}

/// Gradient Boosting classifier
///
/// Gradient Boosting builds models sequentially, where each model corrects
/// the errors of the previous models by fitting to the residuals.
pub struct GradientBoostingClassifier<B: Backend<FloatElem = f32>> {
    /// Configuration
    config: GradientBoostingConfig,
    /// Weak learners (trees)
    estimators: Vec<DecisionTree<B>>,
    /// Initial prediction (prior)
    init_prediction: f32,
    /// Classes seen during training
    classes: Vec<i32>,
    /// Number of classes
    n_classes: usize,
    /// Whether the model is fitted
    is_fitted: bool,
}

impl<B: Backend<FloatElem = f32>> GradientBoostingClassifier<B> {
    /// Create a new Gradient Boosting classifier
    pub fn new(config: GradientBoostingConfig) -> Self {
        Self {
            config,
            estimators: Vec::new(),
            init_prediction: 0.0,
            classes: Vec::new(),
            n_classes: 0,
            is_fitted: false,
        }
    }

    /// Create Gradient Boosting with default configuration
    pub fn default() -> Self {
        Self::new(GradientBoostingConfig::default())
    }

    /// Set number of estimators
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, learning_rate: f32) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Set maximum depth
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.config.max_depth = max_depth;
        self
    }

    /// Fit the Gradient Boosting classifier
    ///
    /// # Arguments
    /// * `x` - Training features of shape [n_samples, n_features]
    /// * `y` - Training labels of shape [n_samples]
    ///
    /// # Returns
    /// * `Result<(), String>` - Ok if training succeeded, Err with error message otherwise
    pub fn fit(&mut self, x: &Tensor<B, 2>, y: &Tensor<B, 1>) -> Result<(), String> {
        let n_samples = x.dims()[0];

        if n_samples == 0 {
            return Err("No training samples provided".to_string());
        }

        // Get unique classes
        let y_data = y
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|_| "Failed to convert labels to vector")?;

        let mut unique_classes = y_data.clone();
        unique_classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_classes.dedup();
        self.classes = unique_classes.iter().map(|&x| x as i32).collect();
        self.n_classes = self.classes.len();

        if self.n_classes < 2 {
            return Err("Need at least 2 classes for classification".to_string());
        }

        // For binary classification, convert to -1/+1
        let y_binary = if self.n_classes == 2 {
            y_data
                .iter()
                .map(|&label| {
                    if label == self.classes[1] as f32 {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .collect::<Vec<f32>>()
        } else {
            y_data.clone()
        };

        // Initialize prediction (log odds for binary classification)
        self.init_prediction = if self.n_classes == 2 {
            let positive_count = y_binary.iter().filter(|&&y| y > 0.0).count() as f32;
            let negative_count = n_samples as f32 - positive_count;
            if negative_count > 0.0 {
                (positive_count / negative_count).ln()
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Initialize predictions with the initial value
        let mut f_values = vec![self.init_prediction; n_samples];

        // Clear previous estimators
        self.estimators.clear();

        // Gradient boosting iterations
        for _m in 0..self.config.n_estimators {
            // Compute pseudo-residuals (negative gradients)
            let residuals = if self.n_classes == 2 {
                // For binary classification with logistic loss
                let mut residuals = Vec::with_capacity(n_samples);
                for i in 0..n_samples {
                    let prob = 1.0 / (1.0 + (-f_values[i]).exp()); // Sigmoid
                    let residual = y_binary[i] - 2.0 * prob + 1.0; // Gradient of binomial deviance
                    residuals.push(residual);
                }
                residuals
            } else {
                // Multi-class not fully implemented
                return Err("Multi-class gradient boosting not fully implemented".to_string());
            };

            // Create tree to fit residuals
            let mut tree = DecisionTree::regressor(x.device().clone())
                .with_max_depth(self.config.max_depth)
                .with_min_samples_split(self.config.min_samples_split)
                .with_min_samples_leaf(self.config.min_samples_leaf)
                .with_criterion(SplitCriterion::MSE); // Use MSE for regression on residuals

            let residuals_tensor =
                Tensor::from_floats(TensorData::new(residuals, [n_samples]), &x.device());

            // Fit tree to residuals
            tree.fit(x.clone(), residuals_tensor)?;

            // Get tree predictions
            let tree_predictions = tree.predict(x.clone())?;
            let tree_pred_data = tree_predictions
                .to_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .map_err(|_| "Failed to convert tree predictions to vector")?;

            // Update predictions with learning rate
            for i in 0..n_samples {
                f_values[i] += self.config.learning_rate * tree_pred_data[i];
            }

            self.estimators.push(tree);
        }

        self.is_fitted = true;
        Ok(())
    }

    /// Predict class labels
    ///
    /// # Arguments
    /// * `x` - Input features of shape [n_samples, n_features]
    ///
    /// # Returns
    /// * `Result<Tensor<B, 1>, String>` - Predicted class labels
    pub fn predict(&self, x: &Tensor<B, 2>) -> Result<Tensor<B, 1>, String> {
        if !self.is_fitted {
            return Err("Model must be fitted before prediction".to_string());
        }

        let decision_scores = self.decision_function(x)?;

        if self.n_classes == 2 {
            // For binary classification, predict based on sign of decision function
            let predictions = decision_scores.clone().sign();
            // Convert -1, 1 to class labels
            let positive_class = self.classes[1] as f32;
            let negative_class = self.classes[0] as f32;
            let predictions =
                predictions.clone().clamp(-1.0, 1.0) * (positive_class - negative_class) / 2.0
                    + (positive_class + negative_class) / 2.0;
            Ok(predictions)
        } else {
            Ok(decision_scores)
        }
    }

    /// Compute decision function scores
    ///
    /// # Arguments
    /// * `x` - Input features of shape [n_samples, n_features]
    ///
    /// # Returns
    /// * `Result<Tensor<B, 1>, String>` - Decision scores
    pub fn decision_function(&self, x: &Tensor<B, 2>) -> Result<Tensor<B, 1>, String> {
        if !self.is_fitted {
            return Err("Model must be fitted before prediction".to_string());
        }

        let n_samples = x.dims()[0];
        let mut scores = vec![self.init_prediction; n_samples];

        // Sum predictions from all trees
        for tree in &self.estimators {
            let tree_predictions = tree.predict(x.clone())?;
            let tree_pred_data = tree_predictions
                .to_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .map_err(|_| "Failed to convert tree predictions to vector")?;

            for i in 0..n_samples {
                scores[i] += self.config.learning_rate * tree_pred_data[i];
            }
        }

        Ok(Tensor::from_floats(
            TensorData::new(scores, [n_samples]),
            &x.device(),
        ))
    }

    /// Predict class probabilities
    ///
    /// # Arguments
    /// * `x` - Input features of shape [n_samples, n_features]
    ///
    /// # Returns
    /// * `Result<Tensor<B, 2>, String>` - Class probabilities of shape [n_samples, n_classes]
    pub fn predict_proba(&self, x: &Tensor<B, 2>) -> Result<Tensor<B, 2>, String> {
        if !self.is_fitted {
            return Err("Model must be fitted before prediction".to_string());
        }

        if self.n_classes == 2 {
            let decision_scores = self.decision_function(x)?;
            let n_samples = x.dims()[0];

            // Convert decision scores to probabilities using sigmoid
            let positive_probs = (decision_scores.clone().neg().exp() + 1.0).recip();
            let negative_probs = positive_probs.clone().neg() + 1.0;

            // Create probability matrix [n_samples, 2]
            let mut proba_data = Vec::with_capacity(n_samples * 2);
            let pos_data = positive_probs
                .to_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .map_err(|_| "Failed to convert probabilities to vector")?;
            let neg_data = negative_probs
                .to_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .map_err(|_| "Failed to convert probabilities to vector")?;

            for i in 0..n_samples {
                proba_data.push(neg_data[i]);
                proba_data.push(pos_data[i]);
            }

            Ok(Tensor::from_floats(
                TensorData::new(proba_data, [n_samples, 2]),
                &x.device(),
            ))
        } else {
            Err("Multi-class probability prediction not implemented".to_string())
        }
    }

    /// Check if the model is fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Get the number of estimators
    pub fn n_estimators_used(&self) -> usize {
        self.estimators.len()
    }
}

impl fmt::Display for AdaBoostConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AdaBoost(n_estimators={}, learning_rate={})",
            self.n_estimators, self.learning_rate
        )
    }
}

impl fmt::Display for GradientBoostingConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GradientBoosting(n_estimators={}, learning_rate={}, max_depth={})",
            self.n_estimators, self.learning_rate, self.max_depth
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DefaultBackend;
    use burn::tensor::TensorData;

    #[test]
    fn test_decision_stump_creation() {
        let _stump = DecisionStump::<DefaultBackend>::new(Default::default());
    }

    #[test]
    fn test_adaboost_creation() {
        let config = AdaBoostConfig::default();
        let _classifier = AdaBoostClassifier::<DefaultBackend>::new(config);
    }

    #[test]
    fn test_gradient_boosting_creation() {
        let config = GradientBoostingConfig::default();
        let _classifier = GradientBoostingClassifier::<DefaultBackend>::new(config);
    }

    #[test]
    fn test_adaboost_simple_fit() {
        let device = &burn::tensor::Device::<DefaultBackend>::default();

        // Simple binary classification dataset
        let x_data = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
        // 4 samples of 2 features, so 4 labels — this was 8, which cannot be
        // shaped [4].
        let y_data = vec![0.0, 0.0, 1.0, 1.0];

        let x = Tensor::<DefaultBackend, 2>::from_floats(TensorData::new(x_data, [4, 2]), &device);
        let y = Tensor::<DefaultBackend, 1>::from_floats(TensorData::new(y_data, [4]), &device);

        let config = AdaBoostConfig {
            n_estimators: 3,
            learning_rate: 1.0,
            random_seed: Some(42),
        };

        let mut classifier = AdaBoostClassifier::new(config);
        let result = classifier.fit(&x, &y);
        assert!(result.is_ok(), "AdaBoost fit should succeed");
        assert!(classifier.is_fitted());
    }

    #[test]
    fn test_gradient_boosting_simple_fit() {
        let device = &burn::tensor::Device::<DefaultBackend>::default();

        // Simple binary classification dataset
        let x_data = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
        // 4 samples of 2 features, so 4 labels — this was 8, which cannot be
        // shaped [4].
        let y_data = vec![0.0, 0.0, 1.0, 1.0];

        let x = Tensor::<DefaultBackend, 2>::from_floats(TensorData::new(x_data, [4, 2]), &device);
        let y = Tensor::<DefaultBackend, 1>::from_floats(TensorData::new(y_data, [4]), &device);

        let config = GradientBoostingConfig {
            n_estimators: 3,
            learning_rate: 0.1,
            max_depth: 2,
            ..Default::default()
        };

        let mut classifier = GradientBoostingClassifier::new(config);
        let result = classifier.fit(&x, &y);
        assert!(result.is_ok(), "Gradient Boosting fit should succeed");
        assert!(classifier.is_fitted());
    }
}
