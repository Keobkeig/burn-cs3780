//! Online learning algorithms implementation using Burn.
//!
//! This module implements various online learning algorithms:
//! - Online Perceptron
//! - Passive-Aggressive algorithms
//! - Online Gradient Descent variants
//! - Stochastic Gradient Descent for various loss functions

use burn::tensor::{backend::Backend, Tensor, TensorData};
use std::fmt;

/// Online learning algorithm trait
pub trait OnlineLearner<B: Backend<FloatElem = f32>> {
    /// Update the model with a single example
    fn partial_fit(&mut self, x: &Tensor<B, 1>, y: f32) -> Result<(), String>;

    /// Predict a single example, as -1 or +1 to match the training labels
    fn predict_one(&self, x: &Tensor<B, 1>) -> Result<f32, String>;

    /// Raw decision score `w . x + b` for a single example.
    ///
    /// Its sign is the prediction; its magnitude is the margin. Counting
    /// mistakes needs this rather than the thresholded class, so that a
    /// correct negative prediction is not mistaken for a zero score.
    fn decision_score(&self, x: &Tensor<B, 1>) -> Result<f32, String>;

    /// Current bias term, or 0 if this learner does not fit an intercept.
    fn bias(&self) -> f32;

    /// Predict multiple examples
    fn predict(&self, x: &Tensor<B, 2>) -> Result<Tensor<B, 1>, String> {
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let mut predictions = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let sample = x.clone().slice([i..i + 1, 0..n_features]).squeeze::<1>();
            let pred = self.predict_one(&sample)?;
            predictions.push(pred);
        }

        Ok(Tensor::from_floats(
            TensorData::new(predictions, [n_samples]),
            &x.device(),
        ))
    }

    /// Check if the model has been initialized
    fn is_initialized(&self) -> bool;

    /// Get the current weights
    fn get_weights(&self) -> Option<&Tensor<B, 1>>;
}

/// Online Perceptron configuration
#[derive(Debug, Clone)]
pub struct OnlinePerceptronConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Random seed for weight initialization
    pub random_seed: Option<u64>,
}

impl Default for OnlinePerceptronConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1.0,
            fit_intercept: true,
            random_seed: None,
        }
    }
}

/// Online Perceptron classifier
///
/// The online perceptron updates weights incrementally with each new example.
/// It's suitable for large datasets that don't fit in memory.
#[derive(Debug, Clone)]
pub struct OnlinePerceptron<B: Backend<FloatElem = f32>> {
    /// Configuration
    config: OnlinePerceptronConfig,
    /// Weight vector
    weights: Option<Tensor<B, 1>>,
    /// Bias term
    bias: f32,
    /// Number of features
    n_features: Option<usize>,
    /// Number of updates performed
    n_updates: usize,
}

impl<B: Backend<FloatElem = f32>> OnlinePerceptron<B> {
    /// Create a new online perceptron
    pub fn new(config: OnlinePerceptronConfig) -> Self {
        Self {
            config,
            weights: None,
            bias: 0.0,
            n_features: None,
            n_updates: 0,
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(OnlinePerceptronConfig::default())
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, learning_rate: f32) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Initialize weights if not already initialized
    fn initialize_weights(&mut self, n_features: usize, device: &burn::tensor::Device<B>) {
        if self.weights.is_none() {
            self.n_features = Some(n_features);
            // Initialize weights to zeros
            self.weights = Some(Tensor::zeros([n_features], device));
            self.bias = 0.0;
        }
    }

    /// Get decision function value for a single example
    pub fn decision_function_one(&self, x: &Tensor<B, 1>) -> Result<f32, String> {
        if let Some(ref weights) = self.weights {
            // Compute w^T x + b
            let score = weights
                .clone()
                .mul(x.clone())
                .sum()
                .to_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .map_err(|_| "Failed to convert score to scalar")?[0];
            Ok(score + self.bias)
        } else {
            Err("Model not initialized".to_string())
        }
    }

    /// Get number of updates performed
    pub fn n_updates(&self) -> usize {
        self.n_updates
    }
}

impl<B: Backend<FloatElem = f32>> OnlineLearner<B> for OnlinePerceptron<B> {
    fn partial_fit(&mut self, x: &Tensor<B, 1>, y: f32) -> Result<(), String> {
        let n_features = x.dims()[0];

        // Initialize weights if needed
        if self.weights.is_none() {
            self.initialize_weights(n_features, &x.device());
        }

        // Check feature dimension consistency
        if let Some(expected_features) = self.n_features {
            if n_features != expected_features {
                return Err(format!(
                    "Feature dimension mismatch: expected {}, got {}",
                    expected_features, n_features
                ));
            }
        }

        // Get current prediction
        let prediction = self.predict_one(x)?;

        // Convert labels to -1/+1 for perceptron
        let y_binary = if y > 0.5 { 1.0 } else { -1.0 };
        let pred_binary = if prediction > 0.0 { 1.0 } else { -1.0 };

        // Update weights if prediction is wrong
        let diff: f32 = y_binary - pred_binary;
        if diff.abs() > 1e-6 {
            if let Some(ref mut weights) = self.weights {
                // w = w + η * y * x
                let update = x.clone() * (self.config.learning_rate * y_binary);
                *weights = weights.clone() + update;

                // Update bias if enabled
                if self.config.fit_intercept {
                    self.bias += self.config.learning_rate * y_binary;
                }
            }
        }

        self.n_updates += 1;
        Ok(())
    }

    fn predict_one(&self, x: &Tensor<B, 1>) -> Result<f32, String> {
        let decision = self.decision_function_one(x)?;
        Ok(if decision > 0.0 { 1.0 } else { -1.0 })
    }

    fn decision_score(&self, x: &Tensor<B, 1>) -> Result<f32, String> {
        self.decision_function_one(x)
    }

    fn bias(&self) -> f32 {
        self.bias
    }

    fn is_initialized(&self) -> bool {
        self.weights.is_some()
    }

    fn get_weights(&self) -> Option<&Tensor<B, 1>> {
        self.weights.as_ref()
    }
}

/// Passive-Aggressive algorithm configuration
#[derive(Debug, Clone)]
pub struct PassiveAggressiveConfig {
    /// Regularization parameter (C)
    pub c: f32,
    /// Loss function: "hinge" or "squared_hinge"
    pub loss: String,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Random seed for weight initialization
    pub random_seed: Option<u64>,
}

impl Default for PassiveAggressiveConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            loss: "hinge".to_string(),
            fit_intercept: true,
            random_seed: None,
        }
    }
}

/// Passive-Aggressive classifier
///
/// Passive-Aggressive algorithms are online learning algorithms that are
/// passive for correct classifications and aggressive for incorrect ones.
#[derive(Debug, Clone)]
pub struct PassiveAggressive<B: Backend<FloatElem = f32>> {
    /// Configuration
    config: PassiveAggressiveConfig,
    /// Weight vector
    weights: Option<Tensor<B, 1>>,
    /// Bias term
    bias: f32,
    /// Number of features
    n_features: Option<usize>,
    /// Number of updates performed
    n_updates: usize,
}

impl<B: Backend<FloatElem = f32>> PassiveAggressive<B> {
    /// Create a new Passive-Aggressive classifier
    pub fn new(config: PassiveAggressiveConfig) -> Self {
        Self {
            config,
            weights: None,
            bias: 0.0,
            n_features: None,
            n_updates: 0,
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(PassiveAggressiveConfig::default())
    }

    /// Set regularization parameter
    pub fn with_c(mut self, c: f32) -> Self {
        self.config.c = c;
        self
    }

    /// Set loss function
    pub fn with_loss(mut self, loss: &str) -> Self {
        self.config.loss = loss.to_string();
        self
    }

    /// Initialize weights if not already initialized
    fn initialize_weights(&mut self, n_features: usize, device: &burn::tensor::Device<B>) {
        if self.weights.is_none() {
            self.n_features = Some(n_features);
            // Initialize weights to zeros
            self.weights = Some(Tensor::zeros([n_features], device));
            self.bias = 0.0;
        }
    }

    /// Compute hinge loss
    fn hinge_loss(&self, prediction: f32, y: f32) -> f32 {
        let margin = y * prediction;
        if margin >= 1.0 {
            0.0
        } else {
            1.0 - margin
        }
    }

    /// Compute squared hinge loss
    fn squared_hinge_loss(&self, prediction: f32, y: f32) -> f32 {
        let hinge = self.hinge_loss(prediction, y);
        hinge * hinge
    }

    /// Get decision function value for a single example
    pub fn decision_function_one(&self, x: &Tensor<B, 1>) -> Result<f32, String> {
        if let Some(ref weights) = self.weights {
            // Compute w^T x + b
            let score = weights
                .clone()
                .mul(x.clone())
                .sum()
                .to_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .map_err(|_| "Failed to convert score to scalar")?[0];
            Ok(score + self.bias)
        } else {
            Err("Model not initialized".to_string())
        }
    }

    /// Get number of updates performed
    pub fn n_updates(&self) -> usize {
        self.n_updates
    }
}

impl<B: Backend<FloatElem = f32>> OnlineLearner<B> for PassiveAggressive<B> {
    fn partial_fit(&mut self, x: &Tensor<B, 1>, y: f32) -> Result<(), String> {
        let n_features = x.dims()[0];

        // Initialize weights if needed
        if self.weights.is_none() {
            self.initialize_weights(n_features, &x.device());
        }

        // Check feature dimension consistency
        if let Some(expected_features) = self.n_features {
            if n_features != expected_features {
                return Err(format!(
                    "Feature dimension mismatch: expected {}, got {}",
                    expected_features, n_features
                ));
            }
        }

        // Convert labels to -1/+1
        let y_binary = if y > 0.5 { 1.0 } else { -1.0 };

        // Get current prediction
        let prediction = self.decision_function_one(x)?;

        // Compute loss
        let loss = match self.config.loss.as_str() {
            "hinge" => self.hinge_loss(prediction, y_binary),
            "squared_hinge" => self.squared_hinge_loss(prediction, y_binary),
            _ => return Err(format!("Unknown loss function: {}", self.config.loss)),
        };

        // Update weights if there's a loss
        if loss > 0.0 {
            if let Some(ref mut weights) = self.weights {
                // Compute norm of x
                let x_norm_sq = x
                    .clone()
                    .powf_scalar(2.0)
                    .sum()
                    .to_data()
                    .convert::<f32>()
                    .to_vec::<f32>()
                    .map_err(|_| "Failed to convert norm to scalar")?[0];

                // Compute learning rate (tau)
                let tau = match self.config.loss.as_str() {
                    "hinge" => {
                        let tau_uncapped = loss / x_norm_sq.max(1e-8);
                        tau_uncapped.min(self.config.c)
                    }
                    "squared_hinge" => {
                        let tau_uncapped =
                            loss / (x_norm_sq + 1.0 / (2.0 * self.config.c)).max(1e-8);
                        tau_uncapped
                    }
                    _ => return Err(format!("Unknown loss function: {}", self.config.loss)),
                };

                // Update weights: w = w + τ * y * x
                let update = x.clone() * (tau * y_binary);
                *weights = weights.clone() + update;

                // Update bias if enabled
                if self.config.fit_intercept {
                    self.bias += tau * y_binary;
                }
            }
        }

        self.n_updates += 1;
        Ok(())
    }

    fn predict_one(&self, x: &Tensor<B, 1>) -> Result<f32, String> {
        let decision = self.decision_function_one(x)?;
        Ok(if decision > 0.0 { 1.0 } else { -1.0 })
    }

    fn decision_score(&self, x: &Tensor<B, 1>) -> Result<f32, String> {
        self.decision_function_one(x)
    }

    fn bias(&self) -> f32 {
        self.bias
    }

    fn is_initialized(&self) -> bool {
        self.weights.is_some()
    }

    fn get_weights(&self) -> Option<&Tensor<B, 1>> {
        self.weights.as_ref()
    }
}

/// Online SGD configuration
#[derive(Debug, Clone)]
pub struct OnlineSGDConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Learning rate schedule: "constant", "optimal", "invscaling"
    pub learning_rate_schedule: String,
    /// Power for inverse scaling schedule
    pub power_t: f32,
    /// Loss function: "hinge", "log", "squared_loss", "huber"
    pub loss: String,
    /// L2 regularization parameter
    pub alpha: f32,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Random seed
    pub random_seed: Option<u64>,
}

impl Default for OnlineSGDConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            learning_rate_schedule: "invscaling".to_string(),
            power_t: 0.5,
            loss: "hinge".to_string(),
            alpha: 0.0001,
            fit_intercept: true,
            random_seed: None,
        }
    }
}

/// Online Stochastic Gradient Descent classifier
///
/// SGD with various loss functions and learning rate schedules
#[derive(Debug, Clone)]
pub struct OnlineSGD<B: Backend<FloatElem = f32>> {
    /// Configuration
    config: OnlineSGDConfig,
    /// Weight vector
    weights: Option<Tensor<B, 1>>,
    /// Bias term
    bias: f32,
    /// Number of features
    n_features: Option<usize>,
    /// Number of updates performed
    n_updates: usize,
    /// Current learning rate
    current_lr: f32,
}

impl<B: Backend<FloatElem = f32>> OnlineSGD<B> {
    /// Create a new Online SGD classifier
    pub fn new(config: OnlineSGDConfig) -> Self {
        Self {
            current_lr: config.learning_rate,
            config,
            weights: None,
            bias: 0.0,
            n_features: None,
            n_updates: 0,
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(OnlineSGDConfig::default())
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, learning_rate: f32) -> Self {
        self.config.learning_rate = learning_rate;
        self.current_lr = learning_rate;
        self
    }

    /// Set loss function
    pub fn with_loss(mut self, loss: &str) -> Self {
        self.config.loss = loss.to_string();
        self
    }

    /// Set regularization parameter
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Initialize weights if not already initialized
    fn initialize_weights(&mut self, n_features: usize, device: &burn::tensor::Device<B>) {
        if self.weights.is_none() {
            self.n_features = Some(n_features);
            // Initialize weights to small random values
            self.weights = Some(Tensor::random(
                [n_features],
                burn::tensor::Distribution::Normal(0.0, 0.01),
                device,
            ));
            self.bias = 0.0;
        }
    }

    /// Update learning rate according to schedule
    fn update_learning_rate(&mut self) {
        match self.config.learning_rate_schedule.as_str() {
            "constant" => {
                // Learning rate stays the same
            }
            "optimal" => {
                // Optimal schedule for SVM
                self.current_lr = 1.0 / (self.config.alpha * (self.n_updates + 1) as f32);
            }
            "invscaling" => {
                // Inverse scaling
                self.current_lr = self.config.learning_rate
                    / ((self.n_updates + 1) as f32).powf(self.config.power_t);
            }
            _ => {
                // Default to constant
            }
        }
    }

    /// Compute loss and gradient
    fn compute_loss_gradient(&self, prediction: f32, y: f32) -> (f32, f32) {
        match self.config.loss.as_str() {
            "hinge" => {
                // Hinge loss for SVM
                let margin = y * prediction;
                if margin >= 1.0 {
                    (0.0, 0.0) // No loss, no gradient
                } else {
                    (1.0 - margin, -y) // Loss and gradient w.r.t. prediction
                }
            }
            "log" => {
                // Logistic loss
                let exp_term = (-y * prediction).exp();
                let loss = (1.0 + exp_term).ln();
                let gradient = -y * exp_term / (1.0 + exp_term);
                (loss, gradient)
            }
            "squared_loss" => {
                // Squared loss for regression
                let diff = prediction - y;
                (0.5 * diff * diff, diff)
            }
            "huber" => {
                // Huber loss (robust regression)
                let delta = 1.0; // Huber parameter
                let diff = (prediction - y).abs();
                if diff <= delta {
                    (0.5 * diff * diff, prediction - y)
                } else {
                    let loss = delta * diff - 0.5 * delta * delta;
                    let gradient = delta * (prediction - y).signum();
                    (loss, gradient)
                }
            }
            _ => (0.0, 0.0), // Unknown loss
        }
    }

    /// Get decision function value for a single example
    pub fn decision_function_one(&self, x: &Tensor<B, 1>) -> Result<f32, String> {
        if let Some(ref weights) = self.weights {
            // Compute w^T x + b
            let score = weights
                .clone()
                .mul(x.clone())
                .sum()
                .to_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .map_err(|_| "Failed to convert score to scalar")?[0];
            Ok(score + self.bias)
        } else {
            Err("Model not initialized".to_string())
        }
    }

    /// Get number of updates performed
    pub fn n_updates(&self) -> usize {
        self.n_updates
    }

    /// Get current learning rate
    pub fn current_learning_rate(&self) -> f32 {
        self.current_lr
    }
}

impl<B: Backend<FloatElem = f32>> OnlineLearner<B> for OnlineSGD<B> {
    fn partial_fit(&mut self, x: &Tensor<B, 1>, y: f32) -> Result<(), String> {
        let n_features = x.dims()[0];

        // Initialize weights if needed
        if self.weights.is_none() {
            self.initialize_weights(n_features, &x.device());
        }

        // Check feature dimension consistency
        if let Some(expected_features) = self.n_features {
            if n_features != expected_features {
                return Err(format!(
                    "Feature dimension mismatch: expected {}, got {}",
                    expected_features, n_features
                ));
            }
        }

        // Convert labels for classification (-1/+1) or keep as is for regression
        let y_target = match self.config.loss.as_str() {
            "hinge" | "log" => {
                if y > 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            _ => y, // Keep original for regression
        };

        // Get current prediction
        let prediction = self.decision_function_one(x)?;

        // Compute loss and gradient
        let (_loss, gradient) = self.compute_loss_gradient(prediction, y_target);

        // Update learning rate
        self.update_learning_rate();

        // Update weights
        if let Some(ref mut weights) = self.weights {
            // Gradient step: w = w - η * (gradient * x + α * w)
            let weight_gradient = x.clone() * gradient + weights.clone() * self.config.alpha;
            let update = weight_gradient * self.current_lr;
            *weights = weights.clone() - update;

            // Update bias if enabled
            if self.config.fit_intercept {
                self.bias -= self.current_lr * gradient;
            }
        }

        self.n_updates += 1;
        Ok(())
    }

    fn predict_one(&self, x: &Tensor<B, 1>) -> Result<f32, String> {
        let decision = self.decision_function_one(x)?;

        // Return prediction based on loss function
        match self.config.loss.as_str() {
            "hinge" => Ok(if decision > 0.0 { 1.0 } else { -1.0 }),
            "log" => {
                // Logistic prediction
                let prob = 1.0 / (1.0 + (-decision).exp());
                Ok(if prob > 0.5 { 1.0 } else { -1.0 })
            }
            "squared_loss" | "huber" => Ok(decision), // Regression: return raw prediction
            _ => Ok(if decision > 0.0 { 1.0 } else { -1.0 }),
        }
    }

    fn decision_score(&self, x: &Tensor<B, 1>) -> Result<f32, String> {
        self.decision_function_one(x)
    }

    fn bias(&self) -> f32 {
        self.bias
    }

    fn is_initialized(&self) -> bool {
        self.weights.is_some()
    }

    fn get_weights(&self) -> Option<&Tensor<B, 1>> {
        self.weights.as_ref()
    }
}

impl fmt::Display for OnlinePerceptronConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OnlinePerceptron(learning_rate={}, fit_intercept={})",
            self.learning_rate, self.fit_intercept
        )
    }
}

impl fmt::Display for PassiveAggressiveConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PassiveAggressive(C={}, loss={})", self.c, self.loss)
    }
}

impl fmt::Display for OnlineSGDConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OnlineSGD(learning_rate={}, loss={}, alpha={})",
            self.learning_rate, self.loss, self.alpha
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DefaultBackend;
    use burn::tensor::TensorData;

    #[test]
    fn test_online_perceptron_creation() {
        let config = OnlinePerceptronConfig::default();
        let _perceptron = OnlinePerceptron::<DefaultBackend>::new(config);
    }

    #[test]
    fn test_passive_aggressive_creation() {
        let config = PassiveAggressiveConfig::default();
        let _pa = PassiveAggressive::<DefaultBackend>::new(config);
    }

    #[test]
    fn test_online_sgd_creation() {
        let config = OnlineSGDConfig::default();
        let _sgd = OnlineSGD::<DefaultBackend>::new(config);
    }

    #[test]
    fn test_online_perceptron_partial_fit() {
        let device = &burn::tensor::Device::<DefaultBackend>::default();
        let config = OnlinePerceptronConfig::default();
        let mut perceptron = OnlinePerceptron::<DefaultBackend>::new(config);

        // Single training example
        let x = Tensor::from_data(TensorData::new(vec![1.0, 2.0], [2]), device);
        let y = 1.0;

        let result = perceptron.partial_fit(&x, y);
        assert!(result.is_ok(), "Partial fit should succeed");
        assert!(perceptron.is_initialized());
        assert_eq!(perceptron.n_updates(), 1);
    }

    #[test]
    fn test_online_perceptron_predict() {
        let device = &burn::tensor::Device::<DefaultBackend>::default();
        let config = OnlinePerceptronConfig::default();
        let mut perceptron = OnlinePerceptron::<DefaultBackend>::new(config);

        // Train with a few examples
        let x1 = Tensor::from_data(TensorData::new(vec![1.0, 1.0], [2]), device);
        let x2 = Tensor::from_data(TensorData::new(vec![-1.0, -1.0], [2]), device);

        let _ = perceptron.partial_fit(&x1, 1.0);
        let _ = perceptron.partial_fit(&x2, 0.0);

        // Test prediction
        let pred = perceptron.predict_one(&x1);
        assert!(pred.is_ok(), "Prediction should succeed");
    }

    #[test]
    fn test_passive_aggressive_partial_fit() {
        let device = &burn::tensor::Device::<DefaultBackend>::default();
        let config = PassiveAggressiveConfig::default();
        let mut pa = PassiveAggressive::<DefaultBackend>::new(config);

        let x = Tensor::from_data(TensorData::new(vec![1.0, 2.0], [2]), device);
        let y = 1.0;

        let result = pa.partial_fit(&x, y);
        assert!(result.is_ok(), "Partial fit should succeed");
        assert!(pa.is_initialized());
    }

    #[test]
    fn test_online_sgd_partial_fit() {
        let device = &burn::tensor::Device::<DefaultBackend>::default();
        let config = OnlineSGDConfig::default();
        let mut sgd = OnlineSGD::<DefaultBackend>::new(config);

        let x = Tensor::from_data(TensorData::new(vec![1.0, 2.0], [2]), device);
        let y = 1.0;

        let result = sgd.partial_fit(&x, y);
        assert!(result.is_ok(), "Partial fit should succeed");
        assert!(sgd.is_initialized());
    }

    #[test]
    fn test_online_sgd_learning_rate_update() {
        let device = &burn::tensor::Device::<DefaultBackend>::default();
        let config = OnlineSGDConfig {
            learning_rate_schedule: "invscaling".to_string(),
            ..Default::default()
        };
        let mut sgd = OnlineSGD::<DefaultBackend>::new(config);

        let x = Tensor::from_data(TensorData::new(vec![1.0, 2.0], [2]), device);
        let initial_lr = sgd.current_learning_rate();

        // Perform several updates
        for i in 0..5 {
            let _ = sgd.partial_fit(&x, if i % 2 == 0 { 1.0 } else { 0.0 });
        }

        let final_lr = sgd.current_learning_rate();
        assert!(
            final_lr < initial_lr,
            "Learning rate should decrease with invscaling"
        );
    }
}
