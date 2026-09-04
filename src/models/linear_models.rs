//! Linear models: Linear and Logistic Regression using Burn.
//!
//! This module implements linear regression and logistic regression with various
//! regularization techniques (Ridge, Lasso, Elastic Net).

use crate::optimizers::{Adam, Optimizer, SGD};
use burn::tensor::{backend::Backend, Tensor, TensorData};
use std::marker::PhantomData;

/// Linear Regression model
#[derive(Debug, Clone)]
pub struct LinearRegression<B: Backend<FloatElem = f32>> {
    weights: Option<Tensor<B, 1>>,
    fit_intercept: bool,
    regularization: Regularization,
    solver: Solver,
    _phantom: PhantomData<B>,
}

/// Logistic Regression model
#[derive(Debug, Clone)]
pub struct LogisticRegression<B: Backend<FloatElem = f32>> {
    weights: Option<Tensor<B, 1>>,
    fit_intercept: bool,
    regularization: Regularization,
    max_iter: usize,
    tolerance: f32,
    learning_rate: f32,
    solver: Solver,
    _phantom: PhantomData<B>,
}

/// Regularization types
#[derive(Debug, Clone)]
pub enum Regularization {
    /// No regularization
    None,
    /// Ridge (L2) regularization
    Ridge {
        /// Regularization strength
        alpha: f32,
    },
    /// Lasso (L1) regularization
    Lasso {
        /// Regularization strength
        alpha: f32,
    },
    /// Elastic Net (L1 + L2) regularization
    ElasticNet {
        /// Regularization strength
        alpha: f32,
        /// L1 ratio (0 = Ridge, 1 = Lasso)
        l1_ratio: f32,
    },
}

/// Solver types
#[derive(Debug, Clone)]
pub enum Solver {
    /// Normal equation (for linear regression)
    Normal,
    /// Stochastic Gradient Descent
    SGD,
    /// Adam optimizer
    Adam,
}

impl<B: Backend<FloatElem = f32>> LinearRegression<B> {
    /// Create a new Linear Regression model
    pub fn new() -> Self {
        Self {
            weights: None,
            fit_intercept: true,
            regularization: Regularization::None,
            solver: Solver::Normal,
            _phantom: PhantomData,
        }
    }

    /// Set whether to fit intercept
    pub fn with_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set regularization
    pub fn with_regularization(mut self, regularization: Regularization) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set solver
    pub fn with_solver(mut self, solver: Solver) -> Self {
        self.solver = solver;
        self
    }

    /// Fit the model
    pub fn fit(&mut self, x: &Tensor<B, 2>, y: &Tensor<B, 1>) {
        let x_processed = if self.fit_intercept {
            self.add_intercept(x)
        } else {
            x.clone()
        };

        match self.solver {
            Solver::Normal => {
                self.weights = Some(self.solve_normal_equation(&x_processed, y));
            }
            Solver::SGD | Solver::Adam => {
                self.weights = Some(self.solve_iterative(&x_processed, y));
            }
        }
    }

    /// Predict using the fitted model
    pub fn predict(&self, x: &Tensor<B, 2>) -> Tensor<B, 1> {
        let weights = self.weights.as_ref().expect("Model not fitted");
        let x_processed = if self.fit_intercept {
            self.add_intercept(x)
        } else {
            x.clone()
        };

        x_processed
            .matmul(weights.clone().unsqueeze_dim(1))
            .squeeze::<1>()
    }

    /// Solve using normal equation: w = (X^T X + λI)^{-1} X^T y
    fn solve_normal_equation(&self, x: &Tensor<B, 2>, y: &Tensor<B, 1>) -> Tensor<B, 1> {
        let xtx = x.clone().transpose().matmul(x.clone());
        let xty = x
            .clone()
            .transpose()
            .matmul(y.clone().unsqueeze_dim(1))
            .squeeze::<1>();

        // Add regularization
        let regularized_xtx = match self.regularization {
            Regularization::Ridge { alpha } => {
                let identity = Tensor::<B, 2>::eye(xtx.dims()[0], &xtx.device());
                xtx.add(identity.mul_scalar(alpha))
            }
            _ => xtx, // Other regularizations need iterative solving
        };

        // For this implementation, we'll use a simplified approach
        // In practice, you'd use proper matrix inversion or solving
        self.solve_linear_system(&regularized_xtx, &xty)
    }

    /// Solve `Aw = b` exactly.
    ///
    /// Falls back to the iterative solver if `A` turns out singular — which
    /// happens with perfectly collinear features.
    fn solve_linear_system(&self, a: &Tensor<B, 2>, b: &Tensor<B, 1>) -> Tensor<B, 1> {
        let n = a.dims()[0];
        let a_data = a
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap_or_default();
        let b_data = b
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap_or_default();

        match crate::utils::MathUtils::solve_linear_system(&a_data, &b_data, n) {
            Some(solution) => Tensor::from_data(TensorData::new(solution, [n]), &a.device()),
            None => self.solve_gradient_descent(a, b),
        }
    }

    /// Last-resort solver for singular systems: plain gradient descent on
    /// `\|Aw - b\|^2` with a step size scaled to the largest row sum, so it
    /// cannot diverge the way a fixed step size does.
    fn solve_gradient_descent(&self, a: &Tensor<B, 2>, b: &Tensor<B, 1>) -> Tensor<B, 1> {
        let n_features = a.dims()[0];
        let mut weights = Tensor::<B, 1>::zeros([n_features], &a.device());

        // A bound on the spectral norm; keeps the step inside the stable range.
        let scale = a.clone().abs().sum_dim(1).max().into_scalar().max(1e-6);
        let learning_rate = 1.0 / scale;

        for _ in 0..2000 {
            let pred = a
                .clone()
                .matmul(weights.clone().unsqueeze_dim(1))
                .reshape([n_features]);
            let residual = pred.sub(b.clone());
            let gradient = a
                .clone()
                .transpose()
                .matmul(residual.unsqueeze_dim(1))
                .reshape([n_features]);
            weights = weights.sub(gradient.mul_scalar(learning_rate / scale));
        }

        weights
    }

    /// Solve using iterative methods
    fn solve_iterative(&self, x: &Tensor<B, 2>, y: &Tensor<B, 1>) -> Tensor<B, 1> {
        let n_features = x.dims()[1];
        let n_samples = x.dims()[0] as f32;
        let mut weights = Tensor::<B, 1>::zeros([n_features], &x.device());

        let mut optimizer: Box<dyn Optimizer<B>> = match self.solver {
            Solver::SGD => Box::new(SGD::new(0.01)),
            Solver::Adam => Box::new(Adam::new(0.001)),
            _ => unreachable!(),
        };

        let max_iter = 1000;
        for _ in 0..max_iter {
            // Forward pass
            let predictions = x
                .clone()
                .matmul(weights.clone().unsqueeze_dim(1))
                .squeeze::<1>();
            let residual = predictions.sub(y.clone());

            // Compute gradient
            let mut gradient = x
                .clone()
                .transpose()
                .matmul(residual.unsqueeze_dim(1))
                .squeeze::<1>()
                .div_scalar(n_samples);

            // Add regularization to gradient
            gradient = self.add_regularization_gradient(&weights, gradient);

            // Update weights
            let weights_2d = weights.clone().unsqueeze_dim(0);
            let gradient_2d = gradient.unsqueeze_dim(0);
            let mut weights_2d_mut = weights_2d;
            optimizer.step(&mut weights_2d_mut, &gradient_2d);
            weights = weights_2d_mut.squeeze::<1>();
        }

        weights
    }

    /// Add regularization to gradient
    fn add_regularization_gradient(
        &self,
        weights: &Tensor<B, 1>,
        gradient: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        match self.regularization {
            Regularization::Ridge { alpha } => gradient.add(weights.clone().mul_scalar(alpha)),
            Regularization::Lasso { alpha } => {
                // Subgradient for L1
                let sign = weights.clone().sign();
                gradient.add(sign.mul_scalar(alpha))
            }
            Regularization::ElasticNet { alpha, l1_ratio } => {
                let l1_term = weights.clone().sign().mul_scalar(alpha * l1_ratio);
                let l2_term = weights.clone().mul_scalar(alpha * (1.0 - l1_ratio));
                gradient.add(l1_term).add(l2_term)
            }
            Regularization::None => gradient,
        }
    }

    /// Add intercept column to feature matrix
    fn add_intercept(&self, x: &Tensor<B, 2>) -> Tensor<B, 2> {
        let [n_samples, _] = x.dims();
        let ones = Tensor::<B, 2>::ones([n_samples, 1], &x.device());
        Tensor::cat(vec![ones, x.clone()], 1)
    }

    /// Get model coefficients
    pub fn coef(&self) -> Option<Tensor<B, 1>> {
        self.weights.clone().map(|w| {
            if self.fit_intercept {
                w.clone().slice([1..w.dims()[0]]) // Skip intercept
            } else {
                w
            }
        })
    }

    /// Get intercept
    pub fn intercept(&self) -> Option<f32> {
        if self.fit_intercept {
            self.weights
                .as_ref()
                .map(|w| w.clone().slice([0..1]).into_scalar())
        } else {
            None
        }
    }
}

impl<B: Backend<FloatElem = f32>> LogisticRegression<B> {
    /// Create a new Logistic Regression model
    pub fn new() -> Self {
        Self {
            weights: None,
            fit_intercept: true,
            regularization: Regularization::None,
            max_iter: 1000,
            tolerance: 1e-6,
            learning_rate: 0.01,
            solver: Solver::Adam,
            _phantom: PhantomData,
        }
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set regularization
    pub fn with_regularization(mut self, regularization: Regularization) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set solver
    pub fn with_solver(mut self, solver: Solver) -> Self {
        self.solver = solver;
        self
    }

    /// Fit the model
    pub fn fit(&mut self, x: &Tensor<B, 2>, y: &Tensor<B, 1>) {
        let x_processed = if self.fit_intercept {
            self.add_intercept(x)
        } else {
            x.clone()
        };

        self.weights = Some(self.fit_iterative(&x_processed, y));
    }

    /// Predict class probabilities
    pub fn predict_proba(&self, x: &Tensor<B, 2>) -> Tensor<B, 1> {
        let weights = self.weights.as_ref().expect("Model not fitted");
        let x_processed = if self.fit_intercept {
            self.add_intercept(x)
        } else {
            x.clone()
        };

        let logits = x_processed
            .matmul(weights.clone().unsqueeze_dim(1))
            .squeeze::<1>();
        self.sigmoid(logits)
    }

    /// Predict classes (0 or 1)
    pub fn predict(&self, x: &Tensor<B, 2>) -> Tensor<B, 1> {
        let probabilities = self.predict_proba(x);
        probabilities.greater_elem(0.5).float()
    }

    /// Fit using iterative optimization
    fn fit_iterative(&self, x: &Tensor<B, 2>, y: &Tensor<B, 1>) -> Tensor<B, 1> {
        let n_features = x.dims()[1];
        let n_samples = x.dims()[0] as f32;
        let mut weights = Tensor::<B, 1>::zeros([n_features], &x.device());

        let mut optimizer: Box<dyn Optimizer<B>> = match self.solver {
            Solver::SGD => Box::new(SGD::new(self.learning_rate)),
            Solver::Adam => Box::new(Adam::new(self.learning_rate)),
            _ => Box::new(Adam::new(self.learning_rate)),
        };

        let mut prev_loss = f32::INFINITY;

        for iter in 0..self.max_iter {
            // Forward pass
            let logits = x
                .clone()
                .matmul(weights.clone().unsqueeze_dim(1))
                .squeeze::<1>();
            let probabilities = self.sigmoid(logits);

            // Compute loss (binary cross-entropy)
            let loss = self.compute_loss(&probabilities, y, &weights, n_samples);

            // Check convergence
            if (prev_loss - loss).abs() < self.tolerance {
                println!("Converged after {} iterations", iter);
                break;
            }
            prev_loss = loss;

            // Compute gradient
            let residual = probabilities.sub(y.clone());
            let mut gradient = x
                .clone()
                .transpose()
                .matmul(residual.unsqueeze_dim(1))
                .squeeze::<1>()
                .div_scalar(n_samples);

            // Add regularization to gradient
            gradient = self.add_regularization_gradient(&weights, gradient);

            // Update weights
            let weights_2d = weights.clone().unsqueeze_dim(0);
            let gradient_2d = gradient.unsqueeze_dim(0);
            let mut weights_2d_mut = weights_2d;
            optimizer.step(&mut weights_2d_mut, &gradient_2d);
            weights = weights_2d_mut.squeeze::<1>();
        }

        weights
    }

    /// Sigmoid activation function
    fn sigmoid(&self, x: Tensor<B, 1>) -> Tensor<B, 1> {
        // σ(x) = 1 / (1 + exp(-x))
        let neg_x = x.neg();
        let exp_neg_x = neg_x.exp();
        let one_plus_exp = exp_neg_x.add_scalar(1.0);
        Tensor::<B, 1>::ones(one_plus_exp.dims(), &one_plus_exp.device()).div(one_plus_exp)
    }

    /// Compute binary cross-entropy loss with regularization
    fn compute_loss(
        &self,
        probabilities: &Tensor<B, 1>,
        y: &Tensor<B, 1>,
        weights: &Tensor<B, 1>,
        n_samples: f32,
    ) -> f32 {
        // Binary cross-entropy: -1/n * Σ[y*log(p) + (1-y)*log(1-p)]
        let epsilon = 1e-15; // Numerical stability
        let p_clipped = probabilities.clone().clamp(epsilon, 1.0 - epsilon);
        let log_p = p_clipped.clone().log();
        let log_1_minus_p = Tensor::<B, 1>::ones(p_clipped.dims(), &p_clipped.device())
            .sub(p_clipped)
            .log();

        let loss_per_sample = y.clone().mul(log_p).add(
            Tensor::<B, 1>::ones(y.dims(), &y.device())
                .sub(y.clone())
                .mul(log_1_minus_p),
        );
        let base_loss: f32 = loss_per_sample.sum().into_scalar() / (-n_samples);

        // Add regularization
        let reg_loss = match self.regularization {
            Regularization::Ridge { alpha } => {
                alpha * weights.clone().powf_scalar(2.0).sum().into_scalar() / (2.0 * n_samples)
            }
            Regularization::Lasso { alpha } => {
                alpha * weights.clone().abs().sum().into_scalar() / n_samples
            }
            Regularization::ElasticNet { alpha, l1_ratio } => {
                let l1_term = alpha * l1_ratio * weights.clone().abs().sum().into_scalar();
                let l2_term =
                    alpha * (1.0 - l1_ratio) * weights.clone().powf_scalar(2.0).sum().into_scalar()
                        / 2.0;
                (l1_term + l2_term) / n_samples
            }
            Regularization::None => 0.0,
        };

        base_loss + reg_loss
    }

    /// Add regularization to gradient
    fn add_regularization_gradient(
        &self,
        weights: &Tensor<B, 1>,
        gradient: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        match self.regularization {
            Regularization::Ridge { alpha } => gradient.add(weights.clone().mul_scalar(alpha)),
            Regularization::Lasso { alpha } => {
                let sign = weights.clone().sign();
                gradient.add(sign.mul_scalar(alpha))
            }
            Regularization::ElasticNet { alpha, l1_ratio } => {
                let l1_term = weights.clone().sign().mul_scalar(alpha * l1_ratio);
                let l2_term = weights.clone().mul_scalar(alpha * (1.0 - l1_ratio));
                gradient.add(l1_term).add(l2_term)
            }
            Regularization::None => gradient,
        }
    }

    /// Add intercept column to feature matrix
    fn add_intercept(&self, x: &Tensor<B, 2>) -> Tensor<B, 2> {
        let [n_samples, _] = x.dims();
        let ones = Tensor::<B, 2>::ones([n_samples, 1], &x.device());
        Tensor::cat(vec![ones, x.clone()], 1)
    }

    /// Get model coefficients
    pub fn coef(&self) -> Option<Tensor<B, 1>> {
        self.weights.clone().map(|w| {
            if self.fit_intercept {
                w.clone().slice([1..w.dims()[0]]) // Skip intercept
            } else {
                w
            }
        })
    }

    /// Get intercept
    pub fn intercept(&self) -> Option<f32> {
        if self.fit_intercept {
            self.weights
                .as_ref()
                .map(|w| w.clone().slice([0..1]).into_scalar())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{datasets, DefaultBackend};

    #[test]
    fn test_linear_regression() {
        let device = Default::default();

        // Create polynomial regression data
        let dataset =
            datasets::make_polynomial_regression::<DefaultBackend>(100, 2, 0.1, &device, Some(42));
        let (train_data, test_data) = dataset.train_test_split(0.8, Some(42));

        let mut model = LinearRegression::new();
        model.fit(&train_data.features, &train_data.labels.squeeze::<1>());

        let predictions = model.predict(&test_data.features);

        // Should have reasonable MSE
        use crate::metrics::RegressionMetrics;
        let mse = RegressionMetrics::mse(&test_data.labels.squeeze::<1>(), &predictions);
        assert!(mse < 1.0, "MSE should be reasonable: {}", mse);
    }

    #[test]
    fn test_logistic_regression() {
        let device = Default::default();

        // Create linearly separable data
        let dataset = datasets::make_linearly_separable::<DefaultBackend>(200, &device, Some(42));
        let (train_data, test_data) = dataset.train_test_split(0.8, Some(42));

        let mut model = LogisticRegression::new().with_max_iter(500);
        model.fit(&train_data.features, &train_data.labels.squeeze::<1>());

        let predictions = model.predict(&test_data.features);

        // Should have good accuracy on linearly separable data
        use crate::metrics::ClassificationMetrics;
        let accuracy =
            ClassificationMetrics::accuracy(&test_data.labels.squeeze::<1>(), &predictions);
        assert!(
            accuracy > 0.8,
            "Accuracy should be > 80% on linearly separable data: {}",
            accuracy
        );
    }

    #[test]
    fn test_ridge_regularization() {
        let device = Default::default();
        let dataset =
            datasets::make_polynomial_regression::<DefaultBackend>(50, 2, 0.1, &device, Some(42));

        let mut model = LinearRegression::new()
            .with_regularization(Regularization::Ridge { alpha: 0.1 })
            .with_solver(Solver::SGD);

        model.fit(&dataset.features, &dataset.labels.squeeze::<1>());

        // Should have fitted weights
        assert!(model.weights.is_some());
    }
}
