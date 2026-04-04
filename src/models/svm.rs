//! Support Vector Machine implementation using Sequential Minimal Optimization (SMO)

use crate::kernels::{Kernel, LinearKernel, PolynomialKernel, RbfKernel, SigmoidKernel};
use burn::tensor::{backend::Backend, Device, Tensor, TensorData};

/// Kernel type enumeration for SVM
#[derive(Debug, Clone)]
pub enum KernelType {
    /// Linear kernel
    Linear,
    /// RBF (Gaussian) kernel
    RBF {
        /// Gamma parameter for RBF kernel
        gamma: f32,
    },
    /// Polynomial kernel
    Polynomial {
        /// Degree of the polynomial
        degree: u32,
        /// Gamma scaling parameter
        gamma: f32,
        /// Independent term
        coef0: f32,
    },
    /// Sigmoid kernel
    Sigmoid {
        /// Gamma scaling parameter
        gamma: f32,
        /// Independent term
        coef0: f32,
    },
}

/// Unified kernel wrapper for SVM
#[derive(Debug, Clone)]
pub enum SVMKernel {
    /// Linear kernel variant
    Linear(LinearKernel),
    /// RBF kernel variant
    RBF(RbfKernel),
    /// Polynomial kernel variant
    Polynomial(PolynomialKernel),
    /// Sigmoid kernel variant
    Sigmoid(SigmoidKernel),
}

impl SVMKernel {
    /// Create a new SVM kernel from kernel type specification
    pub fn new(kernel_type: KernelType) -> Self {
        match kernel_type {
            KernelType::Linear => SVMKernel::Linear(LinearKernel),
            KernelType::RBF { gamma } => SVMKernel::RBF(RbfKernel::new(gamma)),
            KernelType::Polynomial {
                degree,
                gamma,
                coef0,
            } => SVMKernel::Polynomial(PolynomialKernel::new(degree, gamma, coef0)),
            KernelType::Sigmoid { gamma, coef0 } => {
                SVMKernel::Sigmoid(SigmoidKernel::new(gamma, coef0))
            }
        }
    }

    /// Compute kernel matrix between two input tensors
    pub fn compute_matrix<B: Backend<FloatElem = f32>>(
        &self,
        x1: &Tensor<B, 2>,
        x2: &Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        match self {
            SVMKernel::Linear(k) => k.kernel_matrix(x1, x2),
            SVMKernel::RBF(k) => k.kernel_matrix(x1, x2),
            SVMKernel::Polynomial(k) => k.kernel_matrix(x1, x2),
            SVMKernel::Sigmoid(k) => k.kernel_matrix(x1, x2),
        }
    }
}

/// Support Vector Machine for binary and multi-class classification
#[derive(Debug, Clone)]
pub struct SVM<B: Backend<FloatElem = f32>> {
    /// Lagrange multipliers (alpha values)
    alphas: Option<Tensor<B, 1>>,
    /// Support vectors (subset of training data)
    support_vectors: Option<Tensor<B, 2>>,
    /// Support vector labels
    support_labels: Option<Tensor<B, 1>>,
    /// Bias term
    bias: f32,
    /// Regularization parameter
    c: f32,
    /// Tolerance for convergence
    tolerance: f32,
    /// Maximum number of iterations
    max_iterations: usize,
    /// Kernel function
    kernel: SVMKernel,
    /// Device for computations
    device: Device<B>,
}

impl<B: Backend<FloatElem = f32>> SVM<B> {
    /// Create a new SVM with specified parameters
    pub fn new(
        c: f32,
        kernel_type: KernelType,
        tolerance: f32,
        max_iterations: usize,
        device: Device<B>,
    ) -> Self {
        Self {
            alphas: None,
            support_vectors: None,
            support_labels: None,
            bias: 0.0,
            c,
            tolerance,
            max_iterations,
            kernel: SVMKernel::new(kernel_type),
            device,
        }
    }

    /// Create SVM with linear kernel (default parameters)
    pub fn linear(device: Device<B>) -> Self {
        Self::new(1.0, KernelType::Linear, 1e-3, 1000, device)
    }

    /// Create SVM with RBF kernel
    pub fn rbf(gamma: f32, device: Device<B>) -> Self {
        Self::new(1.0, KernelType::RBF { gamma }, 1e-3, 1000, device)
    }

    /// Train the SVM using Sequential Minimal Optimization (SMO)
    pub fn fit(&mut self, x: Tensor<B, 2>, y: Tensor<B, 1>) -> Result<(), String> {
        let [n_samples, _n_features] = x.dims();

        // Convert labels to {-1, 1} format
        let y_svm = self.convert_labels(&y);

        // Initialize alphas to zero
        let mut alphas = Tensor::zeros([n_samples], &self.device);
        let mut bias = 0.0f32;

        // Compute kernel matrix
        let kernel_matrix = self.kernel.compute_matrix(&x, &x);

        // SMO algorithm
        for iteration in 0..self.max_iterations {
            let mut changed_alphas = 0;

            for i in 0..n_samples {
                // Calculate error for example i
                let error_i = self.compute_error(i, &x, &y_svm, &alphas, bias, &kernel_matrix);

                // Check KKT conditions
                let alpha_i = alphas.clone().slice([i..i + 1]).into_scalar();
                let y_i = y_svm.clone().slice([i..i + 1]).into_scalar();

                if (alpha_i < self.c - self.tolerance && y_i * error_i < -self.tolerance)
                    || (alpha_i > self.tolerance && y_i * error_i > self.tolerance)
                {
                    // Select second alpha using heuristic
                    let j = self.select_second_alpha(i, n_samples, &error_i);
                    if i == j {
                        continue;
                    }

                    let error_j = self.compute_error(j, &x, &y_svm, &alphas, bias, &kernel_matrix);

                    // Update alphas
                    let alpha_j = alphas.clone().slice([j..j + 1]).into_scalar();
                    let y_j = y_svm.clone().slice([j..j + 1]).into_scalar();

                    // Compute bounds
                    let (low, high) = if y_i != y_j {
                        let diff = alpha_j - alpha_i;
                        (0.0f32.max(diff), self.c.min(self.c + diff))
                    } else {
                        let sum = alpha_i + alpha_j;
                        (0.0f32.max(sum - self.c), self.c.min(sum))
                    };

                    if (high - low).abs() < self.tolerance {
                        continue;
                    }

                    // Compute eta (second derivative)
                    let k_ii = kernel_matrix
                        .clone()
                        .slice([i..i + 1, i..i + 1])
                        .into_scalar();
                    let k_jj = kernel_matrix
                        .clone()
                        .slice([j..j + 1, j..j + 1])
                        .into_scalar();
                    let k_ij = kernel_matrix
                        .clone()
                        .slice([i..i + 1, j..j + 1])
                        .into_scalar();
                    let eta = k_ii + k_jj - 2.0 * k_ij;

                    if eta <= 0.0 {
                        continue;
                    }

                    // Update alpha_j
                    let new_alpha_j = alpha_j + y_j * (error_i - error_j) / eta;
                    let new_alpha_j = new_alpha_j.max(low).min(high);

                    if (new_alpha_j - alpha_j).abs() < self.tolerance {
                        continue;
                    }

                    // Update alpha_i
                    let new_alpha_i = alpha_i + y_i * y_j * (alpha_j - new_alpha_j);

                    // Update alphas tensor
                    alphas = alphas
                        .slice_assign([i..i + 1], Tensor::from_floats([new_alpha_i], &self.device))
                        .slice_assign([j..j + 1], Tensor::from_floats([new_alpha_j], &self.device));

                    // Update bias
                    let b1 = bias
                        - error_i
                        - y_i * (new_alpha_i - alpha_i) * k_ii
                        - y_j * (new_alpha_j - alpha_j) * k_ij;
                    let b2 = bias
                        - error_j
                        - y_i * (new_alpha_i - alpha_i) * k_ij
                        - y_j * (new_alpha_j - alpha_j) * k_jj;

                    bias = if new_alpha_i > 0.0 && new_alpha_i < self.c {
                        b1
                    } else if new_alpha_j > 0.0 && new_alpha_j < self.c {
                        b2
                    } else {
                        (b1 + b2) / 2.0
                    };

                    changed_alphas += 1;
                }
            }

            // Check for convergence
            if changed_alphas == 0 {
                println!("SVM converged after {} iterations", iteration + 1);
                break;
            }
        }

        // Extract support vectors
        self.extract_support_vectors(&x, &y_svm, &alphas);
        self.bias = bias;

        Ok(())
    }

    /// Predict class labels for new data
    pub fn predict(&self, x: Tensor<B, 2>) -> Result<Tensor<B, 1>, String> {
        let support_vectors = self
            .support_vectors
            .as_ref()
            .ok_or("Model not trained. Call fit() first.")?;
        let support_labels = self
            .support_labels
            .as_ref()
            .ok_or("Model not trained. Call fit() first.")?;
        let alphas = self
            .alphas
            .as_ref()
            .ok_or("Model not trained. Call fit() first.")?;

        let [n_samples, n_features] = x.dims();
        let mut predictions = Vec::new();

        for i in 0..n_samples {
            let sample = x.clone().slice([i..i + 1, 0..n_features]);
            let kernel_values = self.kernel.compute_matrix(&sample, support_vectors);

            // Compute decision function: sum(alpha_i * y_i * K(x, x_i)) + bias
            let weighted_sum =
                (alphas.clone() * support_labels.clone() * kernel_values.squeeze::<1>())
                    .sum()
                    .into_scalar();

            let prediction = if weighted_sum + self.bias > 0.0 {
                1.0
            } else {
                -1.0
            };
            predictions.push(prediction);
        }

        Ok(Tensor::from_floats(
            TensorData::new(predictions, [n_samples]),
            &self.device,
        ))
    }

    /// Get decision function values (distances from hyperplane)
    pub fn decision_function(&self, x: Tensor<B, 2>) -> Result<Tensor<B, 1>, String> {
        let support_vectors = self
            .support_vectors
            .as_ref()
            .ok_or("Model not trained. Call fit() first.")?;
        let support_labels = self
            .support_labels
            .as_ref()
            .ok_or("Model not trained. Call fit() first.")?;
        let alphas = self
            .alphas
            .as_ref()
            .ok_or("Model not trained. Call fit() first.")?;

        let [n_samples, n_features] = x.dims();
        let mut scores = Vec::new();

        for i in 0..n_samples {
            let sample = x.clone().slice([i..i + 1, 0..n_features]);
            let kernel_values = self.kernel.compute_matrix(&sample, support_vectors);

            let weighted_sum =
                (alphas.clone() * support_labels.clone() * kernel_values.squeeze::<1>())
                    .sum()
                    .into_scalar();

            scores.push(weighted_sum + self.bias);
        }

        Ok(Tensor::from_floats(
            TensorData::new(scores, [n_samples]),
            &self.device,
        ))
    }

    /// Get support vectors
    pub fn get_support_vectors(&self) -> Option<&Tensor<B, 2>> {
        self.support_vectors.as_ref()
    }

    /// Get number of support vectors
    pub fn n_support_vectors(&self) -> usize {
        self.support_vectors
            .as_ref()
            .map(|sv| sv.dims()[0])
            .unwrap_or(0)
    }

    // Helper methods

    fn convert_labels(&self, y: &Tensor<B, 1>) -> Tensor<B, 1> {
        // Convert labels to {-1, 1} format
        // Assumes binary classification with labels 0,1
        y.clone() * 2.0 - 1.0
    }

    fn compute_error(
        &self,
        i: usize,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 1>,
        alphas: &Tensor<B, 1>,
        bias: f32,
        kernel_matrix: &Tensor<B, 2>,
    ) -> f32 {
        let [n_samples, _] = x.dims();
        let mut sum = 0.0f32;

        for j in 0..n_samples {
            let alpha_j = alphas.clone().slice([j..j + 1]).into_scalar();
            let y_j = y.clone().slice([j..j + 1]).into_scalar();
            let k_ij = kernel_matrix
                .clone()
                .slice([i..i + 1, j..j + 1])
                .into_scalar();
            sum += alpha_j * y_j * k_ij;
        }

        let y_i = y.clone().slice([i..i + 1]).into_scalar();
        sum + bias - y_i
    }

    fn select_second_alpha(&self, i: usize, n_samples: usize, _error_i: &f32) -> usize {
        // Simple heuristic: select random j != i
        // In practice, more sophisticated heuristics can be used
        let mut j = (i + 1) % n_samples;
        if j == i && n_samples > 1 {
            j = (i + 2) % n_samples;
        }
        j
    }

    fn extract_support_vectors(
        &mut self,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 1>,
        alphas: &Tensor<B, 1>,
    ) {
        let [n_samples, n_features] = x.dims();
        let mut support_indices = Vec::new();
        let mut support_alphas = Vec::new();

        // Find non-zero alphas (support vectors)
        for i in 0..n_samples {
            let alpha = alphas.clone().slice([i..i + 1]).into_scalar();
            if alpha > self.tolerance {
                support_indices.push(i);
                support_alphas.push(alpha);
            }
        }

        if !support_indices.is_empty() {
            // Extract support vectors and labels
            let mut sv_data = Vec::new();
            let mut sv_labels = Vec::new();

            for &idx in &support_indices {
                // Extract features for this support vector
                for j in 0..n_features {
                    let val = x.clone().slice([idx..idx + 1, j..j + 1]).into_scalar();
                    sv_data.push(val);
                }
                let label = y.clone().slice([idx..idx + 1]).into_scalar();
                sv_labels.push(label);
            }

            self.support_vectors = Some(Tensor::from_floats(
                TensorData::new(sv_data, [support_indices.len(), n_features]),
                &self.device,
            ));
            self.support_labels = Some(Tensor::from_floats(
                TensorData::new(sv_labels, [support_indices.len()]),
                &self.device,
            ));
            self.alphas = Some(Tensor::from_floats(
                TensorData::new(support_alphas, [support_indices.len()]),
                &self.device,
            ));
        }
    }
}

impl<B: Backend<FloatElem = f32>> Default for SVM<B>
where
    B: 'static,
{
    fn default() -> Self {
        // Cannot provide a sensible default without knowing the backend
        panic!("Use SVM::linear(device) or SVM::new() to create an SVM instance")
    }
}
