//! Radial Basis Function (RBF/Gaussian) kernel

use burn::tensor::{backend::Backend, Tensor};

use super::Kernel;

/// Radial Basis Function (RBF/Gaussian) kernel: K(x1, x2) = exp(-gamma * ||x1 - x2||^2)
#[derive(Debug, Clone)]
pub struct RbfKernel {
    /// Scaling parameter for the RBF kernel
    pub gamma: f32,
}

impl RbfKernel {
    /// Create a new RBF kernel
    pub fn new(gamma: f32) -> Self {
        Self { gamma }
    }

    /// Create RBF kernel with automatic gamma (1 / n_features)
    pub fn auto(n_features: usize) -> Self {
        Self::new(1.0 / n_features as f32)
    }
}

impl<B: Backend<FloatElem = f32>> Kernel<B> for RbfKernel {
    fn kernel(&self, x1: &Tensor<B, 1>, x2: &Tensor<B, 1>) -> f32 {
        let diff = x1.clone().sub(x2.clone());
        let squared_norm: f32 = diff.clone().mul(diff).sum().into_scalar();
        (-self.gamma * squared_norm).exp()
    }
}
