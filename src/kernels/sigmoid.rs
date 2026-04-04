//! Sigmoid kernel: K(x1, x2) = tanh(gamma * x1^T * x2 + coef0)

use burn::tensor::{backend::Backend, Tensor};

use super::Kernel;

/// Sigmoid kernel: K(x1, x2) = tanh(gamma * x1^T * x2 + coef0)
#[derive(Debug, Clone)]
pub struct SigmoidKernel {
    /// Scaling parameter for the dot product
    pub gamma: f32,
    /// Independent term in the sigmoid function
    pub coef0: f32,
}

impl SigmoidKernel {
    /// Create a new sigmoid kernel
    pub fn new(gamma: f32, coef0: f32) -> Self {
        Self { gamma, coef0 }
    }
}

impl<B: Backend<FloatElem = f32>> Kernel<B> for SigmoidKernel {
    fn kernel(&self, x1: &Tensor<B, 1>, x2: &Tensor<B, 1>) -> f32 {
        let dot_product: f32 = x1.clone().mul(x2.clone()).sum().into_scalar();
        (self.gamma * dot_product + self.coef0).tanh()
    }
}
