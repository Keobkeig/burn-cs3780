//! Polynomial kernel: K(x1, x2) = (gamma * x1^T * x2 + coef0)^degree

use burn::tensor::{backend::Backend, Tensor};

use super::Kernel;

/// Polynomial kernel: K(x1, x2) = (gamma * x1^T * x2 + coef0)^degree
#[derive(Debug, Clone)]
pub struct PolynomialKernel {
    /// Degree of the polynomial
    pub degree: u32,
    /// Scaling parameter for the dot product
    pub gamma: f32,
    /// Independent term in the polynomial
    pub coef0: f32,
}

impl PolynomialKernel {
    /// Create a new polynomial kernel
    pub fn new(degree: u32, gamma: f32, coef0: f32) -> Self {
        Self {
            degree,
            gamma,
            coef0,
        }
    }
}

impl<B: Backend<FloatElem = f32>> Kernel<B> for PolynomialKernel {
    /// Whole Gram matrix at once.
    fn kernel_matrix(&self, x1: &Tensor<B, 2>, x2: &Tensor<B, 2>) -> Tensor<B, 2> {
        x1.clone()
            .matmul(x2.clone().transpose())
            .mul_scalar(self.gamma)
            .add_scalar(self.coef0)
            .powf_scalar(self.degree as f32)
    }

    fn kernel(&self, x1: &Tensor<B, 1>, x2: &Tensor<B, 1>) -> f32 {
        let dot_product: f32 = x1.clone().mul(x2.clone()).sum().into_scalar();
        (self.gamma * dot_product + self.coef0).powf(self.degree as f32)
    }
}
