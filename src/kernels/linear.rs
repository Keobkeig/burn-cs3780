//! Linear kernel: K(x1, x2) = x1^T * x2

use burn::tensor::{backend::Backend, Tensor};

use super::Kernel;

/// Linear kernel: K(x1, x2) = x1^T * x2
#[derive(Debug, Clone)]
pub struct LinearKernel;

impl<B: Backend<FloatElem = f32>> Kernel<B> for LinearKernel {
    /// Whole Gram matrix in one matmul.
    fn kernel_matrix(&self, x1: &Tensor<B, 2>, x2: &Tensor<B, 2>) -> Tensor<B, 2> {
        x1.clone().matmul(x2.clone().transpose())
    }

    fn kernel(&self, x1: &Tensor<B, 1>, x2: &Tensor<B, 1>) -> f32 {
        x1.clone().mul(x2.clone()).sum().into_scalar()
    }
}
