//! Kernel functions for Support Vector Machines and other kernel methods.
//!
//! This module implements various kernel functions used in kernel methods,
//! particularly Support Vector Machines, as covered in CS3780.

pub mod linear;
pub mod polynomial;
pub mod precomputed;
pub mod rbf;
pub mod sigmoid;
pub mod utils;

pub use linear::LinearKernel;
pub use polynomial::PolynomialKernel;
pub use precomputed::PrecomputedKernel;
pub use rbf::RbfKernel;
pub use sigmoid::SigmoidKernel;
pub use utils::{KernelRidge, KernelUtils};

use burn::tensor::{backend::Backend, Tensor, TensorData};

/// Trait for kernel functions (constrained to f32 for simplicity)
pub trait Kernel<B: Backend<FloatElem = f32>> {
    /// Compute the kernel function between two vectors
    fn kernel(&self, x1: &Tensor<B, 1>, x2: &Tensor<B, 1>) -> f32;

    /// Compute the kernel matrix between two sets of vectors
    fn kernel_matrix(&self, x1: &Tensor<B, 2>, x2: &Tensor<B, 2>) -> Tensor<B, 2> {
        let n1 = x1.dims()[0];
        let n2 = x2.dims()[0];
        let device = x1.device();

        let mut kernel_values = Vec::with_capacity(n1 * n2);

        for i in 0..n1 {
            let row_i = x1.clone().slice([i..i + 1]).squeeze::<1>();
            for j in 0..n2 {
                let row_j = x2.clone().slice([j..j + 1]).squeeze::<1>();
                let k_value = self.kernel(&row_i, &row_j);
                kernel_values.push(k_value);
            }
        }

        Tensor::from_data(TensorData::new(kernel_values, [n1, n2]), &device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DefaultBackend;

    #[test]
    fn test_linear_kernel() {
        let device = Default::default();
        let x1 = Tensor::<DefaultBackend, 1>::from_floats([1.0, 2.0, 3.0], &device);
        let x2 = Tensor::<DefaultBackend, 1>::from_floats([4.0, 5.0, 6.0], &device);

        let kernel = LinearKernel;
        let result = kernel.kernel(&x1, &x2);

        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((result - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_rbf_kernel() {
        let device = Default::default();
        let x1 = Tensor::<DefaultBackend, 1>::from_floats([1.0, 2.0], &device);
        let x2 = Tensor::<DefaultBackend, 1>::from_floats([1.0, 2.0], &device);

        let kernel = RbfKernel::new(1.0);
        let result = kernel.kernel(&x1, &x2);

        // Same vectors should give kernel value of 1.0
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_polynomial_kernel() {
        let device = Default::default();
        let x1 = Tensor::<DefaultBackend, 1>::from_floats([1.0, 1.0], &device);
        let x2 = Tensor::<DefaultBackend, 1>::from_floats([1.0, 1.0], &device);

        let kernel = PolynomialKernel::new(2, 1.0, 1.0);
        let result = kernel.kernel(&x1, &x2);

        // (1.0 * (1*1 + 1*1) + 1.0)^2 = (1.0 * 2 + 1.0)^2 = 3^2 = 9
        assert!((result - 9.0).abs() < 1e-6);
    }
}
