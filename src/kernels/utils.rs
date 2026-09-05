//! Kernel utility functions: normalization, validation, and ridge regression

use burn::tensor::{backend::Backend, Tensor};

/// Kernel utilities
pub struct KernelUtils;

impl KernelUtils {
    /// Normalize kernel matrix (center in feature space)
    pub fn normalize_kernel_matrix<B: Backend<FloatElem = f32>>(k: &Tensor<B, 2>) -> Tensor<B, 2> {
        let k_row_mean = k.clone().mean_dim(1);
        let k_col_mean = k.clone().mean_dim(0);
        let k_total_mean: f32 = k.clone().mean().into_scalar();

        let k_row_mean_broadcast = k_row_mean.unsqueeze_dim(1).repeat_dim(1, k.dims()[1]);
        let k_col_mean_broadcast = k_col_mean.unsqueeze_dim(0).repeat_dim(0, k.dims()[0]);

        k.clone()
            .sub(k_row_mean_broadcast)
            .sub(k_col_mean_broadcast)
            .add_scalar(k_total_mean)
    }

    /// Check if kernel matrix is positive semidefinite (simplified check)
    pub fn is_positive_semidefinite<B: Backend<FloatElem = f32>>(k: &Tensor<B, 2>) -> bool {
        let n = k.dims()[0];
        let mut diag_values = Vec::new();

        for i in 0..n {
            let diag_elem: f32 = k.clone().slice([i..i + 1, i..i + 1]).into_scalar();
            diag_values.push(diag_elem);
        }

        diag_values.iter().all(|&x| x >= 0.0)
    }

    /// Compute effective dimension of kernel matrix (simplified)
    pub fn effective_dimension<B: Backend<FloatElem = f32>>(
        k: &Tensor<B, 2>,
        threshold: f32,
    ) -> usize {
        let n = k.dims()[0];
        let mut trace = 0.0f32;

        for i in 0..n {
            let diag_elem: f32 = k.clone().slice([i..i + 1, i..i + 1]).into_scalar();
            trace += diag_elem;
        }

        let frobenius_norm_sq: f32 = k.clone().powf_scalar(2.0).sum().into_scalar();
        let frobenius_norm = frobenius_norm_sq.sqrt();

        if frobenius_norm < threshold {
            0
        } else {
            (trace / frobenius_norm * k.dims()[0] as f32) as usize
        }
    }
}

/// Kernel ridge regression utilities
pub struct KernelRidge;

impl KernelRidge {
    /// Solve kernel ridge regression for the dual coefficients:
    /// `(K + lambda*I) alpha = y`.
    ///
    /// `K + lambda*I` is symmetric positive definite for any `lambda > 0`, so
    /// this is a well-posed dense solve; it goes through the same Gauss-Jordan
    /// routine the normal equation uses. Returns an error only if the system is
    /// singular, which needs `lambda` at or near zero and a rank-deficient
    /// kernel matrix.
    ///
    /// # Arguments
    /// * `kernel_matrix` - Gram matrix `K`, shape `[n, n]`
    /// * `y` - Targets, shape `[n]`
    /// * `lambda` - Ridge penalty; larger values shrink the coefficients
    pub fn solve<B: Backend<FloatElem = f32>>(
        kernel_matrix: &Tensor<B, 2>,
        y: &Tensor<B, 1>,
        lambda: f32,
    ) -> Result<Tensor<B, 1>, String> {
        let [rows, cols] = kernel_matrix.dims();
        if rows != cols {
            return Err("Kernel matrix must be square".to_string());
        }
        if y.dims()[0] != rows {
            return Err("Target length must match the kernel matrix".to_string());
        }

        let device = kernel_matrix.device();
        let identity = Tensor::<B, 2>::eye(rows, &device);
        let regularized = kernel_matrix.clone().add(identity.mul_scalar(lambda));

        let a = regularized
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|_| "Failed to read the kernel matrix")?;
        let b = y
            .clone()
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|_| "Failed to read the targets")?;

        let alpha = crate::utils::MathUtils::solve_linear_system(&a, &b, rows)
            .ok_or_else(|| "Kernel matrix is singular; increase lambda".to_string())?;

        Ok(Tensor::from_data(
            burn::tensor::TensorData::new(alpha, [rows]),
            &device,
        ))
    }

    /// Predict with dual coefficients: `y_hat = K_test * alpha`.
    ///
    /// `k_test` is the kernel between the query points and the training points,
    /// shape `[n_queries, n_train]`.
    pub fn predict<B: Backend<FloatElem = f32>>(
        k_test: &Tensor<B, 2>,
        alpha: &Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let queries = k_test.dims()[0];
        k_test
            .clone()
            .matmul(alpha.clone().unsqueeze_dim(1))
            .reshape([queries])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{Kernel, RbfKernel};
    use crate::DefaultBackend;
    use burn::tensor::TensorData;

    #[test]
    fn kernel_ridge_recovers_the_training_targets() {
        let device = Default::default();
        // A one-dimensional non-linear target; an RBF kernel with a small
        // ridge should interpolate it closely.
        let xs: Vec<f32> = (0..12).map(|i| i as f32 * 0.4 - 2.2).collect();
        let ys: Vec<f32> = xs.iter().map(|x| (x * 1.3).sin()).collect();

        let x = Tensor::<DefaultBackend, 2>::from_data(
            TensorData::new(xs.clone(), [xs.len(), 1]),
            &device,
        );
        let y = Tensor::<DefaultBackend, 1>::from_data(
            TensorData::new(ys.clone(), [ys.len()]),
            &device,
        );

        let kernel = RbfKernel::new(1.0);
        let k = kernel.kernel_matrix(&x, &x);

        let alpha = KernelRidge::solve(&k, &y, 1e-6).expect("solve should succeed");
        let fitted = KernelRidge::predict(&k, &alpha);

        let fitted = fitted
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("read fitted values");
        for (got, want) in fitted.iter().zip(ys.iter()) {
            assert!(
                (got - want).abs() < 1e-2,
                "fitted {got} should track target {want}"
            );
        }
    }

    #[test]
    fn kernel_ridge_shrinks_with_lambda() {
        let device = Default::default();
        let xs: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
        let ys: Vec<f32> = xs.iter().map(|x| x * 2.0).collect();

        let x = Tensor::<DefaultBackend, 2>::from_data(
            TensorData::new(xs.clone(), [xs.len(), 1]),
            &device,
        );
        let y = Tensor::<DefaultBackend, 1>::from_data(
            TensorData::new(ys.clone(), [ys.len()]),
            &device,
        );

        let k = RbfKernel::new(0.5).kernel_matrix(&x, &x);

        let magnitude = |lambda: f32| -> f32 {
            KernelRidge::solve(&k, &y, lambda)
                .expect("solve should succeed")
                .abs()
                .sum()
                .into_scalar()
        };

        assert!(
            magnitude(10.0) < magnitude(0.01),
            "a larger ridge penalty must shrink the dual coefficients"
        );
    }

    #[test]
    fn kernel_ridge_rejects_mismatched_shapes() {
        let device = Default::default();
        let k = Tensor::<DefaultBackend, 2>::eye(3, &device);
        let y =
            Tensor::<DefaultBackend, 1>::from_data(TensorData::new(vec![1.0, 2.0], [2]), &device);
        assert!(KernelRidge::solve(&k, &y, 1.0).is_err());
    }
}
