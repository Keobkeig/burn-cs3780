//! Data preprocessing utilities: standardization, normalization, feature engineering

use burn::tensor::{backend::Backend, Tensor};

/// Data preprocessing utilities
pub struct Preprocessing;

impl Preprocessing {
    /// Standardize features (z-score normalization)
    pub fn standardize<B: Backend<FloatElem = f32>>(
        data: &Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 1>, Tensor<B, 1>) {
        let mean = data.clone().mean_dim(0);
        let std = Self::std_dim(data, 0);

        let standardized = data
            .clone()
            .sub(mean.clone().unsqueeze_dim(0))
            .div(std.clone().unsqueeze_dim(0).add_scalar(1e-8));

        (standardized, mean.squeeze::<1>(), std.squeeze::<1>())
    }

    /// Min-Max normalization
    pub fn min_max_normalize<B: Backend<FloatElem = f32>>(
        data: &Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 1>, Tensor<B, 1>) {
        let min_vals = data.clone().min_dim(0);
        let max_vals = data.clone().max_dim(0);
        let range = max_vals.clone().sub(min_vals.clone()).add_scalar(1e-8);

        let normalized = data
            .clone()
            .sub(min_vals.clone().unsqueeze_dim(0))
            .div(range.unsqueeze_dim(0));

        (normalized, min_vals.squeeze::<1>(), max_vals.squeeze::<1>())
    }

    /// Add polynomial features up to the given degree
    pub fn polynomial_features<B: Backend<FloatElem = f32>>(
        data: &Tensor<B, 2>,
        degree: usize,
    ) -> Tensor<B, 2> {
        let [n_samples, n_features] = data.dims();
        let mut feature_columns = vec![data.clone()];

        for d in 2..=degree {
            for i in 0..n_features {
                let feature_col = data.clone().slice([0..n_samples, i..i + 1]);
                let powered = feature_col.powf_scalar(d as f32);
                feature_columns.push(powered);
            }
        }

        let mut result = feature_columns[0].clone();
        for col in &feature_columns[1..] {
            result = Tensor::cat(vec![result, col.clone()], 1);
        }

        result
    }

    /// Add bias term (column of ones) to the left of the feature matrix
    pub fn add_bias<B: Backend<FloatElem = f32>>(data: &Tensor<B, 2>) -> Tensor<B, 2> {
        let [n_samples, _] = data.dims();
        let ones = Tensor::<B, 2>::ones([n_samples, 1], &data.device());
        Tensor::cat(vec![ones, data.clone()], 1)
    }

    fn std_dim<B: Backend<FloatElem = f32>>(data: &Tensor<B, 2>, dim: usize) -> Tensor<B, 1> {
        let mean = data.clone().mean_dim(dim);
        let variance = data
            .clone()
            .sub(mean.unsqueeze_dim(dim))
            .powf_scalar(2.0)
            .mean_dim(dim);
        variance.sqrt().squeeze::<1>()
    }
}
