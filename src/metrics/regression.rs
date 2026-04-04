//! Regression evaluation metrics

use burn::tensor::{backend::Backend, Tensor};

/// Regression metrics
pub struct RegressionMetrics;

impl RegressionMetrics {
    /// Calculate Mean Squared Error (MSE)
    pub fn mse<B: Backend<FloatElem = f32>>(y_true: &Tensor<B, 1>, y_pred: &Tensor<B, 1>) -> f32 {
        let diff = y_true.clone().sub(y_pred.clone());
        diff.powf_scalar(2.0).mean().into_scalar()
    }

    /// Calculate Root Mean Squared Error (RMSE)
    pub fn rmse<B: Backend<FloatElem = f32>>(y_true: &Tensor<B, 1>, y_pred: &Tensor<B, 1>) -> f32 {
        Self::mse(y_true, y_pred).sqrt()
    }

    /// Calculate Mean Absolute Error (MAE)
    pub fn mae<B: Backend<FloatElem = f32>>(y_true: &Tensor<B, 1>, y_pred: &Tensor<B, 1>) -> f32 {
        y_true.clone().sub(y_pred.clone()).abs().mean().into_scalar()
    }

    /// Calculate R-squared (coefficient of determination)
    pub fn r2_score<B: Backend<FloatElem = f32>>(
        y_true: &Tensor<B, 1>,
        y_pred: &Tensor<B, 1>,
    ) -> f32 {
        let y_mean: f32 = y_true.clone().mean().into_scalar();

        let tss: f32 = y_true
            .clone()
            .sub_scalar(y_mean)
            .powf_scalar(2.0)
            .sum()
            .into_scalar();

        let rss: f32 = y_true
            .clone()
            .sub(y_pred.clone())
            .powf_scalar(2.0)
            .sum()
            .into_scalar();

        if tss == 0.0 {
            1.0
        } else {
            1.0 - (rss / tss)
        }
    }

    /// Calculate explained variance score
    pub fn explained_variance_score<B: Backend<FloatElem = f32>>(
        y_true: &Tensor<B, 1>,
        y_pred: &Tensor<B, 1>,
    ) -> f32 {
        let y_true_mean: f32 = y_true.clone().mean().into_scalar();
        let y_pred_mean: f32 = y_pred.clone().mean().into_scalar();

        let y_true_var: f32 = y_true
            .clone()
            .sub_scalar(y_true_mean)
            .powf_scalar(2.0)
            .mean()
            .into_scalar();
        let residual_var: f32 = y_true
            .clone()
            .sub(y_pred.clone())
            .sub_scalar(y_true_mean - y_pred_mean)
            .powf_scalar(2.0)
            .mean()
            .into_scalar();

        if y_true_var == 0.0 {
            1.0
        } else {
            1.0 - (residual_var / y_true_var)
        }
    }
}
