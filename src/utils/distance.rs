//! Distance metrics for comparing vectors and computing pairwise distances

use burn::tensor::{backend::Backend, Tensor};

/// Distance metrics
pub struct Distance;

impl Distance {
    /// Euclidean distance between two vectors
    pub fn euclidean<B: Backend<FloatElem = f32>>(x1: &Tensor<B, 1>, x2: &Tensor<B, 1>) -> f32 {
        let diff = x1.clone().sub(x2.clone());
        let squared_diff: f32 = diff.clone().mul(diff).sum().into_scalar();
        squared_diff.sqrt()
    }

    /// Manhattan distance between two vectors
    pub fn manhattan<B: Backend<FloatElem = f32>>(x1: &Tensor<B, 1>, x2: &Tensor<B, 1>) -> f32 {
        x1.clone().sub(x2.clone()).abs().sum().into_scalar()
    }

    /// Cosine similarity between two vectors
    pub fn cosine_similarity<B: Backend<FloatElem = f32>>(
        x1: &Tensor<B, 1>,
        x2: &Tensor<B, 1>,
    ) -> f32 {
        let dot_product: f32 = x1.clone().mul(x2.clone()).sum().into_scalar();
        let norm1: f32 = x1.clone().mul(x1.clone()).sum().into_scalar().sqrt();
        let norm2: f32 = x2.clone().mul(x2.clone()).sum().into_scalar().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }

    /// Squared-norm rows of `x`, shaped `[n, 1]`.
    fn squared_norms<B: Backend<FloatElem = f32>>(x: &Tensor<B, 2>) -> Tensor<B, 2> {
        x.clone().powf_scalar(2.0).sum_dim(1)
    }

    /// Pairwise Euclidean distances between two sets of points, `[n1, n2]`.
    ///
    /// Expanded as `||a||^2 + ||b||^2 - 2 a.b` so the whole matrix is three
    /// tensor ops. Evaluating a model over a mesh means tens of thousands of
    /// rows, and a per-pair loop spends all its time allocating.
    pub fn pairwise_euclidean<B: Backend<FloatElem = f32>>(
        x1: &Tensor<B, 2>,
        x2: &Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let cross = x1.clone().matmul(x2.clone().transpose());
        let squared =
            Self::squared_norms(x1) + Self::squared_norms(x2).transpose() - cross.mul_scalar(2.0);
        // Cancellation can push exact zeros slightly negative.
        squared.clamp_min(0.0).sqrt()
    }

    /// Pairwise Manhattan distances, `[n1, n2]`.
    ///
    /// No dot-product identity here, so this broadcasts to `[n1, n2, d]` and
    /// reduces. Fine for the low-dimensional data these models see.
    pub fn pairwise_manhattan<B: Backend<FloatElem = f32>>(
        x1: &Tensor<B, 2>,
        x2: &Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [n1, _] = x1.dims();
        let [n2, _] = x2.dims();
        let a = x1.clone().unsqueeze_dim::<3>(1);
        let b = x2.clone().unsqueeze_dim::<3>(0);
        (a - b).abs().sum_dim(2).reshape([n1, n2])
    }

    /// Pairwise cosine distances (`1 - similarity`), `[n1, n2]`.
    pub fn pairwise_cosine<B: Backend<FloatElem = f32>>(
        x1: &Tensor<B, 2>,
        x2: &Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let cross = x1.clone().matmul(x2.clone().transpose());
        // Floor the norms so a zero vector gives similarity 0 rather than NaN.
        let n1 = Self::squared_norms(x1).sqrt().clamp_min(1e-12);
        let n2 = Self::squared_norms(x2).sqrt().clamp_min(1e-12).transpose();
        cross.div(n1 * n2).neg().add_scalar(1.0)
    }
}
