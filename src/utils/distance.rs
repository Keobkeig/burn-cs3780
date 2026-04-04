//! Distance metrics for comparing vectors and computing pairwise distances

use burn::tensor::{backend::Backend, Tensor, TensorData};

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

    /// Pairwise Euclidean distances between two sets of points
    pub fn pairwise_euclidean<B: Backend<FloatElem = f32>>(
        x1: &Tensor<B, 2>,
        x2: &Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let n1 = x1.dims()[0];
        let n2 = x2.dims()[0];
        let device = x1.device();

        let mut distances = Vec::with_capacity(n1 * n2);

        for i in 0..n1 {
            let row_i = x1.clone().slice([i..i + 1]).squeeze::<1>();
            for j in 0..n2 {
                let row_j = x2.clone().slice([j..j + 1]).squeeze::<1>();
                let dist = Self::euclidean(&row_i, &row_j);
                distances.push(dist);
            }
        }

        Tensor::from_data(TensorData::new(distances, [n1, n2]), &device)
    }
}
