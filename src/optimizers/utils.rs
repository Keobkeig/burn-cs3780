//! Tensor/vector conversion utilities for optimizers

use burn::tensor::{backend::Backend, Tensor, TensorData};

pub(crate) fn tensor_to_vec<B: Backend<FloatElem = f32>>(tensor: &Tensor<B, 2>) -> Vec<f32> {
    tensor
        .to_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .unwrap_or_default()
}

pub(crate) fn vec_to_tensor<B: Backend<FloatElem = f32>>(
    vec: &[f32],
    dims: [usize; 2],
    device: &B::Device,
) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(vec.to_vec(), dims), device)
}
