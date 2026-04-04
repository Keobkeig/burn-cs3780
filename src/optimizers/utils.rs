//! Tensor/vector conversion utilities for optimizers

use burn::tensor::{backend::Backend, Tensor, TensorData};

pub(crate) fn tensor_to_vec<B: Backend<FloatElem = f32>>(tensor: &Tensor<B, 2>) -> Vec<f32> {
    let [rows, cols] = tensor.dims();
    let mut result = Vec::with_capacity(rows * cols);

    for i in 0..rows {
        for j in 0..cols {
            let val: f32 = tensor
                .clone()
                .slice([i..i + 1, j..j + 1])
                .squeeze::<1>()
                .squeeze::<1>()
                .into_scalar();
            result.push(val);
        }
    }
    result
}

pub(crate) fn vec_to_tensor<B: Backend<FloatElem = f32>>(
    vec: &[f32],
    dims: [usize; 2],
    device: &B::Device,
) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(vec.to_vec(), dims), device)
}
