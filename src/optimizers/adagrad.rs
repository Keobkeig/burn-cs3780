//! AdaGrad optimizer with adaptive per-parameter learning rates

use burn::tensor::{backend::Backend, Tensor};

use super::utils::{tensor_to_vec, vec_to_tensor};
use super::Optimizer;

/// AdaGrad optimizer
#[derive(Debug, Clone)]
pub struct AdaGrad {
    /// Learning rate for parameter updates
    pub learning_rate: f32,
    /// Small constant for numerical stability
    pub epsilon: f32,
    /// Weight decay (L2 regularization) coefficient
    pub weight_decay: f32,
    sum_squared_gradients: Option<Vec<f32>>,
}

impl AdaGrad {
    /// Create a new AdaGrad optimizer
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            epsilon: 1e-10,
            weight_decay: 0.0,
            sum_squared_gradients: None,
        }
    }
}

impl<B: Backend<FloatElem = f32>> Optimizer<B> for AdaGrad {
    fn step(&mut self, params: &mut Tensor<B, 2>, gradients: &Tensor<B, 2>) {
        let mut grad = gradients.clone();

        if self.weight_decay != 0.0 {
            grad = grad.add(params.clone().mul_scalar(self.weight_decay));
        }

        let param_size = params.dims()[0] * params.dims()[1];
        let grad_data = tensor_to_vec(&grad);

        if self.sum_squared_gradients.is_none() {
            self.sum_squared_gradients = Some(vec![0.0; param_size]);
        }

        if let Some(ref mut sum_sq_grad) = &mut self.sum_squared_gradients {
            let mut updates = vec![0.0; param_size];

            for i in 0..param_size {
                sum_sq_grad[i] += grad_data[i] * grad_data[i];
                let adjusted_lr = self.learning_rate / (sum_sq_grad[i].sqrt() + self.epsilon);
                updates[i] = adjusted_lr * grad_data[i];
            }

            let update_tensor = vec_to_tensor(&updates, params.dims(), &params.device());
            *params = params.clone().sub(update_tensor);
        }
    }

    fn reset(&mut self) {
        self.sum_squared_gradients = None;
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}
