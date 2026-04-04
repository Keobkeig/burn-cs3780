//! Stochastic Gradient Descent (SGD) optimizer

use burn::tensor::{backend::Backend, Tensor};

use super::utils::{tensor_to_vec, vec_to_tensor};
use super::Optimizer;

/// Stochastic Gradient Descent (SGD) optimizer
#[derive(Debug, Clone)]
pub struct SGD {
    /// Learning rate for parameter updates
    pub learning_rate: f32,
    /// Momentum factor for accelerated gradient descent
    pub momentum: f32,
    /// Dampening factor for momentum term
    pub dampening: f32,
    /// Weight decay (L2 regularization) coefficient
    pub weight_decay: f32,
    /// Whether to use Nesterov momentum
    pub nesterov: bool,
    /// Velocity buffer for momentum updates
    velocity: Option<Vec<f32>>,
}

impl SGD {
    /// Create a new SGD optimizer
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            velocity: None,
        }
    }

    /// Create SGD with momentum
    pub fn with_momentum(learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            velocity: None,
        }
    }

    /// Create SGD with Nesterov momentum
    pub fn with_nesterov(learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: true,
            velocity: None,
        }
    }
}

impl<B: Backend<FloatElem = f32>> Optimizer<B> for SGD {
    fn step(&mut self, params: &mut Tensor<B, 2>, gradients: &Tensor<B, 2>) {
        let mut grad = gradients.clone();

        if self.weight_decay != 0.0 {
            grad = grad.add(params.clone().mul_scalar(self.weight_decay));
        }

        if self.momentum != 0.0 {
            let param_size = params.dims()[0] * params.dims()[1];

            if self.velocity.is_none() {
                self.velocity = Some(vec![0.0; param_size]);
            }

            if let Some(ref mut vel) = self.velocity {
                let grad_data = tensor_to_vec(&grad);

                for i in 0..param_size {
                    vel[i] = self.momentum * vel[i] + (1.0 - self.dampening) * grad_data[i];
                }

                let velocity_tensor = vec_to_tensor(vel, params.dims(), &params.device());

                if self.nesterov {
                    let update = grad.add(velocity_tensor.mul_scalar(self.momentum));
                    *params = params.clone().sub(update.mul_scalar(self.learning_rate));
                } else {
                    *params = params
                        .clone()
                        .sub(velocity_tensor.mul_scalar(self.learning_rate));
                }
            }
        } else {
            *params = params.clone().sub(grad.mul_scalar(self.learning_rate));
        }
    }

    fn reset(&mut self) {
        self.velocity = None;
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}
