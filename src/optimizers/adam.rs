//! Adam optimizer with optional AMSGrad variant

use burn::tensor::{backend::Backend, Tensor};

use super::utils::{tensor_to_vec, vec_to_tensor};
use super::Optimizer;

/// Adam optimizer
#[derive(Debug, Clone)]
pub struct Adam {
    /// Learning rate for parameter updates
    pub learning_rate: f32,
    /// Exponential decay rate for first moment estimates
    pub beta1: f32,
    /// Exponential decay rate for second moment estimates
    pub beta2: f32,
    /// Small constant for numerical stability
    pub epsilon: f32,
    /// Weight decay (L2 regularization) coefficient
    pub weight_decay: f32,
    /// Whether to use AMSGrad variant
    pub amsgrad: bool,
    step_count: u64,
    m: Option<Vec<f32>>,
    v: Option<Vec<f32>>,
    v_hat_max: Option<Vec<f32>>,
}

impl Adam {
    /// Create a new Adam optimizer
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            step_count: 0,
            m: None,
            v: None,
            v_hat_max: None,
        }
    }

    /// Create Adam with custom parameters
    pub fn with_params(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay: 0.0,
            amsgrad: false,
            step_count: 0,
            m: None,
            v: None,
            v_hat_max: None,
        }
    }

    /// Enable AMSGrad variant
    pub fn with_amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }
}

impl<B: Backend<FloatElem = f32>> Optimizer<B> for Adam {
    fn step(&mut self, params: &mut Tensor<B, 2>, gradients: &Tensor<B, 2>) {
        self.step_count += 1;

        let mut grad = gradients.clone();

        if self.weight_decay != 0.0 {
            grad = grad.add(params.clone().mul_scalar(self.weight_decay));
        }

        let param_size = params.dims()[0] * params.dims()[1];
        let grad_data = tensor_to_vec(&grad);

        if self.m.is_none() {
            self.m = Some(vec![0.0; param_size]);
            self.v = Some(vec![0.0; param_size]);
            if self.amsgrad {
                self.v_hat_max = Some(vec![0.0; param_size]);
            }
        }

        if let (Some(ref mut m), Some(ref mut v)) = (&mut self.m, &mut self.v) {
            for i in 0..param_size {
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad_data[i];
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad_data[i] * grad_data[i];
            }

            let bias_correction1 = 1.0 - self.beta1.powi(self.step_count as i32);
            let bias_correction2 = 1.0 - self.beta2.powi(self.step_count as i32);

            let step_size = self.learning_rate * (bias_correction2.sqrt() / bias_correction1);

            let mut updates = vec![0.0; param_size];

            if self.amsgrad {
                if let Some(ref mut v_hat_max) = &mut self.v_hat_max {
                    for i in 0..param_size {
                        let v_hat = v[i] / bias_correction2;
                        v_hat_max[i] = v_hat_max[i].max(v_hat);
                        updates[i] = step_size * m[i] / (v_hat_max[i].sqrt() + self.epsilon);
                    }
                }
            } else {
                for i in 0..param_size {
                    let m_hat = m[i] / bias_correction1;
                    let v_hat = v[i] / bias_correction2;
                    updates[i] = step_size * m_hat / (v_hat.sqrt() + self.epsilon);
                }
            }

            let update_tensor = vec_to_tensor(&updates, params.dims(), &params.device());
            *params = params.clone().sub(update_tensor);
        }
    }

    fn reset(&mut self) {
        self.step_count = 0;
        self.m = None;
        self.v = None;
        self.v_hat_max = None;
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}
