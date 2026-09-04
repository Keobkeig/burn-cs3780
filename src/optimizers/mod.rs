//! Optimization algorithms for machine learning models.
//!
//! This module implements various optimization algorithms used in machine learning,
//! including gradient descent variants and specialized optimizers for different algorithms.

pub mod adagrad;
pub mod adam;
pub mod schedulers;
pub mod sgd;
pub(crate) mod utils;

pub use adagrad::AdaGrad;
pub use adam::Adam;
pub use schedulers::{ExponentialLR, LRScheduler, StepLR};
pub use sgd::SGD;

use burn::tensor::{backend::Backend, Tensor};

/// Trait for optimization algorithms
pub trait Optimizer<B: Backend<FloatElem = f32>> {
    /// Update parameters given gradients
    fn step(&mut self, params: &mut Tensor<B, 2>, gradients: &Tensor<B, 2>);

    /// Reset optimizer state
    fn reset(&mut self);

    /// Get learning rate
    fn learning_rate(&self) -> f32;

    /// Set learning rate
    fn set_learning_rate(&mut self, lr: f32);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DefaultBackend;

    #[test]
    fn test_sgd_optimizer() {
        let device = Default::default();
        let mut params =
            Tensor::<DefaultBackend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
        let gradients = Tensor::<DefaultBackend, 2>::from_floats([[0.1, 0.2], [0.3, 0.4]], &device);

        let mut optimizer = SGD::new(0.1);
        optimizer.step(&mut params, &gradients);

        let new_val: f32 = params.clone().slice([0..1, 0..1]).into_scalar();
        assert!((new_val - 0.99).abs() < 1e-5);
    }

    #[test]
    fn test_step_lr_scheduler() {
        let mut scheduler = StepLR::new(0.1, 2, 0.1);

        assert!((scheduler.step() - 0.1).abs() < 1e-6);
        assert!((scheduler.step() - 0.1).abs() < 1e-6);
        assert!((scheduler.step() - 0.01).abs() < 1e-6);
    }
}
