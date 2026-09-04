//! Learning rate schedulers for controlling training dynamics

/// Trait for learning rate schedulers
pub trait LRScheduler {
    /// Get next learning rate and advance scheduler
    fn step(&mut self) -> f32;
    /// Reset scheduler state
    fn reset(&mut self);
}

/// Step learning rate scheduler: decays LR by gamma every step_size epochs
#[derive(Debug, Clone)]
pub struct StepLR {
    initial_lr: f32,
    step_size: u64,
    gamma: f32,
    current_step: u64,
}

impl StepLR {
    /// Create a new step learning rate scheduler
    pub fn new(initial_lr: f32, step_size: u64, gamma: f32) -> Self {
        Self {
            initial_lr,
            step_size,
            gamma,
            current_step: 0,
        }
    }
}

impl LRScheduler for StepLR {
    fn step(&mut self) -> f32 {
        // Compute for the current step, then advance — the first call must
        // return the initial rate, as it does in every other framework.
        let decay_factor = self.gamma.powi((self.current_step / self.step_size) as i32);
        self.current_step += 1;
        self.initial_lr * decay_factor
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

/// Exponential learning rate scheduler: decays LR by gamma each step
#[derive(Debug, Clone)]
pub struct ExponentialLR {
    initial_lr: f32,
    gamma: f32,
    current_step: u64,
}

impl ExponentialLR {
    /// Create a new exponential learning rate scheduler
    pub fn new(initial_lr: f32, gamma: f32) -> Self {
        Self {
            initial_lr,
            gamma,
            current_step: 0,
        }
    }
}

impl LRScheduler for ExponentialLR {
    fn step(&mut self) -> f32 {
        let lr = self.initial_lr * self.gamma.powi(self.current_step as i32);
        self.current_step += 1;
        lr
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}
