//! Mathematical utilities: activation functions, encoding, and sampling

/// Mathematical utilities
pub struct MathUtils;

impl MathUtils {
    /// Sigmoid activation function
    pub fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Softmax function for a slice of values
    pub fn softmax(x: &[f32]) -> Vec<f32> {
        let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = x.iter().map(|&val| (val - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        exp_vals.iter().map(|&val| val / sum_exp).collect()
    }

    /// ReLU activation function
    pub fn relu(x: f32) -> f32 {
        x.max(0.0)
    }

    /// Leaky ReLU activation function
    pub fn leaky_relu(x: f32, alpha: f32) -> f32 {
        if x >= 0.0 { x } else { alpha * x }
    }

    /// Tanh activation function
    pub fn tanh(x: f32) -> f32 {
        x.tanh()
    }

    /// One-hot encode a slice of class indices
    pub fn one_hot_encode(labels: &[usize], num_classes: usize) -> Vec<Vec<f32>> {
        labels
            .iter()
            .map(|&label| {
                let mut encoding = vec![0.0; num_classes];
                if label < num_classes {
                    encoding[label] = 1.0;
                }
                encoding
            })
            .collect()
    }

    /// Sample from a categorical distribution
    pub fn categorical_sample(probabilities: &[f32], rng: &mut impl rand::Rng) -> usize {
        let total: f32 = probabilities.iter().sum();
        let mut cumsum = 0.0;
        let rand_val: f32 = rng.gen_range(0.0..total);

        for (i, &prob) in probabilities.iter().enumerate() {
            cumsum += prob;
            if rand_val <= cumsum {
                return i;
            }
        }
        probabilities.len() - 1
    }
}
