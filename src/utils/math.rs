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
        if x >= 0.0 {
            x
        } else {
            alpha * x
        }
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

    /// Solve `Ax = b` by Gauss-Jordan elimination with partial pivoting.
    ///
    /// `a` is row-major `n x n`. Returns `None` if the matrix is singular, so
    /// the caller can fall back to an iterative solver.
    pub fn solve_linear_system(a: &[f32], b: &[f32], n: usize) -> Option<Vec<f32>> {
        // Augmented matrix, n rows of n + 1 columns.
        let w = n + 1;
        let mut m = vec![0.0f32; n * w];
        for row in 0..n {
            m[row * w..row * w + n].copy_from_slice(&a[row * n..row * n + n]);
            m[row * w + n] = b[row];
        }

        for col in 0..n {
            let mut pivot = col;
            for row in col + 1..n {
                if m[row * w + col].abs() > m[pivot * w + col].abs() {
                    pivot = row;
                }
            }
            if m[pivot * w + col].abs() < 1e-10 {
                return None;
            }
            if pivot != col {
                for k in 0..w {
                    m.swap(col * w + k, pivot * w + k);
                }
            }

            let diagonal = m[col * w + col];
            for k in col..w {
                m[col * w + k] /= diagonal;
            }
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = m[row * w + col];
                if factor == 0.0 {
                    continue;
                }
                for k in col..w {
                    m[row * w + k] -= factor * m[col * w + k];
                }
            }
        }

        Some((0..n).map(|row| m[row * w + n]).collect())
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
