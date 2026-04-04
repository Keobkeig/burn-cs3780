//! Cross-validation utilities

/// Cross-validation utilities
pub struct CrossValidation;

impl CrossValidation {
    /// Generate k-fold cross-validation indices
    pub fn k_fold_indices(
        n_samples: usize,
        k_folds: usize,
        shuffle: bool,
        seed: Option<u64>,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        use rand::{Rng, SeedableRng};

        let mut indices: Vec<usize> = (0..n_samples).collect();

        if shuffle {
            let mut rng = match seed {
                Some(s) => rand::rngs::StdRng::seed_from_u64(s),
                None => rand::rngs::StdRng::from_entropy(),
            };

            for i in (1..indices.len()).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }
        }

        let fold_size = n_samples / k_folds;
        let mut folds = Vec::new();

        for i in 0..k_folds {
            let start = i * fold_size;
            let end = if i == k_folds - 1 {
                n_samples
            } else {
                start + fold_size
            };

            let test_indices = indices[start..end].to_vec();
            let train_indices = indices[..start]
                .iter()
                .chain(indices[end..].iter())
                .cloned()
                .collect();

            folds.push((train_indices, test_indices));
        }

        folds
    }
}
