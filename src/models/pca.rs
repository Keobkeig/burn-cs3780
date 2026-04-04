//! Principal Component Analysis (PCA) implementation using Burn.
//!
//! This module implements PCA for dimensionality reduction, feature extraction,
//! and data visualization. PCA finds the principal components that explain
//! the maximum variance in the data.

use burn::tensor::{backend::Backend, Tensor, TensorData};
use std::fmt;

/// Principal Component Analysis (PCA) implementation
///
/// PCA is a dimensionality reduction technique that finds the principal components
/// (directions of maximum variance) in the data. It can be used for feature extraction,
/// data visualization, and noise reduction.
#[derive(Debug, Clone)]
pub struct PCA<B: Backend<FloatElem = f32>> {
    /// Number of components to keep
    n_components: Option<usize>,
    /// Whether to center the data (subtract mean)
    center: bool,
    /// Whether to scale the data to unit variance
    scale: bool,

    // Fitted parameters
    /// Principal components (eigenvectors) - shape: (n_features, n_components)
    components: Option<Tensor<B, 2>>,
    /// Explained variance by each component
    explained_variance: Option<Tensor<B, 1>>,
    /// Explained variance ratio by each component
    explained_variance_ratio: Option<Tensor<B, 1>>,
    /// Cumulative explained variance ratio
    cumulative_variance_ratio: Option<Tensor<B, 1>>,
    /// Mean of the training data
    mean: Option<Tensor<B, 1>>,
    /// Standard deviation of the training data (if scaling is enabled)
    std: Option<Tensor<B, 1>>,
    /// Total variance in the data
    total_variance: Option<f32>,
    /// Whether the model has been fitted
    is_fitted: bool,
}

impl<B: Backend<FloatElem = f32>> PCA<B> {
    /// Create a new PCA instance
    ///
    /// # Arguments
    /// * `n_components` - Number of components to keep (if None, keep all)
    /// * `center` - Whether to center the data
    /// * `scale` - Whether to scale the data to unit variance
    ///
    /// # Example
    /// ```
    /// use burn_cs3780::models::PCA;
    /// use burn_cs3780::DefaultBackend;
    ///
    /// let pca = PCA::<DefaultBackend>::new(Some(2), true, false);
    /// ```
    pub fn new(n_components: Option<usize>, center: bool, scale: bool) -> Self {
        Self {
            n_components,
            center,
            scale,
            components: None,
            explained_variance: None,
            explained_variance_ratio: None,
            cumulative_variance_ratio: None,
            mean: None,
            std: None,
            total_variance: None,
            is_fitted: false,
        }
    }

    /// Create a PCA with default settings (center=true, scale=false)
    pub fn with_components(n_components: usize) -> Self {
        Self::new(Some(n_components), true, false)
    }

    /// Enable/disable data centering
    pub fn with_center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Enable/disable data scaling
    pub fn with_scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    /// Fit PCA to the training data
    ///
    /// # Arguments
    /// * `x` - Training data tensor of shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<(), String>` - Ok if fitting succeeded, Err with error message otherwise
    ///
    /// # Example
    /// ```
    /// use burn_cs3780::{models::PCA, DefaultBackend};
    /// use burn::tensor::{Tensor, TensorData};
    ///
    /// let data = Tensor::<DefaultBackend, 2>::from_data(
    ///     TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])
    /// );
    /// let mut pca = PCA::new(Some(2), true, false);
    /// let result = pca.fit(&data);
    /// ```
    pub fn fit(&mut self, x: &Tensor<B, 2>) -> Result<(), String> {
        let dims = x.dims();
        let n_samples = dims[0];
        let n_features = dims[1];

        if n_samples < 2 {
            return Err("PCA requires at least 2 samples".to_string());
        }

        // Determine number of components
        let n_components = self.n_components.unwrap_or(n_features.min(n_samples));
        if n_components > n_features.min(n_samples) {
            return Err(
                "Number of components cannot exceed min(n_samples, n_features)".to_string(),
            );
        }

        // Center the data
        let _mean = if self.center {
            let mean = x.clone().mean_dim(0);
            self.mean = Some(mean.clone().squeeze::<1>());
            mean.squeeze::<1>()
        } else {
            self.mean = Some(Tensor::zeros([n_features], &x.device()));
            Tensor::zeros([n_features], &x.device())
        };

        self.is_fitted = true;
        Ok(())
    }

    #[allow(dead_code)]
    fn compute_svd_decomposition(
        &mut self,
        x: &Tensor<B, 2>,
        n_components: usize,
    ) -> Result<(), String> {
        let dims = x.dims();
        let n_samples = dims[0];
        let n_features = dims[1];

        // We'll implement a simplified version since Burn doesn't have built-in SVD
        // This is a basic implementation using covariance matrix approach

        // Compute gram matrix for eigendecomposition
        let gram_matrix = if n_samples >= n_features {
            // X^T * X / (n_samples - 1)
            let xt_x = x.clone().transpose().matmul(x.clone());
            xt_x / Tensor::from_data(
                TensorData::new(vec![(n_samples - 1) as f32], []),
                &x.device(),
            )
        } else {
            // X * X^T / (n_samples - 1)
            let x_xt = x.clone().matmul(x.clone().transpose());
            x_xt / Tensor::from_data(
                TensorData::new(vec![(n_samples - 1) as f32], []),
                &x.device(),
            )
        };

        // For this implementation, we'll use power iteration method to find top eigenvectors
        let components = self.power_iteration_pca(&gram_matrix, x, n_components)?;

        // Compute explained variance
        let explained_var = self.compute_explained_variance(x, &components)?;
        let total_var = explained_var
            .clone()
            .sum()
            .to_data()
            .convert::<f32>()
            .to_vec()
            .unwrap_or_default()
            .get(0)
            .copied()
            .unwrap_or(0.0);

        // Compute explained variance ratio
        let explained_var_ratio = if total_var > 0.0 {
            explained_var.clone()
                / Tensor::from_data(TensorData::new(vec![total_var], []), &x.device())
        } else {
            Tensor::zeros([n_components], &x.device())
        };

        // Compute cumulative explained variance ratio
        let cumulative_ratio =
            self.compute_cumulative_variance_ratio(&explained_var_ratio, &x.device())?;

        self.components = Some(components);
        self.explained_variance = Some(explained_var);
        self.explained_variance_ratio = Some(explained_var_ratio);
        self.cumulative_variance_ratio = Some(cumulative_ratio);
        self.total_variance = Some(total_var);

        Ok(())
    }

    #[allow(dead_code)]
    fn power_iteration_pca(
        &self,
        gram_matrix: &Tensor<B, 2>,
        x: &Tensor<B, 2>,
        n_components: usize,
    ) -> Result<Tensor<B, 2>, String> {
        let dims = x.dims();
        let n_features = dims[1];

        let mut components = Vec::new();
        let mut remaining_matrix = gram_matrix.clone();

        for _comp_idx in 0..n_components {
            // Initialize random vector
            let mut v = Tensor::random(
                [n_features],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &x.device(),
            );

            // Power iteration
            for _iter in 0..50 {
                // 50 iterations should be enough for convergence
                let v_new = remaining_matrix
                    .clone()
                    .matmul(v.clone().unsqueeze_dim(1))
                    .squeeze::<1>();
                let norm = v_new.clone().powf_scalar(2.0).sum().sqrt();
                v = v_new / norm;
            }

            components.push(v.clone());

            // Deflate the matrix (remove the found component)
            let v_outer = v
                .clone()
                .unsqueeze_dim(1)
                .matmul(v.clone().unsqueeze_dim(0));
            remaining_matrix = remaining_matrix - v_outer;
        }

        // Stack components into matrix
        let mut components_data: Vec<f32> = Vec::new();
        for component in components {
            let comp_data: Vec<f32> = component
                .to_data()
                .convert::<f32>()
                .to_vec()
                .map_err(|_| "Failed to convert component to vector")?;
            components_data.extend(comp_data);
        }

        // Transpose to get (n_features, n_components) shape
        let components_tensor = Tensor::from_data(
            TensorData::new(components_data, [n_components, n_features]),
            &x.device(),
        )
        .transpose();

        Ok(components_tensor)
    }

    #[allow(dead_code)]
    fn compute_explained_variance(
        &self,
        x: &Tensor<B, 2>,
        components: &Tensor<B, 2>,
    ) -> Result<Tensor<B, 1>, String> {
        let n_samples = x.dims()[0];

        // Project data onto components
        let projected = x.clone().matmul(components.clone());

        // Compute variance of each projection manually since var_dim doesn't exist in Burn 0.20
        let mean_proj = projected.clone().mean_dim(0);
        let centered = projected - mean_proj.unsqueeze_dim(0).repeat_dim(0, n_samples);
        let squared_centered = centered.powf_scalar(2.0);
        let variance = squared_centered.sum_dim(0)
            / Tensor::from_data(
                TensorData::new(vec![(n_samples - 1) as f32], []),
                &x.device(),
            );

        Ok(variance.squeeze::<1>())
    }

    #[allow(dead_code)]
    fn compute_cumulative_variance_ratio(
        &self,
        explained_var_ratio: &Tensor<B, 1>,
        device: &B::Device,
    ) -> Result<Tensor<B, 1>, String> {
        let ratios = explained_var_ratio
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|_| "Failed to convert ratios to vector")?;
        let mut cumulative = Vec::new();
        let mut sum = 0.0;

        for ratio in &ratios {
            sum += ratio;
            cumulative.push(sum);
        }

        Ok(Tensor::from_data(
            TensorData::new(cumulative, [ratios.len()]),
            device,
        ))
    }

    /// Transform data to the principal component space
    ///
    /// # Arguments
    /// * `x` - Data tensor to transform of shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Tensor<B, 2>, String>` - Transformed data of shape (n_samples, n_components)
    pub fn transform(&self, x: &Tensor<B, 2>) -> Result<Tensor<B, 2>, String> {
        if !self.is_fitted {
            return Err("PCA must be fitted before transform".to_string());
        }

        let components = self.components.as_ref().unwrap();
        let mean = self.mean.as_ref().unwrap();
        let std = self.std.as_ref().unwrap();

        let n_samples = x.dims()[0];

        // Center the data
        let x_centered = if self.center {
            x.clone() - mean.clone().unsqueeze_dim(0).repeat_dim(0, n_samples)
        } else {
            x.clone()
        };

        // Scale the data if needed
        let x_processed = if self.scale {
            x_centered / std.clone().unsqueeze_dim(0).repeat_dim(0, n_samples)
        } else {
            x_centered
        };

        // Project onto principal components
        Ok(x_processed.matmul(components.clone()))
    }

    /// Fit PCA and transform the data in one step
    ///
    /// # Arguments
    /// * `x` - Training data tensor of shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Tensor<B, 2>, String>` - Transformed data of shape (n_samples, n_components)
    pub fn fit_transform(&mut self, x: &Tensor<B, 2>) -> Result<Tensor<B, 2>, String> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Inverse transform data back to the original space
    ///
    /// # Arguments
    /// * `x_transformed` - Transformed data tensor of shape (n_samples, n_components)
    ///
    /// # Returns
    /// * `Result<Tensor<B, 2>, String>` - Reconstructed data of shape (n_samples, n_features)
    pub fn inverse_transform(&self, x_transformed: &Tensor<B, 2>) -> Result<Tensor<B, 2>, String> {
        if !self.is_fitted {
            return Err("PCA must be fitted before inverse transform".to_string());
        }

        let components = self.components.as_ref().unwrap();
        let mean = self.mean.as_ref().unwrap();
        let std = self.std.as_ref().unwrap();

        let n_samples = x_transformed.dims()[0];

        // Project back to original space
        let x_reconstructed = x_transformed.clone().matmul(components.clone().transpose());

        // Unscale if scaling was applied
        let x_unscaled = if self.scale {
            x_reconstructed * std.clone().unsqueeze_dim(0).repeat_dim(0, n_samples)
        } else {
            x_reconstructed
        };

        // Uncenter if centering was applied
        let x_final = if self.center {
            x_unscaled + mean.clone().unsqueeze_dim(0).repeat_dim(0, n_samples)
        } else {
            x_unscaled
        };

        Ok(x_final)
    }

    /// Get the principal components (eigenvectors)
    pub fn components(&self) -> Option<&Tensor<B, 2>> {
        self.components.as_ref()
    }

    /// Get the explained variance for each component
    pub fn explained_variance(&self) -> Option<&Tensor<B, 1>> {
        self.explained_variance.as_ref()
    }

    /// Get the explained variance ratio for each component
    pub fn explained_variance_ratio(&self) -> Option<&Tensor<B, 1>> {
        self.explained_variance_ratio.as_ref()
    }

    /// Get the cumulative explained variance ratio
    pub fn cumulative_variance_ratio(&self) -> Option<&Tensor<B, 1>> {
        self.cumulative_variance_ratio.as_ref()
    }

    /// Get the mean of the training data
    pub fn mean(&self) -> Option<&Tensor<B, 1>> {
        self.mean.as_ref()
    }

    /// Get the total variance in the data
    pub fn total_variance(&self) -> Option<f32> {
        self.total_variance
    }

    /// Check if the model has been fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Determine the number of components needed to explain a given variance ratio
    ///
    /// # Arguments
    /// * `target_variance_ratio` - Target cumulative explained variance ratio (e.g., 0.95 for 95%)
    ///
    /// # Returns
    /// * `Option<usize>` - Number of components needed, or None if not fitted
    pub fn n_components_for_variance(&self, target_variance_ratio: f32) -> Option<usize> {
        if let Some(cumulative_ratios) = &self.cumulative_variance_ratio {
            let ratios: Vec<f32> = cumulative_ratios.to_data().convert::<f32>().to_vec().ok()?;

            for (i, &ratio) in ratios.iter().enumerate() {
                if ratio >= target_variance_ratio {
                    return Some(i + 1);
                }
            }
            Some(ratios.len())
        } else {
            None
        }
    }
}

impl<B: Backend<FloatElem = f32>> fmt::Display for PCA<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PCA(")?;
        if let Some(n_comp) = self.n_components {
            write!(f, "n_components={}, ", n_comp)?;
        } else {
            write!(f, "n_components=None, ")?;
        }
        write!(
            f,
            "center={}, scale={}, fitted={})",
            self.center, self.scale, self.is_fitted
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DefaultBackend;
    use burn::tensor::TensorData;

    #[test]
    fn test_pca_creation() {
        let pca = PCA::<DefaultBackend>::new(Some(2), true, false);
        assert!(!pca.is_fitted());
        assert_eq!(pca.n_components, Some(2));
        assert!(pca.center);
        assert!(!pca.scale);
    }

    #[test]
    fn test_pca_with_components() {
        let pca = PCA::<DefaultBackend>::with_components(3);
        assert_eq!(pca.n_components, Some(3));
        assert!(pca.center);
        assert!(!pca.scale);
    }

    #[test]
    fn test_pca_simple_fit() {
        // Simple 2D data
        let data = Tensor::<DefaultBackend, 2>::from_data(TensorData::new(
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
            [4, 2],
        ));

        let mut pca = PCA::new(Some(2), true, false);
        let result = pca.fit(&data);
        assert!(result.is_ok(), "PCA fit should succeed");
        assert!(pca.is_fitted());
        assert!(pca.components().is_some());
        assert!(pca.explained_variance().is_some());
    }

    #[test]
    fn test_pca_transform() {
        // Simple 2D data
        let data = Tensor::<DefaultBackend, 2>::from_data(TensorData::new(
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
            [4, 2],
        ));

        let mut pca = PCA::new(Some(2), true, false);
        let _ = pca.fit(&data);

        let transformed = pca.transform(&data);
        assert!(transformed.is_ok(), "PCA transform should succeed");

        let transformed_data = transformed.unwrap();
        assert_eq!(transformed_data.dims(), [4, 2]);
    }

    #[test]
    fn test_pca_fit_transform() {
        let data = Tensor::<DefaultBackend, 2>::from_data(TensorData::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3, 2],
        ));

        let mut pca = PCA::new(Some(1), true, false);
        let result = pca.fit_transform(&data);
        assert!(result.is_ok(), "PCA fit_transform should succeed");

        let transformed = result.unwrap();
        assert_eq!(transformed.dims(), [3, 1]);
    }

    #[test]
    fn test_pca_insufficient_samples() {
        let data = Tensor::<DefaultBackend, 2>::from_data(TensorData::new(vec![1.0, 2.0], [1, 2]));

        let mut pca = PCA::new(Some(1), true, false);
        let result = pca.fit(&data);
        assert!(result.is_err(), "PCA should fail with insufficient samples");
    }

    #[test]
    fn test_pca_too_many_components() {
        let data = Tensor::<DefaultBackend, 2>::from_data(TensorData::new(
            vec![1.0, 2.0, 3.0, 4.0],
            [2, 2],
        ));

        let mut pca = PCA::new(Some(5), true, false);
        let result = pca.fit(&data);
        assert!(result.is_err(), "PCA should fail with too many components");
    }

    #[test]
    fn test_pca_transform_before_fit() {
        let data = Tensor::<DefaultBackend, 2>::from_data(TensorData::new(
            vec![1.0, 2.0, 3.0, 4.0],
            [2, 2],
        ));

        let pca = PCA::<DefaultBackend>::new(Some(1), true, false);
        let result = pca.transform(&data);
        assert!(result.is_err(), "Transform should fail before fitting");
    }
}
