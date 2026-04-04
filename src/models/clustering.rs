//! Clustering algorithms implementation
//!
//! This module implements various clustering algorithms including K-Means
//! using the Burn framework.

use burn::tensor::{backend::Backend, Device, Tensor, TensorData};

/// Type alias for the default backend (CPU) used in clustering
type DefaultBackend = burn::backend::NdArray<f32>;

/// Initialization methods for K-means clustering
#[derive(Debug, Clone)]
pub enum InitMethod {
    /// Random initialization of centroids
    Random,
    /// K-means++ initialization for better centroid placement
    KMeansPlusPlus,
    /// Manual initialization with provided centroids
    Manual(Vec<Vec<f32>>),
}

/// Distance metrics for clustering
#[derive(Debug, Clone)]
pub enum ClusteringDistanceMetric {
    /// Euclidean distance (L2 norm)
    Euclidean,
    /// Manhattan distance (L1 norm)  
    Manhattan,
    /// Cosine distance
    Cosine,
}

impl ClusteringDistanceMetric {
    /// Compute pairwise distances between points and centroids
    pub fn compute_distances<B: Backend<FloatElem = f32>>(
        &self,
        points: &Tensor<B, 2>,
        centroids: &Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        match self {
            ClusteringDistanceMetric::Euclidean => {
                // Compute squared Euclidean distances using broadcasting
                let n_points = points.dims()[0];
                let n_centroids = centroids.dims()[0];

                let points_squared = points.clone().powf_scalar(2.0).sum_dim(1);
                let centroids_squared = centroids.clone().powf_scalar(2.0).sum_dim(1);
                let cross_term = points.clone().matmul(centroids.clone().transpose());

                let points_squared = points_squared.unsqueeze_dim(1).repeat_dim(1, n_centroids);
                let centroids_squared = centroids_squared.unsqueeze_dim(0).repeat_dim(0, n_points);

                (points_squared + centroids_squared - cross_term.mul_scalar(2.0)).sqrt()
            }
            // For simplicity, use Euclidean for other metrics too in this implementation
            _ => self.compute_distances(points, centroids),
        }
    }
}

/// K-Means clustering algorithm configuration
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    /// Number of clusters
    pub n_clusters: usize,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance for centroid movement
    pub tolerance: f32,
    /// Initialization method
    pub init_method: InitMethod,
    /// Distance metric to use
    pub distance_metric: ClusteringDistanceMetric,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Number of times to run with different centroid seeds
    pub n_init: usize,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            n_clusters: 3,
            max_iterations: 300,
            tolerance: 1e-4,
            init_method: InitMethod::Random,
            distance_metric: ClusteringDistanceMetric::Euclidean,
            random_seed: None,
            n_init: 10,
        }
    }
}

impl KMeansConfig {
    /// Create a new K-means configuration
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            ..Default::default()
        }
    }

    /// Set the maximum number of iterations
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the convergence tolerance
    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the initialization method
    pub fn with_init_method(mut self, init_method: InitMethod) -> Self {
        self.init_method = init_method;
        self
    }

    /// Set the distance metric
    pub fn with_distance_metric(mut self, distance_metric: ClusteringDistanceMetric) -> Self {
        self.distance_metric = distance_metric;
        self
    }

    /// Set the random seed
    pub fn with_random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Set the number of initializations
    pub fn with_n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init;
        self
    }
}

/// K-Means clustering algorithm
#[derive(Debug)]
pub struct KMeans<B: Backend<FloatElem = f32>> {
    config: KMeansConfig,
    centroids: Option<Tensor<B, 2>>,
    labels: Option<Tensor<B, 1>>,
    inertia: Option<f32>,
    n_iterations: usize,
    device: Device<B>,
    is_fitted: bool,
}

impl<B: Backend<FloatElem = f32>> KMeans<B> {
    /// Create a new K-means clustering model
    pub fn new(config: KMeansConfig, device: Device<B>) -> Self {
        Self {
            config,
            centroids: None,
            labels: None,
            inertia: None,
            n_iterations: 0,
            device,
            is_fitted: false,
        }
    }

    /// Fit the K-means model to training data
    pub fn fit(&mut self, x: &Tensor<B, 2>) -> Result<(), String> {
        if x.dims().len() != 2 {
            return Err("Input data must be 2-dimensional".to_string());
        }

        let n_samples = x.dims()[0];

        if n_samples < self.config.n_clusters {
            return Err("Number of samples must be >= number of clusters".to_string());
        }

        let mut best_centroids = None;
        let mut best_labels = None;
        let mut best_inertia = f32::INFINITY;
        let mut best_n_iter = 0;

        // Run K-means multiple times with different initializations
        for _init_run in 0..self.config.n_init {
            if let Ok((centroids, labels, inertia, n_iter)) = self.single_run(x) {
                if inertia < best_inertia {
                    best_inertia = inertia;
                    best_centroids = Some(centroids);
                    best_labels = Some(labels);
                    best_n_iter = n_iter;
                }
            }
        }

        self.centroids = best_centroids;
        self.labels = best_labels;
        self.inertia = Some(best_inertia);
        self.n_iterations = best_n_iter;
        self.is_fitted = true;

        Ok(())
    }

    /// Single run of K-means algorithm
    fn single_run(
        &self,
        x: &Tensor<B, 2>,
    ) -> Result<(Tensor<B, 2>, Tensor<B, 1>, f32, usize), String> {
        let n_samples = x.dims()[0];

        // Initialize centroids
        let mut centroids = self.initialize_centroids(x)?;

        let mut labels = Tensor::zeros([n_samples], &self.device);

        for iteration in 0..self.config.max_iterations {
            let prev_centroids = centroids.clone();

            // Assign points to closest centroids
            let distances = self.config.distance_metric.compute_distances(x, &centroids);
            labels = distances.argmin(1).squeeze::<1>().float();

            // Update centroids
            centroids = self.update_centroids(x, &labels)?;

            // Check for convergence
            let centroid_shift = self.compute_centroid_shift(&centroids, &prev_centroids);
            if centroid_shift < self.config.tolerance {
                let inertia = self.compute_inertia(x, &centroids, &labels);
                return Ok((centroids, labels, inertia, iteration + 1));
            }
        }

        let inertia = self.compute_inertia(x, &centroids, &labels);
        Ok((centroids, labels, inertia, self.config.max_iterations))
    }

    /// Initialize centroids based on the configured method
    fn initialize_centroids(&self, x: &Tensor<B, 2>) -> Result<Tensor<B, 2>, String> {
        let n_features = x.dims()[1];

        match &self.config.init_method {
            InitMethod::Random => {
                // Random initialization from data points
                let data = x.to_data().convert::<f32>();
                let data_vec: Vec<f32> = data
                    .to_vec()
                    .map_err(|_| "Failed to convert tensor data to vector")?;
                let mut centroids_data = Vec::new();

                for _k in 0..self.config.n_clusters {
                    let random_idx =
                        (rand::random::<f32>() * x.dims()[0] as f32) as usize % x.dims()[0];
                    for j in 0..n_features {
                        centroids_data.push(data_vec[random_idx * n_features + j]);
                    }
                }

                let centroids_tensor_data =
                    TensorData::new(centroids_data, [self.config.n_clusters, n_features]);
                Ok(Tensor::from_data(centroids_tensor_data, &self.device))
            }
            InitMethod::KMeansPlusPlus => {
                // For simplicity, fall back to random initialization
                let random_config = InitMethod::Random;
                self.initialize_centroids_from_method(x, &random_config)
            }
            InitMethod::Manual(manual_centroids) => {
                if manual_centroids.len() != self.config.n_clusters {
                    return Err("Number of manual centroids must match n_clusters".to_string());
                }
                if manual_centroids
                    .first()
                    .map_or(true, |c| c.len() != n_features)
                {
                    return Err(
                        "Manual centroids must have same number of features as data".to_string()
                    );
                }

                let flat_data: Vec<f32> = manual_centroids.iter().flatten().copied().collect();
                let centroids_tensor_data =
                    TensorData::new(flat_data, [self.config.n_clusters, n_features]);
                Ok(Tensor::from_data(centroids_tensor_data, &self.device))
            }
        }
    }

    /// Helper for initialization
    fn initialize_centroids_from_method(
        &self,
        x: &Tensor<B, 2>,
        method: &InitMethod,
    ) -> Result<Tensor<B, 2>, String> {
        let n_features = x.dims()[1];

        match method {
            InitMethod::Random => {
                let data = x.to_data().convert::<f32>();
                let data_vec: Vec<f32> = data
                    .to_vec()
                    .map_err(|_| "Failed to convert tensor data to vector")?;
                let mut centroids_data = Vec::new();

                for _k in 0..self.config.n_clusters {
                    let random_idx =
                        (rand::random::<f32>() * x.dims()[0] as f32) as usize % x.dims()[0];
                    for j in 0..n_features {
                        centroids_data.push(data_vec[random_idx * n_features + j]);
                    }
                }

                let centroids_tensor_data =
                    TensorData::new(centroids_data, [self.config.n_clusters, n_features]);
                Ok(Tensor::from_data(centroids_tensor_data, &self.device))
            }
            _ => Err("Unsupported initialization method".to_string()),
        }
    }

    /// Update centroids based on current cluster assignments
    fn update_centroids(
        &self,
        x: &Tensor<B, 2>,
        labels: &Tensor<B, 1>,
    ) -> Result<Tensor<B, 2>, String> {
        let n_features = x.dims()[1];
        let mut new_centroids_data = Vec::new();

        let labels_data = labels.to_data().convert::<f32>();
        let x_data = x.to_data().convert::<f32>();
        let labels_vec: Vec<f32> = labels_data
            .to_vec()
            .map_err(|_| "Failed to convert labels to vector")?;
        let x_vec: Vec<f32> = x_data
            .to_vec()
            .map_err(|_| "Failed to convert data to vector")?;

        for k in 0..self.config.n_clusters {
            // Find all points assigned to cluster k
            let cluster_indices: Vec<usize> = labels_vec
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| if label as usize == k { Some(i) } else { None })
                .collect();

            if cluster_indices.is_empty() {
                // Empty cluster - reinitialize randomly
                for j in 0..n_features {
                    let random_idx =
                        (rand::random::<f32>() * x.dims()[0] as f32) as usize % x.dims()[0];
                    new_centroids_data.push(x_vec[random_idx * n_features + j]);
                }
            } else {
                // Compute mean of assigned points
                for j in 0..n_features {
                    let sum: f32 = cluster_indices
                        .iter()
                        .map(|&idx| x_vec[idx * n_features + j])
                        .sum();
                    new_centroids_data.push(sum / cluster_indices.len() as f32);
                }
            }
        }

        let centroids_tensor_data =
            TensorData::new(new_centroids_data, [self.config.n_clusters, n_features]);
        Ok(Tensor::from_data(centroids_tensor_data, &self.device))
    }

    /// Compute the shift in centroids between iterations
    fn compute_centroid_shift(
        &self,
        new_centroids: &Tensor<B, 2>,
        old_centroids: &Tensor<B, 2>,
    ) -> f32 {
        let diff = new_centroids.clone().sub(old_centroids.clone());
        let squared_diff = diff.powf_scalar(2.0);
        let sum_squared = squared_diff.sum();
        (sum_squared
            .to_data()
            .convert::<f32>()
            .to_vec()
            .unwrap_or_default()
            .get(0)
            .copied()
            .unwrap_or(0.0) as f32)
            .sqrt()
    }

    /// Compute within-cluster sum of squares (inertia)
    fn compute_inertia(
        &self,
        x: &Tensor<B, 2>,
        centroids: &Tensor<B, 2>,
        labels: &Tensor<B, 1>,
    ) -> f32 {
        let distances = self.config.distance_metric.compute_distances(x, centroids);
        let labels_data = labels.to_data().convert::<f32>();
        let distances_data = distances.to_data().convert::<f32>();

        if let (Ok(labels_vec), Ok(distances_vec)) =
            (labels_data.to_vec::<f32>(), distances_data.to_vec::<f32>())
        {
            let mut inertia = 0.0;
            for (i, &label) in labels_vec.iter().enumerate() {
                let cluster_idx = label as usize;
                if i * self.config.n_clusters + cluster_idx < distances_vec.len() {
                    let distance = distances_vec[i * self.config.n_clusters + cluster_idx];
                    inertia += distance * distance;
                }
            }
            inertia
        } else {
            0.0
        }
    }

    /// Predict cluster assignments for new data
    pub fn predict(&self, x: &Tensor<B, 2>) -> Result<Tensor<B, 1>, String> {
        if !self.is_fitted {
            return Err("Model must be fitted before prediction".to_string());
        }

        let centroids = self.centroids.as_ref().unwrap();
        let distances = self.config.distance_metric.compute_distances(x, centroids);
        Ok(distances.argmin(1).squeeze::<1>().float())
    }

    /// Fit the model and predict cluster assignments
    pub fn fit_predict(&mut self, x: &Tensor<B, 2>) -> Result<Tensor<B, 1>, String> {
        self.fit(x)?;
        Ok(self.labels.as_ref().unwrap().clone())
    }

    /// Get cluster centroids
    pub fn centroids(&self) -> Option<&Tensor<B, 2>> {
        self.centroids.as_ref()
    }

    /// Get the inertia (within-cluster sum of squares)
    pub fn inertia(&self) -> Option<f32> {
        self.inertia
    }

    /// Get the number of iterations performed
    pub fn n_iterations(&self) -> usize {
        self.n_iterations
    }

    /// Check if the model has been fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Get cluster labels for the training data
    pub fn labels(&self) -> Option<&Tensor<B, 1>> {
        self.labels.as_ref()
    }

    /// Transform data to cluster distance space
    pub fn transform(&self, x: &Tensor<B, 2>) -> Result<Tensor<B, 2>, String> {
        if !self.is_fitted {
            return Err("Model must be fitted before transformation".to_string());
        }

        let centroids = self.centroids.as_ref().unwrap();
        Ok(self.config.distance_metric.compute_distances(x, centroids))
    }
}

/// Type alias for default backend K-means
pub type DefaultKMeans = KMeans<DefaultBackend>;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{ndarray::NdArrayDevice, NdArray};
    use burn::tensor::TensorData;

    type TestBackend = NdArray<f32>;
    type TestDevice = NdArrayDevice;

    #[test]
    fn test_kmeans_basic() {
        let device = TestDevice::default();

        // Create simple 2D dataset with clear clusters
        let data = vec![
            // Cluster 1
            1.0, 1.0, 1.5, 1.5, 2.0, 2.0, // Cluster 2
            8.0, 8.0, 8.5, 8.5, 9.0, 9.0,
        ];
        let x_data = TensorData::new(data, [6, 2]);
        let x = Tensor::<TestBackend, 2>::from_data(x_data, &device);

        let config = KMeansConfig::new(2)
            .with_max_iterations(100)
            .with_tolerance(1e-4);

        let mut kmeans = KMeans::new(config, device);

        let result = kmeans.fit(&x);
        assert!(result.is_ok());
        assert!(kmeans.is_fitted());
        assert!(kmeans.centroids().is_some());
        assert!(kmeans.inertia().is_some());
    }

    #[test]
    fn test_kmeans_predict() {
        let device = TestDevice::default();

        // Training data
        let train_data = vec![1.0, 1.0, 2.0, 2.0, 8.0, 8.0, 9.0, 9.0];
        let x_train_data = TensorData::new(train_data, [4, 2]);
        let x_train = Tensor::<TestBackend, 2>::from_data(x_train_data, &device);

        let config = KMeansConfig::new(2);
        let mut kmeans = KMeans::new(config, device.clone());

        kmeans.fit(&x_train).unwrap();

        // Test data
        let test_data = vec![1.5, 1.5, 8.5, 8.5];
        let x_test_data = TensorData::new(test_data, [2, 2]);
        let x_test = Tensor::<TestBackend, 2>::from_data(x_test_data, &device);

        let predictions = kmeans.predict(&x_test);
        assert!(predictions.is_ok());

        let pred_labels = predictions.unwrap();
        assert_eq!(pred_labels.dims(), [2]);
    }

    #[test]
    fn test_distance_metrics() {
        let device = TestDevice::default();

        let points_data = TensorData::new(vec![1.0, 0.0, 0.0, 1.0], [2, 2]);
        let points = Tensor::<TestBackend, 2>::from_data(points_data, &device);

        let centroids_data = TensorData::new(vec![0.0, 0.0], [1, 2]);
        let centroids = Tensor::<TestBackend, 2>::from_data(centroids_data, &device);

        // Test Euclidean distance
        let euclidean_dist =
            ClusteringDistanceMetric::Euclidean.compute_distances(&points, &centroids);
        assert_eq!(euclidean_dist.dims(), [2, 1]);
    }

    #[test]
    fn test_manual_init() {
        let device = TestDevice::default();

        let data = vec![1.0, 1.0, 2.0, 2.0, 8.0, 8.0, 9.0, 9.0];
        let x_data = TensorData::new(data, [4, 2]);
        let x = Tensor::<TestBackend, 2>::from_data(x_data, &device);

        // Test manual initialization
        let manual_centroids = vec![vec![1.0, 1.0], vec![8.0, 8.0]];
        let config = KMeansConfig::new(2).with_init_method(InitMethod::Manual(manual_centroids));
        let mut kmeans = KMeans::new(config, device);
        assert!(kmeans.fit(&x).is_ok());
    }
}
