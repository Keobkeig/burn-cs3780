//! Autoencoder implementations for unsupervised learning using Burn.
//!
//! This module implements various types of autoencoders:
//! - Standard Autoencoders for dimensionality reduction and feature learning
//! - Variational Autoencoders (VAE) for generative modeling
//! - Denoising Autoencoders for robust feature learning
//! - Sparse Autoencoders with L1 regularization

use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::{backend::Backend, Device, Tensor, TensorData};
use std::fmt;

/// Activation function types for autoencoders
#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    /// ReLU activation
    Relu,
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
    /// Linear activation (identity)
    Linear,
}

/// Configuration for standard autoencoder
#[derive(Debug, Clone)]
pub struct AutoencoderConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden layer dimensions (encoder layers)
    pub hidden_dims: Vec<usize>,
    /// Latent/bottleneck dimension
    pub latent_dim: usize,
    /// Activation function for hidden layers
    pub activation: ActivationType,
    /// Dropout rate (0.0 = no dropout)
    pub dropout_rate: f32,
    /// Whether to use batch normalization
    pub use_batch_norm: bool,
    /// Whether to tie encoder and decoder weights (decoder = encoder^T)
    pub tied_weights: bool,
}

impl Default for AutoencoderConfig {
    fn default() -> Self {
        Self {
            input_dim: 784, // Default for MNIST
            hidden_dims: vec![512, 256],
            latent_dim: 64,
            activation: ActivationType::Relu,
            dropout_rate: 0.1,
            use_batch_norm: false,
            tied_weights: false,
        }
    }
}

/// Standard Autoencoder for dimensionality reduction and feature learning
#[derive(Module, Debug)]
pub struct Autoencoder<B: Backend> {
    /// Encoder layers
    encoder_layers: Vec<Linear<B>>,
    /// Decoder layers
    decoder_layers: Vec<Linear<B>>,
    /// Dropout layer
    dropout: Dropout,
}

impl<B: Backend<FloatElem = f32>> Autoencoder<B> {
    /// Create a new autoencoder
    pub fn new(config: AutoencoderConfig, device: Device<B>) -> Self {
        let dropout = DropoutConfig::new(config.dropout_rate as f64).init();

        // Build encoder layers
        let mut encoder_layers = Vec::new();
        let mut input_size = config.input_dim;

        for &hidden_size in &config.hidden_dims {
            let layer = LinearConfig::new(input_size, hidden_size).init(&device);
            encoder_layers.push(layer);
            input_size = hidden_size;
        }

        // Add final encoder layer to latent space
        let encoder_final = LinearConfig::new(input_size, config.latent_dim).init(&device);
        encoder_layers.push(encoder_final);

        // Build decoder layers
        let mut decoder_layers = Vec::new();
        let mut layer_sizes = config.hidden_dims.clone();
        layer_sizes.reverse();
        layer_sizes.push(config.input_dim);

        input_size = config.latent_dim;
        for &output_size in &layer_sizes {
            let layer = LinearConfig::new(input_size, output_size).init(&device);
            decoder_layers.push(layer);
            input_size = output_size;
        }

        Self {
            encoder_layers,
            decoder_layers,
            dropout,
        }
    }

    /// Forward pass through the entire autoencoder
    pub fn forward(&self, input: Tensor<B, 2>, activation: ActivationType) -> Tensor<B, 2> {
        let latent = self.encode(input, activation);
        self.decode(latent, activation)
    }

    /// Encode input to latent representation
    pub fn encode(&self, mut input: Tensor<B, 2>, activation: ActivationType) -> Tensor<B, 2> {
        for (i, layer) in self.encoder_layers.iter().enumerate() {
            input = layer.forward(input);

            // Apply activation (except for the last layer which goes to latent space)
            if i < self.encoder_layers.len() - 1 {
                input = self.apply_activation(input, activation);
                input = self.dropout.forward(input);
            }
        }
        input
    }

    /// Decode latent representation to output
    pub fn decode(&self, mut latent: Tensor<B, 2>, activation: ActivationType) -> Tensor<B, 2> {
        for (i, layer) in self.decoder_layers.iter().enumerate() {
            latent = layer.forward(latent);

            // Apply activation (sigmoid for final layer, configured activation for others)
            if i < self.decoder_layers.len() - 1 {
                latent = self.apply_activation(latent, activation);
                latent = self.dropout.forward(latent);
            } else {
                // Final layer - use sigmoid for reconstruction
                latent = self.apply_sigmoid(latent);
            }
        }
        latent
    }

    /// Compute reconstruction loss (MSE)
    pub fn reconstruction_loss(&self, input: Tensor<B, 2>, output: Tensor<B, 2>) -> Tensor<B, 1> {
        let diff = input - output;
        let squared_diff = diff.clone() * diff;
        squared_diff.mean()
    }

    /// Apply activation function
    fn apply_activation(&self, tensor: Tensor<B, 2>, activation: ActivationType) -> Tensor<B, 2> {
        match activation {
            ActivationType::Relu => burn::tensor::activation::relu(tensor),
            ActivationType::Sigmoid => self.apply_sigmoid(tensor),
            ActivationType::Tanh => tensor.tanh(),
            ActivationType::Linear => tensor,
        }
    }

    /// Apply sigmoid activation
    fn apply_sigmoid(&self, tensor: Tensor<B, 2>) -> Tensor<B, 2> {
        // Sigmoid: 1 / (1 + exp(-x))
        let neg_tensor = tensor.neg();
        let exp_tensor = neg_tensor.exp();
        let one_plus_exp = exp_tensor + 1.0;
        one_plus_exp.recip()
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        self.encoder_layers.len()
    }
}

/// Configuration for Variational Autoencoder (VAE)
#[derive(Debug, Clone)]
pub struct VAEConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden layer dimensions (encoder layers)
    pub hidden_dims: Vec<usize>,
    /// Latent dimension
    pub latent_dim: usize,
    /// Beta parameter for KL divergence weighting
    pub beta: f32,
    /// Activation function
    pub activation: ActivationType,
    /// Dropout rate
    pub dropout_rate: f32,
}

impl Default for VAEConfig {
    fn default() -> Self {
        Self {
            input_dim: 784,
            hidden_dims: vec![512, 256],
            latent_dim: 64,
            beta: 1.0,
            activation: ActivationType::Relu,
            dropout_rate: 0.1,
        }
    }
}

/// Variational Autoencoder for generative modeling
#[derive(Module, Debug)]
pub struct VariationalAutoencoder<B: Backend> {
    /// Encoder layers (shared)
    encoder_layers: Vec<Linear<B>>,
    /// Mean projection layer
    mu_layer: Linear<B>,
    /// Log variance projection layer
    logvar_layer: Linear<B>,
    /// Decoder layers
    decoder_layers: Vec<Linear<B>>,
    /// Dropout layer
    dropout: Dropout,
}

impl<B: Backend<FloatElem = f32>> VariationalAutoencoder<B> {
    /// Create a new VAE
    pub fn new(config: VAEConfig, device: Device<B>) -> Self {
        let dropout = DropoutConfig::new(config.dropout_rate as f64).init();

        // Build encoder layers (up to the latent projection)
        let mut encoder_layers = Vec::new();
        let mut input_size = config.input_dim;

        for &hidden_size in &config.hidden_dims {
            let layer = LinearConfig::new(input_size, hidden_size).init(&device);
            encoder_layers.push(layer);
            input_size = hidden_size;
        }

        // Mean and logvar projection layers
        let mu_layer = LinearConfig::new(input_size, config.latent_dim).init(&device);
        let logvar_layer = LinearConfig::new(input_size, config.latent_dim).init(&device);

        // Build decoder layers
        let mut decoder_layers = Vec::new();
        let mut layer_sizes = config.hidden_dims.clone();
        layer_sizes.reverse();
        layer_sizes.push(config.input_dim);

        input_size = config.latent_dim;
        for &output_size in &layer_sizes {
            let layer = LinearConfig::new(input_size, output_size).init(&device);
            decoder_layers.push(layer);
            input_size = output_size;
        }

        Self {
            encoder_layers,
            mu_layer,
            logvar_layer,
            decoder_layers,
            dropout,
        }
    }

    /// Forward pass through VAE
    pub fn forward(
        &self,
        input: Tensor<B, 2>,
        activation: ActivationType,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let (mu, logvar) = self.encode(input, activation);
        let z = self.reparameterize(mu.clone(), logvar.clone());
        let reconstruction = self.decode(z, activation);
        (reconstruction, mu, logvar)
    }

    /// Encode input to latent distribution parameters
    pub fn encode(
        &self,
        mut input: Tensor<B, 2>,
        activation: ActivationType,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Pass through encoder layers
        for layer in &self.encoder_layers {
            input = layer.forward(input);
            input = self.apply_activation(input, activation);
            input = self.dropout.forward(input);
        }

        // Project to mean and log variance
        let mu = self.mu_layer.forward(input.clone());
        let logvar = self.logvar_layer.forward(input);

        (mu, logvar)
    }

    /// Reparameterization trick: z = μ + σ * ε where ε ~ N(0,1)
    pub fn reparameterize(&self, mu: Tensor<B, 2>, logvar: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch_size, latent_dim] = mu.dims();

        // Sample epsilon from standard normal
        let epsilon_data: Vec<f32> = (0..batch_size * latent_dim)
            .map(|_| {
                // Box-Muller transform for normal distribution
                let u1: f32 = rand::random::<f32>().max(1e-8);
                let u2: f32 = rand::random::<f32>();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            })
            .collect();

        let epsilon = Tensor::from_floats(
            TensorData::new(epsilon_data, [batch_size, latent_dim]),
            &mu.device(),
        );

        // σ = exp(0.5 * log_var) = sqrt(var)
        let std = (logvar * 0.5).exp();

        // z = μ + σ * ε
        mu + std * epsilon
    }

    /// Decode latent sample to reconstruction
    pub fn decode(&self, mut z: Tensor<B, 2>, activation: ActivationType) -> Tensor<B, 2> {
        for (i, layer) in self.decoder_layers.iter().enumerate() {
            z = layer.forward(z);

            if i < self.decoder_layers.len() - 1 {
                z = self.apply_activation(z, activation);
                z = self.dropout.forward(z);
            } else {
                // Final layer uses sigmoid
                z = self.apply_sigmoid(z);
            }
        }
        z
    }

    /// Compute VAE loss (reconstruction + KL divergence)
    pub fn vae_loss(
        &self,
        input: Tensor<B, 2>,
        reconstruction: Tensor<B, 2>,
        mu: Tensor<B, 2>,
        logvar: Tensor<B, 2>,
        beta: f32,
    ) -> Tensor<B, 1> {
        // Reconstruction loss (Binary Cross Entropy or MSE)
        let recon_loss = self.reconstruction_loss(input, reconstruction);

        // KL divergence: -0.5 * sum(1 + log_var - mu^2 - var)
        let mu_squared = mu.clone() * mu;
        let var = logvar.clone().exp();
        let kl_elements: Tensor<B, 2> = 1.0 + logvar - mu_squared - var;
        let kl_loss = kl_elements.sum() * -0.5;

        // Total loss with beta weighting
        recon_loss + kl_loss * beta
    }

    /// Reconstruction loss
    fn reconstruction_loss(&self, input: Tensor<B, 2>, output: Tensor<B, 2>) -> Tensor<B, 1> {
        let diff = input - output;
        let squared_diff = diff.clone() * diff;
        squared_diff.mean()
    }

    /// Sample from the latent space
    pub fn sample(
        &self,
        n_samples: usize,
        latent_dim: usize,
        device: &Device<B>,
        activation: ActivationType,
    ) -> Tensor<B, 2> {
        let latent_samples_data: Vec<f32> = (0..n_samples * latent_dim)
            .map(|_| {
                let u1: f32 = rand::random::<f32>().max(1e-8);
                let u2: f32 = rand::random::<f32>();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            })
            .collect();

        let latent_samples = Tensor::from_floats(
            TensorData::new(latent_samples_data, [n_samples, latent_dim]),
            device,
        );

        self.decode(latent_samples, activation)
    }

    /// Apply activation function
    fn apply_activation(&self, tensor: Tensor<B, 2>, activation: ActivationType) -> Tensor<B, 2> {
        match activation {
            ActivationType::Relu => burn::tensor::activation::relu(tensor),
            ActivationType::Sigmoid => self.apply_sigmoid(tensor),
            ActivationType::Tanh => tensor.tanh(),
            ActivationType::Linear => tensor,
        }
    }

    /// Apply sigmoid activation
    fn apply_sigmoid(&self, tensor: Tensor<B, 2>) -> Tensor<B, 2> {
        let neg_tensor = tensor.neg();
        let exp_tensor = neg_tensor.exp();
        let one_plus_exp = exp_tensor + 1.0;
        one_plus_exp.recip()
    }
}

/// Configuration for Denoising Autoencoder
#[derive(Debug, Clone)]
pub struct DenoisingAutoencoderConfig {
    /// Base autoencoder configuration
    pub base_config: AutoencoderConfig,
    /// Noise level (standard deviation for Gaussian noise)
    pub noise_level: f32,
    /// Type of noise to apply
    pub noise_type: NoiseType,
}

/// Types of noise for denoising autoencoders
#[derive(Debug, Clone, Copy)]
pub enum NoiseType {
    /// Gaussian noise
    Gaussian,
    /// Salt and pepper noise (random pixels set to 0 or 1)
    SaltPepper,
    /// Dropout noise (random pixels set to 0)
    Dropout,
}

impl Default for DenoisingAutoencoderConfig {
    fn default() -> Self {
        Self {
            base_config: AutoencoderConfig::default(),
            noise_level: 0.1,
            noise_type: NoiseType::Gaussian,
        }
    }
}

/// Denoising Autoencoder for robust feature learning
#[derive(Module, Debug)]
pub struct DenoisingAutoencoder<B: Backend> {
    /// Base autoencoder
    autoencoder: Autoencoder<B>,
}

impl<B: Backend<FloatElem = f32>> DenoisingAutoencoder<B> {
    /// Create a new denoising autoencoder
    pub fn new(config: DenoisingAutoencoderConfig, device: Device<B>) -> Self {
        let autoencoder = Autoencoder::new(config.base_config.clone(), device);

        Self { autoencoder }
    }

    /// Forward pass with noise injection during training
    pub fn forward_train(
        &self,
        input: Tensor<B, 2>,
        activation: ActivationType,
        noise_level: f32,
        noise_type: NoiseType,
    ) -> Tensor<B, 2> {
        let noisy_input = self.add_noise(input.clone(), noise_level, noise_type);
        self.autoencoder.forward(noisy_input, activation)
    }

    /// Forward pass without noise (for inference)
    pub fn forward(&self, input: Tensor<B, 2>, activation: ActivationType) -> Tensor<B, 2> {
        self.autoencoder.forward(input, activation)
    }

    /// Encode clean input to latent space
    pub fn encode(&self, input: Tensor<B, 2>, activation: ActivationType) -> Tensor<B, 2> {
        self.autoencoder.encode(input, activation)
    }

    /// Decode latent representation
    pub fn decode(&self, latent: Tensor<B, 2>, activation: ActivationType) -> Tensor<B, 2> {
        self.autoencoder.decode(latent, activation)
    }

    /// Add noise to input based on configuration
    pub fn add_noise(
        &self,
        input: Tensor<B, 2>,
        noise_level: f32,
        noise_type: NoiseType,
    ) -> Tensor<B, 2> {
        let [batch_size, input_dim] = input.dims();

        match noise_type {
            NoiseType::Gaussian => {
                // Add Gaussian noise
                let noise_data: Vec<f32> = (0..batch_size * input_dim)
                    .map(|_| {
                        let u1: f32 = rand::random::<f32>().max(1e-8);
                        let u2: f32 = rand::random::<f32>();
                        let noise =
                            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                        noise * noise_level
                    })
                    .collect();

                let noise = Tensor::from_floats(
                    TensorData::new(noise_data, [batch_size, input_dim]),
                    &input.device(),
                );

                input + noise
            }
            NoiseType::SaltPepper => {
                // Salt and pepper noise
                let input_data = input
                    .to_data()
                    .convert::<f32>()
                    .to_vec::<f32>()
                    .expect("Failed to convert input to vector");

                let noisy_data: Vec<f32> = input_data
                    .into_iter()
                    .map(|x| {
                        if rand::random::<f32>() < noise_level {
                            if rand::random::<f32>() < 0.5 {
                                0.0
                            } else {
                                1.0
                            }
                        } else {
                            x
                        }
                    })
                    .collect();

                Tensor::from_floats(
                    TensorData::new(noisy_data, [batch_size, input_dim]),
                    &input.device(),
                )
            }
            NoiseType::Dropout => {
                // Dropout noise
                let input_data = input
                    .to_data()
                    .convert::<f32>()
                    .to_vec::<f32>()
                    .expect("Failed to convert input to vector");

                let noisy_data: Vec<f32> = input_data
                    .into_iter()
                    .map(|x| {
                        if rand::random::<f32>() < noise_level {
                            0.0
                        } else {
                            x
                        }
                    })
                    .collect();

                Tensor::from_floats(
                    TensorData::new(noisy_data, [batch_size, input_dim]),
                    &input.device(),
                )
            }
        }
    }

    /// Compute denoising loss (reconstruction loss between clean input and noisy reconstruction)
    pub fn denoising_loss(
        &self,
        clean_input: Tensor<B, 2>,
        reconstruction: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        self.autoencoder
            .reconstruction_loss(clean_input, reconstruction)
    }
}

/// Configuration for Sparse Autoencoder
#[derive(Debug, Clone)]
pub struct SparseAutoencoderConfig {
    /// Base autoencoder configuration
    pub base_config: AutoencoderConfig,
    /// Sparsity regularization weight
    pub sparsity_weight: f32,
    /// Target activation level (sparsity parameter ρ)
    pub sparsity_target: f32,
    /// Beta parameter for KL divergence in sparsity constraint
    pub sparsity_beta: f32,
}

impl Default for SparseAutoencoderConfig {
    fn default() -> Self {
        Self {
            base_config: AutoencoderConfig::default(),
            sparsity_weight: 0.01,
            sparsity_target: 0.05, // 5% activation
            sparsity_beta: 3.0,
        }
    }
}

/// Sparse Autoencoder with L1 regularization and sparsity constraints
#[derive(Module, Debug)]
pub struct SparseAutoencoder<B: Backend> {
    /// Base autoencoder
    autoencoder: Autoencoder<B>,
}

impl<B: Backend<FloatElem = f32>> SparseAutoencoder<B> {
    /// Create a new sparse autoencoder
    pub fn new(config: SparseAutoencoderConfig, device: Device<B>) -> Self {
        let autoencoder = Autoencoder::new(config.base_config.clone(), device);

        Self { autoencoder }
    }

    /// Forward pass
    pub fn forward(&self, input: Tensor<B, 2>, activation: ActivationType) -> Tensor<B, 2> {
        self.autoencoder.forward(input, activation)
    }

    /// Encode to latent space with activation tracking
    pub fn encode(
        &self,
        input: Tensor<B, 2>,
        activation: ActivationType,
    ) -> (Tensor<B, 2>, Vec<Tensor<B, 2>>) {
        // For now, use the standard encode and return empty activations
        // This is a limitation of the current structure - we'd need to refactor
        // the base autoencoder to expose intermediate activations
        let latent = self.autoencoder.encode(input, activation);
        let activations = Vec::new(); // Empty for now
        (latent, activations)
    }

    /// Decode from latent space
    pub fn decode(&self, latent: Tensor<B, 2>, activation: ActivationType) -> Tensor<B, 2> {
        self.autoencoder.decode(latent, activation)
    }

    /// Compute sparse autoencoder loss (reconstruction + sparsity)
    pub fn sparse_loss(
        &self,
        input: Tensor<B, 2>,
        reconstruction: Tensor<B, 2>,
        activations: &[Tensor<B, 2>],
        sparsity_weight: f32,
        _sparsity_target: f32,
    ) -> Tensor<B, 1> {
        // Reconstruction loss
        let recon_loss = self
            .autoencoder
            .reconstruction_loss(input, reconstruction.clone());

        // Sparsity loss (KL divergence between average activation and target)
        let mut sparsity_loss =
            Tensor::from_floats(TensorData::new(vec![0.0], [1]), &reconstruction.device());

        for activation in activations {
            // Average activation across batch
            let avg_activation = activation.clone().mean_dim(0);

            // Simple L1 sparsity penalty instead of KL divergence for now
            let sparsity_penalty = avg_activation.abs().mean();
            sparsity_loss = sparsity_loss + sparsity_penalty;
        }

        // Total loss
        recon_loss + sparsity_loss * sparsity_weight
    }

    /// Get sparsity statistics for monitoring
    pub fn sparsity_stats(&self, activations: &[Tensor<B, 2>]) -> Vec<f32> {
        activations
            .iter()
            .map(|activation| {
                let avg_activation = activation.clone().mean();
                avg_activation
                    .to_data()
                    .convert::<f32>()
                    .to_vec::<f32>()
                    .expect("Failed to convert sparsity stats")[0]
            })
            .collect()
    }
}

// Display implementations
impl fmt::Display for AutoencoderConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AutoencoderConfig(input_dim={}, hidden_dims={:?}, latent_dim={}, activation={:?})",
            self.input_dim, self.hidden_dims, self.latent_dim, self.activation
        )
    }
}

impl fmt::Display for VAEConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VAEConfig(input_dim={}, hidden_dims={:?}, latent_dim={}, beta={})",
            self.input_dim, self.hidden_dims, self.latent_dim, self.beta
        )
    }
}

impl fmt::Display for DenoisingAutoencoderConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DenoisingAutoencoderConfig(noise_level={}, noise_type={:?})",
            self.noise_level, self.noise_type
        )
    }
}

impl fmt::Display for SparseAutoencoderConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SparseAutoencoderConfig(sparsity_weight={}, sparsity_target={})",
            self.sparsity_weight, self.sparsity_target
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DefaultBackend;
    use burn::tensor::Device;

    type TestBackend = DefaultBackend;

    #[test]
    fn test_autoencoder_creation() {
        let device = Device::<TestBackend>::default();
        let config = AutoencoderConfig {
            input_dim: 100,
            hidden_dims: vec![50, 25],
            latent_dim: 10,
            ..Default::default()
        };

        let _autoencoder: Autoencoder<TestBackend> = Autoencoder::new(config, device);
        // Can't easily test dimensions without stored config, but creation should work
    }

    #[test]
    fn test_autoencoder_forward() {
        let device = Device::<TestBackend>::default();
        let config = AutoencoderConfig {
            input_dim: 10,
            hidden_dims: vec![5],
            latent_dim: 2,
            ..Default::default()
        };

        let autoencoder = Autoencoder::new(config.clone(), device.clone());
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let input = Tensor::from_floats(TensorData::new(input_data, [1, 10]), &device);

        let output = autoencoder.forward(input.clone(), config.activation);
        assert_eq!(output.dims(), [1, 10]);

        let latent = autoencoder.encode(input.clone(), config.activation);
        assert_eq!(latent.dims(), [1, 2]);

        let reconstructed = autoencoder.decode(latent, config.activation);
        assert_eq!(reconstructed.dims(), [1, 10]);
    }

    #[test]
    fn test_vae_creation() {
        let device = Device::<TestBackend>::default();
        let config = VAEConfig {
            input_dim: 100,
            hidden_dims: vec![50],
            latent_dim: 10,
            ..Default::default()
        };

        let _vae = VariationalAutoencoder::new(config, device);
    }

    #[test]
    fn test_vae_forward() {
        let device = Device::<TestBackend>::default();
        let config = VAEConfig {
            input_dim: 10,
            hidden_dims: vec![5],
            latent_dim: 2,
            ..Default::default()
        };

        let vae = VariationalAutoencoder::new(config.clone(), device.clone());
        let input_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let input = Tensor::from_floats(TensorData::new(input_data, [1, 10]), &device);

        let (reconstruction, mu, logvar) = vae.forward(input.clone(), config.activation);
        assert_eq!(reconstruction.dims(), [1, 10]);
        assert_eq!(mu.dims(), [1, 2]);
        assert_eq!(logvar.dims(), [1, 2]);

        let loss = vae.vae_loss(input, reconstruction, mu, logvar, config.beta);
        assert_eq!(loss.dims(), [1]);
    }

    #[test]
    fn test_denoising_autoencoder() {
        let device = Device::<TestBackend>::default();
        let config = DenoisingAutoencoderConfig {
            base_config: AutoencoderConfig {
                input_dim: 10,
                hidden_dims: vec![5],
                latent_dim: 2,
                ..Default::default()
            },
            noise_level: 0.1,
            noise_type: NoiseType::Gaussian,
        };

        let denoising_ae = DenoisingAutoencoder::new(config.clone(), device.clone());
        let input_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let input = Tensor::from_floats(TensorData::new(input_data, [1, 10]), &device);

        let noisy_input =
            denoising_ae.add_noise(input.clone(), config.noise_level, config.noise_type);
        assert_eq!(noisy_input.dims(), [1, 10]);

        let reconstruction = denoising_ae.forward_train(
            input.clone(),
            config.base_config.activation,
            config.noise_level,
            config.noise_type,
        );
        assert_eq!(reconstruction.dims(), [1, 10]);

        let loss = denoising_ae.denoising_loss(input, reconstruction);
        assert_eq!(loss.dims(), [1]);
    }

    #[test]
    fn test_sparse_autoencoder() {
        let device = Device::<TestBackend>::default();
        let config = SparseAutoencoderConfig {
            base_config: AutoencoderConfig {
                input_dim: 10,
                hidden_dims: vec![8],
                latent_dim: 3,
                ..Default::default()
            },
            sparsity_weight: 0.01,
            sparsity_target: 0.05,
            sparsity_beta: 3.0,
        };

        let sparse_ae = SparseAutoencoder::new(config, device.clone());
        let input_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let input = Tensor::from_floats(TensorData::new(input_data, [1, 10]), &device);

        let (latent, activations) = sparse_ae.encode(input.clone());
        assert_eq!(latent.dims(), [1, 3]);
        assert!(!activations.is_empty());

        let reconstruction = sparse_ae.decode(latent);
        assert_eq!(reconstruction.dims(), [1, 10]);

        let loss = sparse_ae.sparse_loss(input, reconstruction, &activations);
        assert_eq!(loss.dims(), [1]);

        let stats = sparse_ae.sparsity_stats(&activations);
        assert_eq!(stats.len(), activations.len());
    }

    #[test]
    fn test_different_noise_types() {
        let device = Device::<TestBackend>::default();
        let input_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let input = Tensor::from_floats(TensorData::new(input_data, [1, 10]), &device);

        for noise_type in [
            NoiseType::Gaussian,
            NoiseType::SaltPepper,
            NoiseType::Dropout,
        ] {
            let config = DenoisingAutoencoderConfig {
                base_config: AutoencoderConfig {
                    input_dim: 10,
                    hidden_dims: vec![5],
                    latent_dim: 2,
                    ..Default::default()
                },
                noise_level: 0.1,
                noise_type,
            };

            let denoising_ae = DenoisingAutoencoder::new(config, device.clone());
            let noisy_input = denoising_ae.add_noise(input.clone());
            assert_eq!(noisy_input.dims(), [1, 10]);
        }
    }
}
