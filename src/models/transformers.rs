//! Transformer architecture implementation using Burn.
//!
//! This module implements the transformer architecture including:
//! - Multi-head attention mechanism
//! - Position encoding
//! - Transformer encoder and decoder blocks
//! - Full transformer models for sequence-to-sequence tasks

use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::Device;
use burn::tensor::{activation::softmax, backend::Backend, Tensor};
use std::fmt;

/// Multi-head attention configuration
#[derive(Debug, Clone)]
pub struct MultiHeadAttentionConfig {
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Dropout probability
    pub dropout: f64,
}

impl Default for MultiHeadAttentionConfig {
    fn default() -> Self {
        Self {
            d_model: 512,
            n_heads: 8,
            dropout: 0.1,
        }
    }
}

/// Multi-head attention module
#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    /// Query projection
    query: Linear<B>,
    /// Key projection
    key: Linear<B>,
    /// Value projection
    value: Linear<B>,
    /// Output projection
    output: Linear<B>,
    /// Dropout layer
    dropout: Dropout,
    /// Number of heads
    n_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Model dimension
    d_model: usize,
}

impl<B: Backend> MultiHeadAttention<B> {
    /// Create a new multi-head attention module
    pub fn new(config: &MultiHeadAttentionConfig, device: &Device<B>) -> Self {
        assert!(
            config.d_model % config.n_heads == 0,
            "d_model must be divisible by n_heads"
        );

        let head_dim = config.d_model / config.n_heads;

        Self {
            query: LinearConfig::new(config.d_model, config.d_model).init(device),
            key: LinearConfig::new(config.d_model, config.d_model).init(device),
            value: LinearConfig::new(config.d_model, config.d_model).init(device),
            output: LinearConfig::new(config.d_model, config.d_model).init(device),
            dropout: DropoutConfig::new(config.dropout).init(),
            n_heads: config.n_heads,
            head_dim,
            d_model: config.d_model,
        }
    }

    /// Forward pass for multi-head attention
    ///
    /// # Arguments
    /// * `query` - Query tensor of shape [batch_size, seq_len, d_model]
    /// * `key` - Key tensor of shape [batch_size, seq_len, d_model]
    /// * `value` - Value tensor of shape [batch_size, seq_len, d_model]
    /// * `mask` - Optional attention mask
    ///
    /// # Returns
    /// * Output tensor of shape [batch_size, seq_len, d_model]
    pub fn forward(
        &self,
        query: Tensor<B, 3>,
        key: Tensor<B, 3>,
        value: Tensor<B, 3>,
        mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let batch_size = query.dims()[0];
        let seq_len = query.dims()[1];

        // Apply linear projections
        let q = self.query.forward(query);
        let k = self.key.forward(key);
        let v = self.value.forward(value);

        // Reshape for multi-head attention: [batch, seq_len, n_heads, head_dim]
        let q = q.reshape([batch_size, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape([batch_size, seq_len, self.n_heads, self.head_dim]);
        let v = v.reshape([batch_size, seq_len, self.n_heads, self.head_dim]);

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // Apply scaled dot-product attention
        let attention_output = self.scaled_dot_product_attention(q, k, v, mask);

        // Transpose back to [batch, seq_len, n_heads, head_dim]
        let attention_output = attention_output.swap_dims(1, 2);

        // Reshape to [batch, seq_len, d_model]
        let attention_output = attention_output.reshape([batch_size, seq_len, self.d_model]);

        // Apply output projection
        self.output.forward(attention_output)
    }

    /// Scaled dot-product attention
    fn scaled_dot_product_attention(
        &self,
        query: Tensor<B, 4>,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
        mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 4> {
        // Compute attention scores: Q @ K^T / sqrt(d_k)
        let scores = query.matmul(key.transpose());
        let scale = (self.head_dim as f32).sqrt();
        let scores = scores / scale;

        // Apply mask if provided
        let scores = if let Some(mask) = mask {
            scores + mask * -1e9 // Large negative value for masked positions
        } else {
            scores
        };

        // Apply softmax
        let attention_weights = softmax(scores, 3);

        // Apply dropout
        let attention_weights = self.dropout.forward(attention_weights);

        // Apply attention to values
        attention_weights.matmul(value)
    }
}

/// Position encoding configuration
#[derive(Debug, Clone)]
pub struct PositionEncodingConfig {
    /// Model dimension
    pub d_model: usize,
    /// Maximum sequence length
    pub max_len: usize,
    /// Dropout probability
    pub dropout: f64,
}

impl Default for PositionEncodingConfig {
    fn default() -> Self {
        Self {
            d_model: 512,
            max_len: 5000,
            dropout: 0.1,
        }
    }
}

/// Position encoding module
#[derive(Module, Debug)]
pub struct PositionEncoding<B: Backend> {
    /// Precomputed position encodings
    pe: Tensor<B, 2>,
    /// Dropout layer
    dropout: Dropout,
    /// Model dimension
    d_model: usize,
}

impl<B: Backend> PositionEncoding<B> {
    /// Create a new position encoding module
    pub fn new(config: &PositionEncodingConfig, device: &Device<B>) -> Self {
        let pe = Self::compute_position_encoding(config.max_len, config.d_model, device);

        Self {
            pe,
            dropout: DropoutConfig::new(config.dropout).init(),
            d_model: config.d_model,
        }
    }

    /// Forward pass for position encoding
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch_size, seq_len, d_model]
    ///
    /// # Returns
    /// * Output tensor with position encodings added
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let seq_len = x.dims()[1];
        let pe_slice = self.pe.clone().slice([0..seq_len, 0..self.d_model]);
        let pe_broadcasted = pe_slice.unsqueeze_dim::<3>(0).expand(x.shape());

        let output = x + pe_broadcasted;
        self.dropout.forward(output)
    }

    /// Compute position encoding matrix
    fn compute_position_encoding(
        max_len: usize,
        d_model: usize,
        device: &Device<B>,
    ) -> Tensor<B, 2> {
        let mut pe_data = Vec::with_capacity(max_len * d_model);

        for pos in 0..max_len {
            for i in 0..d_model {
                let angle = pos as f32 / 10000_f32.powf((2 * (i / 2)) as f32 / d_model as f32);
                if i % 2 == 0 {
                    pe_data.push(angle.sin());
                } else {
                    pe_data.push(angle.cos());
                }
            }
        }

        Tensor::from_data(
            burn::tensor::TensorData::new(pe_data, [max_len, d_model]),
            device,
        )
    }
}

/// Transformer encoder layer configuration
#[derive(Debug, Clone)]
pub struct TransformerEncoderLayerConfig {
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Feed-forward dimension
    pub d_ff: usize,
    /// Dropout probability
    pub dropout: f64,
}

impl Default for TransformerEncoderLayerConfig {
    fn default() -> Self {
        Self {
            d_model: 512,
            n_heads: 8,
            d_ff: 2048,
            dropout: 0.1,
        }
    }
}

/// Transformer encoder layer
#[derive(Module, Debug)]
pub struct TransformerEncoderLayer<B: Backend> {
    /// Multi-head attention
    attention: MultiHeadAttention<B>,
    /// Feed-forward network
    feed_forward: FeedForward<B>,
    /// Layer normalization for attention
    norm1: LayerNorm<B>,
    /// Layer normalization for feed-forward
    norm2: LayerNorm<B>,
    /// Dropout
    dropout: Dropout,
}

impl<B: Backend> TransformerEncoderLayer<B> {
    /// Create a new transformer encoder layer
    pub fn new(config: &TransformerEncoderLayerConfig, device: &Device<B>) -> Self {
        let attention_config = MultiHeadAttentionConfig {
            d_model: config.d_model,
            n_heads: config.n_heads,
            dropout: config.dropout,
        };

        let ff_config = FeedForwardConfig {
            d_model: config.d_model,
            d_ff: config.d_ff,
            dropout: config.dropout,
        };

        Self {
            attention: MultiHeadAttention::new(&attention_config, device),
            feed_forward: FeedForward::new(&ff_config, device),
            norm1: LayerNormConfig::new(config.d_model).init(device),
            norm2: LayerNormConfig::new(config.d_model).init(device),
            dropout: DropoutConfig::new(config.dropout).init(),
        }
    }

    /// Forward pass for transformer encoder layer
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch_size, seq_len, d_model]
    /// * `mask` - Optional attention mask
    ///
    /// # Returns
    /// * Output tensor of shape [batch_size, seq_len, d_model]
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 4>>) -> Tensor<B, 3> {
        // Multi-head attention with residual connection
        let attention_output = self
            .attention
            .forward(x.clone(), x.clone(), x.clone(), mask);
        let attention_output = self.dropout.forward(attention_output);
        let x = self.norm1.forward(x + attention_output);

        // Feed-forward with residual connection
        let ff_output = self.feed_forward.forward(x.clone());
        let ff_output = self.dropout.forward(ff_output);
        self.norm2.forward(x + ff_output)
    }
}

/// Feed-forward network configuration
#[derive(Debug, Clone)]
pub struct FeedForwardConfig {
    /// Model dimension
    pub d_model: usize,
    /// Feed-forward dimension
    pub d_ff: usize,
    /// Dropout probability
    pub dropout: f64,
}

/// Feed-forward network
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    /// First linear layer
    linear1: Linear<B>,
    /// Second linear layer
    linear2: Linear<B>,
    /// Dropout
    dropout: Dropout,
}

impl<B: Backend> FeedForward<B> {
    /// Create a new feed-forward network
    pub fn new(config: &FeedForwardConfig, device: &Device<B>) -> Self {
        Self {
            linear1: LinearConfig::new(config.d_model, config.d_ff).init(device),
            linear2: LinearConfig::new(config.d_ff, config.d_model).init(device),
            dropout: DropoutConfig::new(config.dropout).init(),
        }
    }

    /// Forward pass for feed-forward network
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear1.forward(x);
        let x = burn::tensor::activation::relu(x); // ReLU activation
        let x = self.dropout.forward(x);
        self.linear2.forward(x)
    }
}

/// Transformer encoder configuration
#[derive(Debug, Clone)]
pub struct TransformerEncoderConfig {
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of encoder layers
    pub n_layers: usize,
    /// Feed-forward dimension
    pub d_ff: usize,
    /// Maximum sequence length
    pub max_len: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Dropout probability
    pub dropout: f64,
}

impl Default for TransformerEncoderConfig {
    fn default() -> Self {
        Self {
            d_model: 512,
            n_heads: 8,
            n_layers: 6,
            d_ff: 2048,
            max_len: 5000,
            vocab_size: 10000,
            dropout: 0.1,
        }
    }
}

/// Transformer encoder
#[derive(Module, Debug)]
pub struct TransformerEncoder<B: Backend> {
    /// Token embedding
    embedding: Linear<B>,
    /// Position encoding
    position_encoding: PositionEncoding<B>,
    /// Encoder layers
    layers: Vec<TransformerEncoderLayer<B>>,
    /// Output layer normalization
    norm: LayerNorm<B>,
    /// Dropout
    dropout: Dropout,
    /// Model dimension
    d_model: usize,
}

impl<B: Backend> TransformerEncoder<B> {
    /// Create a new transformer encoder
    pub fn new(config: &TransformerEncoderConfig, device: &Device<B>) -> Self {
        let layer_config = TransformerEncoderLayerConfig {
            d_model: config.d_model,
            n_heads: config.n_heads,
            d_ff: config.d_ff,
            dropout: config.dropout,
        };

        let pe_config = PositionEncodingConfig {
            d_model: config.d_model,
            max_len: config.max_len,
            dropout: config.dropout,
        };

        let mut layers = Vec::new();
        for _ in 0..config.n_layers {
            layers.push(TransformerEncoderLayer::new(&layer_config, device));
        }

        Self {
            embedding: LinearConfig::new(config.vocab_size, config.d_model).init(device),
            position_encoding: PositionEncoding::new(&pe_config, device),
            layers,
            norm: LayerNormConfig::new(config.d_model).init(device),
            dropout: DropoutConfig::new(config.dropout).init(),
            d_model: config.d_model,
        }
    }

    /// Forward pass for transformer encoder
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs of shape [batch_size, seq_len]
    /// * `mask` - Optional attention mask
    ///
    /// # Returns
    /// * Output tensor of shape [batch_size, seq_len, d_model]
    pub fn forward(&self, input_ids: Tensor<B, 2>, mask: Option<Tensor<B, 4>>) -> Tensor<B, 3> {
        // Convert input IDs to one-hot encoding for embedding
        let batch_size = input_ids.dims()[0];
        let seq_len = input_ids.dims()[1];

        // Simple embedding lookup (in practice, you'd use a proper embedding layer)
        let input_ids_3d =
            input_ids
                .unsqueeze_dim::<3>(2)
                .expand([batch_size, seq_len, self.d_model]);

        // Token embedding
        let mut x = self.embedding.forward(input_ids_3d);

        // Scale by sqrt(d_model)
        x = x * (self.d_model as f32).sqrt();

        // Add position encoding
        x = self.position_encoding.forward(x);

        // Apply dropout
        x = self.dropout.forward(x);

        // Pass through encoder layers
        for layer in &self.layers {
            x = layer.forward(x, mask.clone());
        }

        // Final layer normalization
        self.norm.forward(x)
    }
}

/// Transformer for sequence classification
#[derive(Module, Debug)]
pub struct TransformerClassifier<B: Backend> {
    /// Transformer encoder
    encoder: TransformerEncoder<B>,
    /// Classification head
    classifier: Linear<B>,
    /// Dropout for classification head
    dropout: Dropout,
}

impl<B: Backend> TransformerClassifier<B> {
    /// Create a new transformer classifier
    pub fn new(
        encoder_config: &TransformerEncoderConfig,
        num_classes: usize,
        device: &Device<B>,
    ) -> Self {
        Self {
            encoder: TransformerEncoder::new(encoder_config, device),
            classifier: LinearConfig::new(encoder_config.d_model, num_classes).init(device),
            dropout: DropoutConfig::new(encoder_config.dropout).init(),
        }
    }

    /// Forward pass for classification
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs of shape [batch_size, seq_len]
    /// * `mask` - Optional attention mask
    ///
    /// # Returns
    /// * Logits tensor of shape [batch_size, num_classes]
    pub fn forward(&self, input_ids: Tensor<B, 2>, mask: Option<Tensor<B, 4>>) -> Tensor<B, 2> {
        // Get encoder output
        let encoder_output = self.encoder.forward(input_ids, mask);

        // Use [CLS] token representation (first token) for classification
        let cls_output = encoder_output.clone().slice([
            0..encoder_output.dims()[0],
            0..1,
            0..encoder_output.dims()[2],
        ]);
        let cls_output = cls_output.squeeze::<2>();

        // Apply dropout and classification head
        let cls_output = self.dropout.forward(cls_output);
        self.classifier.forward(cls_output)
    }
}

/// Attention visualization utilities
impl<B: Backend> MultiHeadAttention<B> {
    /// Get attention weights for visualization
    ///
    /// This is useful for understanding what the model is attending to
    pub fn get_attention_weights(
        &self,
        query: Tensor<B, 3>,
        key: Tensor<B, 3>,
        _value: Tensor<B, 3>,
        mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 4> {
        let batch_size = query.dims()[0];
        let seq_len = query.dims()[1];

        // Apply linear projections
        let q = self.query.forward(query);
        let k = self.key.forward(key);

        // Reshape for multi-head attention
        let q = q.reshape([batch_size, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape([batch_size, seq_len, self.n_heads, self.head_dim]);

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);

        // Compute attention scores
        let scores = q.matmul(k.transpose());
        let scale = (self.head_dim as f32).sqrt();
        let scores = scores / scale;

        // Apply mask if provided
        let scores = if let Some(mask) = mask {
            scores + mask * -1e9
        } else {
            scores
        };

        // Return attention weights without applying to values
        softmax(scores, 3)
    }
}

impl fmt::Display for TransformerEncoderConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TransformerEncoder(d_model={}, n_heads={}, n_layers={}, d_ff={}, vocab_size={})",
            self.d_model, self.n_heads, self.n_layers, self.d_ff, self.vocab_size
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DefaultBackend;
    use burn::tensor::Device;

    #[test]
    fn test_multihead_attention_creation() {
        let device = Device::<DefaultBackend>::default();
        let config = MultiHeadAttentionConfig::default();
        let _attention = MultiHeadAttention::<DefaultBackend>::new(&config, &device);
    }

    #[test]
    fn test_position_encoding_creation() {
        let device = Device::<DefaultBackend>::default();
        let config = PositionEncodingConfig::default();
        let _pe = PositionEncoding::<DefaultBackend>::new(&config, &device);
    }

    #[test]
    fn test_transformer_encoder_creation() {
        let device = Device::<DefaultBackend>::default();
        let config = TransformerEncoderConfig::default();
        let _encoder = TransformerEncoder::<DefaultBackend>::new(&config, &device);
    }

    #[test]
    fn test_transformer_classifier_creation() {
        let device = Device::<DefaultBackend>::default();
        let config = TransformerEncoderConfig::default();
        let _classifier = TransformerClassifier::<DefaultBackend>::new(&config, 10, &device);
    }

    #[test]
    fn test_multihead_attention_forward() {
        let device = Device::<DefaultBackend>::default();
        let config = MultiHeadAttentionConfig {
            d_model: 64,
            n_heads: 4,
            dropout: 0.0, // No dropout for testing
        };

        let attention = MultiHeadAttention::<DefaultBackend>::new(&config, &device);

        // Create test input
        let batch_size = 2;
        let seq_len = 10;
        let input = Tensor::<DefaultBackend, 3>::random(
            [batch_size, seq_len, config.d_model],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = attention.forward(input.clone(), input.clone(), input, None);
        assert_eq!(output.dims(), [batch_size, seq_len, config.d_model]);
    }

    #[test]
    fn test_transformer_encoder_layer_forward() {
        let device = Device::<DefaultBackend>::default();
        let config = TransformerEncoderLayerConfig {
            d_model: 64,
            n_heads: 4,
            d_ff: 128,
            dropout: 0.0,
        };

        let layer = TransformerEncoderLayer::<DefaultBackend>::new(&config, &device);

        let batch_size = 2;
        let seq_len = 10;
        let input = Tensor::<DefaultBackend, 3>::random(
            [batch_size, seq_len, config.d_model],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = layer.forward(input, None);
        assert_eq!(output.dims(), [batch_size, seq_len, config.d_model]);
    }
}
