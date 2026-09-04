//! Transformer architecture implementation using Burn.
//!
//! This module implements the transformer architecture including:
//! - Multi-head attention mechanism
//! - Position encoding
//! - Transformer encoder and decoder blocks
//! - Full transformer models for sequence-to-sequence tasks

use burn::module::Module;
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
    LinearConfig,
};
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

impl<B: Backend> TransformerEncoderLayer<B> {
    /// Attention weights this layer would produce for `x`.
    ///
    /// Shape `[batch, heads, seq, seq]`; row `i` is how much position `i`
    /// attends to every position.
    pub fn attention_weights(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 4>>) -> Tensor<B, 4> {
        self.attention
            .get_attention_weights(x.clone(), x.clone(), x, mask)
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
    /// Token embedding table, `[vocab_size, d_model]`
    embedding: Embedding<B>,
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
            // Burn initializes embeddings at N(0, 1), but the forward pass
            // rescales by sqrt(d_model) as in the paper — which assumes
            // N(0, 1/d_model). Left at the default, attention scores land
            // around +/-30 and softmax is saturated before training starts,
            // so no gradient ever reaches the attention weights.
            embedding: EmbeddingConfig::new(config.vocab_size, config.d_model)
                .with_initializer(burn::module::Initializer::Normal {
                    mean: 0.0,
                    std: 1.0 / (config.d_model as f64).sqrt(),
                })
                .init(device),
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
        // Token embedding. This used to broadcast the raw id across every
        // channel and push it through a Linear, which made every token's
        // vector a scalar multiple of every other token's — the model could
        // only ever see ids as magnitudes, never as distinct symbols.
        let mut x = self.embedding.forward(input_ids.int());

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

    /// Attention weights from every layer, in order.
    ///
    /// Each entry is `[batch, heads, seq, seq]`. The stack is re-run rather
    /// than cached so `forward` stays allocation-free for training.
    pub fn attention_weights(
        &self,
        input_ids: Tensor<B, 2>,
        mask: Option<Tensor<B, 4>>,
    ) -> Vec<Tensor<B, 4>> {
        let mut x = self.embedding.forward(input_ids.int()) * (self.d_model as f32).sqrt();
        x = self.dropout.forward(self.position_encoding.forward(x));

        let mut weights = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            weights.push(layer.attention_weights(x.clone(), mask.clone()));
            x = layer.forward(x, mask.clone());
        }
        weights
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

    /// The underlying encoder, for attention visualization.
    pub fn encoder(&self) -> &TransformerEncoder<B> {
        &self.encoder
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
        let [batch_size, _, d_model] = encoder_output.dims();
        let cls_output = encoder_output
            .slice([0..batch_size, 0..1, 0..d_model])
            // reshape, not squeeze: a single-item batch would squeeze away
            // the batch axis too and leave nothing.
            .reshape([batch_size, d_model]);

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

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

impl<B: burn::tensor::backend::AutodiffBackend<FloatElem = f32>> TransformerClassifier<B> {
    /// Train the classifier on tokenized sequences.
    ///
    /// Mini-batch Adam over `[n_samples, seq_len]` token ids with integer
    /// class labels. Returns the trained model and the mean loss per epoch.
    ///
    /// # Arguments
    /// * `input_ids` - Token ids as floats, `[n_samples, seq_len]`
    /// * `y` - Class indices as floats, `[n_samples]`
    /// * `epochs` - Passes over the data
    /// * `lr` - Adam learning rate
    /// * `batch_size` - Sequences per gradient step
    pub fn train_classifier(
        mut self,
        input_ids: Tensor<B, 2>,
        y: Tensor<B, 1>,
        epochs: usize,
        lr: f64,
        batch_size: usize,
    ) -> (Self, Vec<f32>) {
        use burn::nn::loss::CrossEntropyLossConfig;
        use burn::optim::{AdamConfig, GradientsParams, Optimizer};

        let device = y.device();
        let n_samples = input_ids.dims()[0];
        let batch_size = batch_size.clamp(1, n_samples.max(1));
        let targets = y.int();
        let loss_fn = CrossEntropyLossConfig::new().init(&device);
        let mut optimizer = AdamConfig::new().init();
        let mut history = Vec::with_capacity(epochs);

        for _ in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut steps = 0;
            let mut start = 0;

            while start < n_samples {
                let end = (start + batch_size).min(n_samples);
                let batch = input_ids.clone().slice([start..end]);
                let batch_targets = targets.clone().slice([start..end]);

                let loss = loss_fn.forward(self.forward(batch, None), batch_targets);
                epoch_loss += loss.clone().into_scalar();
                steps += 1;

                let grads = GradientsParams::from_grads(loss.backward(), &self);
                self = optimizer.step(lr, self, grads);
                start = end;
            }

            history.push(if steps > 0 {
                epoch_loss / steps as f32
            } else {
                f32::NAN
            });
        }

        (self, history)
    }
}

// ---------------------------------------------------------------------------
// Character tokenizer
// ---------------------------------------------------------------------------

/// A minimal character vocabulary for the text demos.
///
/// Id 0 is the classification token the classifier reads its answer from,
/// id 1 is padding, and 2..28 are the letters a-z. Everything else is
/// dropped, which suits the word lists in [`crate::models::naive_bayes`].
pub struct CharTokenizer;

impl CharTokenizer {
    /// Reserved id for the leading `[CLS]` position.
    pub const CLS: f32 = 0.0;
    /// Reserved id for padding.
    pub const PAD: f32 = 1.0;
    /// Total vocabulary size: two reserved ids plus the alphabet.
    pub const VOCAB: usize = 28;

    /// Tokenize one word into `seq_len` ids, `[CLS]` first and padded after.
    ///
    /// Words longer than `seq_len - 1` are truncated.
    pub fn encode(word: &str, seq_len: usize) -> Vec<f32> {
        let mut ids = Vec::with_capacity(seq_len);
        ids.push(Self::CLS);
        for ch in word.to_lowercase().chars() {
            if ids.len() >= seq_len {
                break;
            }
            if ch.is_ascii_lowercase() {
                ids.push((ch as u8 - b'a') as f32 + 2.0);
            }
        }
        ids.resize(seq_len, Self::PAD);
        ids
    }

    /// The characters that survived tokenization, aligned to `encode`'s
    /// output so a page can label an attention matrix.
    pub fn tokens(word: &str, seq_len: usize) -> Vec<String> {
        let mut out = vec!["[CLS]".to_string()];
        for ch in word.to_lowercase().chars() {
            if out.len() >= seq_len {
                break;
            }
            if ch.is_ascii_lowercase() {
                out.push(ch.to_string());
            }
        }
        while out.len() < seq_len {
            out.push("·".to_string());
        }
        out
    }

    /// Tokenize a batch of words into a `[n_words, seq_len]` tensor.
    pub fn encode_batch<B: Backend<FloatElem = f32>>(
        words: &[String],
        seq_len: usize,
        device: &Device<B>,
    ) -> Tensor<B, 2> {
        let mut data = Vec::with_capacity(words.len() * seq_len);
        for word in words {
            data.extend(Self::encode(word, seq_len));
        }
        Tensor::from_data(
            burn::tensor::TensorData::new(data, [words.len(), seq_len]),
            device,
        )
    }
}

/// Generate strings for a letter-search task.
///
/// Half the strings contain `target` and half do not, at a random position
/// and with random length. Unlike the word lists this produces as many
/// examples as a model needs.
///
/// The task is deliberately one a single attention layer can solve: the only
/// way the `[CLS]` position can know the answer is to attend to the position
/// holding the target, so a trained head has to point at it. A bag-of-letters
/// model would also solve the task — the point is to watch *where* the head
/// looks, not to beat a baseline.
///
/// Returns the strings and their labels, 1.0 for "contains the target".
pub fn make_letter_search_words(
    n_samples: usize,
    target: char,
    seed: u64,
) -> (Vec<String>, Vec<f32>) {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let alphabet: Vec<char> = "abcdefghlmnoprstu"
        .chars()
        .filter(|&c| c != target)
        .collect();

    let mut words = Vec::with_capacity(n_samples);
    let mut labels = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let contains = i % 2 == 0;
        let length = rng.gen_range(4..=9);
        let mut chars: Vec<char> = (0..length)
            .map(|_| alphabet[rng.gen_range(0..alphabet.len())])
            .collect();

        if contains {
            let at = rng.gen_range(0..chars.len());
            chars[at] = target;
        }

        words.push(chars.into_iter().collect());
        labels.push(if contains { 1.0 } else { 0.0 });
    }

    (words, labels)
}

// ---------------------------------------------------------------------------
// Character-level language model
// ---------------------------------------------------------------------------

/// Causal attention mask, shaped `[1, 1, seq_len, seq_len]`.
///
/// `1.0` marks a position to hide and `0.0` one to keep, matching the
/// `scores + mask * -1e9` convention in [`MultiHeadAttention`]. Broadcasts
/// over batch and heads. Without it every position can see the whole
/// sequence, and next-character prediction is trivially solved by reading
/// the answer.
pub fn causal_mask<B: Backend>(seq_len: usize, device: &Device<B>) -> Tensor<B, 4> {
    let mut data = Vec::with_capacity(seq_len * seq_len);
    for query in 0..seq_len {
        for key in 0..seq_len {
            data.push(if key > query { 1.0 } else { 0.0 });
        }
    }
    Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(data, [seq_len, seq_len]),
        device,
    )
    .reshape([1, 1, seq_len, seq_len])
}

/// A transformer that predicts the next character at every position.
///
/// Same encoder as the classifier, with two differences: attention is masked
/// so a position can only see what came before it, and the head projects to
/// the vocabulary at every position rather than to classes at one.
#[derive(Module, Debug)]
pub struct TransformerLanguageModel<B: Backend> {
    encoder: TransformerEncoder<B>,
    head: Linear<B>,
    seq_len: usize,
}

impl<B: Backend> TransformerLanguageModel<B> {
    /// Build an untrained model over `config.vocab_size` symbols.
    pub fn new(config: &TransformerEncoderConfig, device: &Device<B>) -> Self {
        Self {
            encoder: TransformerEncoder::new(config, device),
            head: LinearConfig::new(config.d_model, config.vocab_size).init(device),
            seq_len: config.max_len,
        }
    }

    /// Next-token logits at every position, `[batch, seq_len, vocab_size]`.
    pub fn forward(&self, input_ids: Tensor<B, 2>) -> Tensor<B, 3> {
        let mask = causal_mask::<B>(input_ids.dims()[1], &input_ids.device());
        self.head
            .forward(self.encoder.forward(input_ids, Some(mask)))
    }

    /// Sequence length the model was built for.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }
}

impl<B: burn::tensor::backend::AutodiffBackend<FloatElem = f32>> TransformerLanguageModel<B> {
    /// Train to predict `targets` from `inputs`, both `[n_samples, seq_len]`.
    ///
    /// Loss is cross-entropy over every position at once — the causal mask is
    /// what makes that legitimate, since position `i` cannot see its own
    /// answer. Returns the trained model and the mean loss per epoch.
    pub fn train_lm(
        mut self,
        inputs: Tensor<B, 2>,
        targets: Tensor<B, 2>,
        epochs: usize,
        lr: f64,
        batch_size: usize,
    ) -> (Self, Vec<f32>) {
        use burn::nn::loss::CrossEntropyLossConfig;
        use burn::optim::{AdamConfig, GradientsParams, Optimizer};

        let device = inputs.device();
        let [n_samples, seq_len] = inputs.dims();
        let batch_size = batch_size.clamp(1, n_samples.max(1));
        let vocab = self.head.weight.dims()[1];
        let target_ids = targets.int();

        let loss_fn = CrossEntropyLossConfig::new().init(&device);
        let mut optimizer = AdamConfig::new().init();
        let mut history = Vec::with_capacity(epochs);

        for _ in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut steps = 0;
            let mut start = 0;

            while start < n_samples {
                let end = (start + batch_size).min(n_samples);
                let rows = end - start;

                let logits = self
                    .forward(inputs.clone().slice([start..end]))
                    .reshape([rows * seq_len, vocab]);
                let flat_targets = target_ids
                    .clone()
                    .slice([start..end])
                    .reshape([rows * seq_len]);

                let loss = loss_fn.forward(logits, flat_targets);
                epoch_loss += loss.clone().into_scalar();
                steps += 1;

                let grads = GradientsParams::from_grads(loss.backward(), &self);
                self = optimizer.step(lr, self, grads);
                start = end;
            }

            history.push(if steps > 0 {
                epoch_loss / steps as f32
            } else {
                f32::NAN
            });
        }

        (self, history)
    }
}

impl CharTokenizer {
    /// The next-character targets for `word`: the encoding shifted left, so
    /// position `i` of the input is asked to produce position `i + 1`.
    ///
    /// The final real character is asked to produce [`Self::PAD`], which is
    /// how the model learns where words end.
    pub fn encode_target(word: &str, seq_len: usize) -> Vec<f32> {
        let mut ids: Vec<f32> = Self::encode(word, seq_len).into_iter().skip(1).collect();
        ids.push(Self::PAD);
        ids
    }
}

/// Generate words in a synthetic language with vowel harmony.
///
/// Syllables are onset + vowel + optional coda, and every vowel in a word is
/// drawn from the same set — front (e, i) or back (a, o, u). That is a real
/// phenomenon (Turkish, Finnish) and a deliberately awkward one to learn: the
/// constraint holds across the whole word, so a model that only looks at the
/// previous character or two cannot enforce it. Attention can.
///
/// Unlike the built-in word lists this can produce as many distinct words as
/// a model needs, which is the difference between learning the rule and
/// memorizing the list.
pub fn make_harmony_words(n_words: usize, seed: u64) -> Vec<String> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    const ONSETS: [char; 11] = ['b', 'd', 'g', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't'];
    const CODAS: [char; 4] = ['n', 's', 'r', 'l'];
    const FRONT: [char; 2] = ['e', 'i'];
    const BACK: [char; 3] = ['a', 'o', 'u'];

    let mut words = Vec::with_capacity(n_words);
    while words.len() < n_words {
        let front = rng.gen_bool(0.5);
        let syllables = rng.gen_range(2..=3);
        let mut word = String::new();

        for i in 0..syllables {
            word.push(ONSETS[rng.gen_range(0..ONSETS.len())]);
            word.push(if front {
                FRONT[rng.gen_range(0..FRONT.len())]
            } else {
                BACK[rng.gen_range(0..BACK.len())]
            });
            // Codas only between syllables and at the end, never doubling up.
            if rng.gen_bool(0.35) || i == syllables - 1 && rng.gen_bool(0.25) {
                word.push(CODAS[rng.gen_range(0..CODAS.len())]);
            }
        }

        if !words.contains(&word) {
            words.push(word);
        }
    }

    words
}

/// Whether every vowel in `word` comes from the same harmony set.
///
/// The rule [`make_harmony_words`] generates by, and the one a generated
/// sample either respects or does not.
pub fn obeys_vowel_harmony(word: &str) -> bool {
    let mut front = false;
    let mut back = false;
    for ch in word.chars() {
        match ch {
            'e' | 'i' => front = true,
            'a' | 'o' | 'u' => back = true,
            _ => {}
        }
    }
    !(front && back)
}
