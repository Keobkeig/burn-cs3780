//! Comprehensive Transformers Example
//!
//! This example demonstrates the use of Transformer models from the burn-cs3780 library
//! for sequence-to-sequence learning tasks. We'll show:
//!
//! 1. Basic transformer encoder usage
//! 2. Multi-head attention mechanisms
//! 3. Position encoding for sequence modeling
//! 4. Text classification with transformers
//! 5. Sequence generation capabilities

use burn::prelude::*;
use burn_cs3780::models::transformers::{
    MultiHeadAttention, MultiHeadAttentionConfig, PositionalEncoding, PositionalEncodingConfig,
    TransformerEncoder, TransformerEncoderConfig,
};
use burn_cs3780::DefaultBackend;

type MyBackend = DefaultBackend;

fn main() {
    println!("🤖 Burn CS3780 - Transformer Models Example");
    println!("============================================\n");

    // Run different transformer examples
    basic_transformer_encoder_example();
    multi_head_attention_example();
    positional_encoding_example();
    text_classification_example();
    sequence_modeling_example();
}

/// Demonstrates basic transformer encoder usage
fn basic_transformer_encoder_example() {
    println!("📝 1. Basic Transformer Encoder");
    println!("-------------------------------");

    let device = Device::<MyBackend>::default();

    // Create transformer encoder configuration
    let config = TransformerEncoderConfig {
        input_dim: 512,
        hidden_dim: 2048,
        num_heads: 8,
        num_layers: 6,
        dropout_rate: 0.1,
        max_seq_length: 1000,
    };

    println!("Creating transformer encoder with config:");
    println!("  - Input dimension: {}", config.input_dim);
    println!("  - Hidden dimension: {}", config.hidden_dim);
    println!("  - Number of heads: {}", config.num_heads);
    println!("  - Number of layers: {}", config.num_layers);
    println!("  - Dropout rate: {}", config.dropout_rate);
    println!("  - Max sequence length: {}", config.max_seq_length);

    let transformer = TransformerEncoder::new(config, device.clone());

    // Create sample input sequence (batch_size=2, seq_length=10, input_dim=512)
    let batch_size = 2;
    let seq_length = 10;
    let input_dim = 512;

    let input_data: Vec<f32> = (0..batch_size * seq_length * input_dim)
        .map(|i| (i as f32) * 0.01)
        .collect();

    let input = Tensor::from_floats(
        TensorData::new(input_data, [batch_size, seq_length, input_dim]),
        &device,
    );

    println!("\nInput shape: {:?}", input.dims());

    // Forward pass through transformer
    let output = transformer.forward(input.clone());
    println!("Output shape: {:?}", output.dims());

    // Demonstrate attention weights extraction
    let attention_weights = transformer.get_attention_weights(input);
    println!("Attention weights shape: {:?}", attention_weights.dims());

    println!("✅ Basic transformer encoder example completed!\n");
}

/// Demonstrates multi-head attention mechanism
fn multi_head_attention_example() {
    println!("🔍 2. Multi-Head Attention Mechanism");
    println!("------------------------------------");

    let device = Device::<MyBackend>::default();

    // Create multi-head attention configuration
    let config = MultiHeadAttentionConfig {
        input_dim: 256,
        num_heads: 8,
        dropout_rate: 0.1,
    };

    println!("Creating multi-head attention with:");
    println!("  - Input dimension: {}", config.input_dim);
    println!("  - Number of heads: {}", config.num_heads);
    println!("  - Dropout rate: {}", config.dropout_rate);

    let attention = MultiHeadAttention::new(config, device.clone());

    // Create query, key, value tensors
    let batch_size = 4;
    let seq_length = 12;
    let input_dim = 256;

    let query_data: Vec<f32> = (0..batch_size * seq_length * input_dim)
        .map(|i| (i as f32) * 0.001)
        .collect();
    let key_data: Vec<f32> = (0..batch_size * seq_length * input_dim)
        .map(|i| (i as f32 + 1000.0) * 0.001)
        .collect();
    let value_data: Vec<f32> = (0..batch_size * seq_length * input_dim)
        .map(|i| (i as f32 + 2000.0) * 0.001)
        .collect();

    let query = Tensor::from_floats(
        TensorData::new(query_data, [batch_size, seq_length, input_dim]),
        &device,
    );
    let key = Tensor::from_floats(
        TensorData::new(key_data, [batch_size, seq_length, input_dim]),
        &device,
    );
    let value = Tensor::from_floats(
        TensorData::new(value_data, [batch_size, seq_length, input_dim]),
        &device,
    );

    println!("\nInput tensors shape:");
    println!("  - Query: {:?}", query.dims());
    println!("  - Key: {:?}", key.dims());
    println!("  - Value: {:?}", value.dims());

    // Forward pass through attention
    let (output, attention_weights) = attention.forward(query, key, value);

    println!("\nAttention output:");
    println!("  - Output shape: {:?}", output.dims());
    println!(
        "  - Attention weights shape: {:?}",
        attention_weights.dims()
    );

    // Analyze attention patterns
    let attention_stats = analyze_attention_patterns(&attention_weights);
    println!("  - Average attention entropy: {:.4}", attention_stats.0);
    println!("  - Max attention weight: {:.4}", attention_stats.1);
    println!("  - Min attention weight: {:.4}", attention_stats.2);

    println!("✅ Multi-head attention example completed!\n");
}

/// Demonstrates positional encoding
fn positional_encoding_example() {
    println!("📍 3. Positional Encoding");
    println!("-------------------------");

    let device = Device::<MyBackend>::default();

    // Create positional encoding configuration
    let config = PositionalEncodingConfig {
        input_dim: 512,
        max_seq_length: 1000,
        dropout_rate: 0.1,
    };

    println!("Creating positional encoding with:");
    println!("  - Input dimension: {}", config.input_dim);
    println!("  - Max sequence length: {}", config.max_seq_length);
    println!("  - Dropout rate: {}", config.dropout_rate);

    let pos_encoding = PositionalEncoding::new(config, device.clone());

    // Create sample sequence
    let batch_size = 3;
    let seq_length = 50;
    let input_dim = 512;

    let input_data: Vec<f32> = (0..batch_size * seq_length * input_dim)
        .map(|i| if i % 100 == 0 { 1.0 } else { 0.0 })
        .collect();

    let input = Tensor::from_floats(
        TensorData::new(input_data, [batch_size, seq_length, input_dim]),
        &device,
    );

    println!("\nInput shape: {:?}", input.dims());

    // Apply positional encoding
    let encoded_output = pos_encoding.forward(input.clone());
    println!("Encoded output shape: {:?}", encoded_output.dims());

    // Demonstrate position-dependent patterns
    let position_analysis = analyze_positional_patterns(&input, &encoded_output);
    println!("Positional encoding analysis:");
    println!("  - Average magnitude change: {:.6}", position_analysis.0);
    println!("  - Position variance: {:.6}", position_analysis.1);

    println!("✅ Positional encoding example completed!\n");
}

/// Demonstrates text classification with transformers
fn text_classification_example() {
    println!("📚 4. Text Classification with Transformers");
    println!("-------------------------------------------");

    let device = Device::<MyBackend>::default();

    // Create a simplified transformer for classification
    let config = TransformerEncoderConfig {
        input_dim: 128, // Smaller for this example
        hidden_dim: 512,
        num_heads: 4,
        num_layers: 3,
        dropout_rate: 0.1,
        max_seq_length: 100,
    };

    println!("Creating transformer for text classification:");
    println!("  - Vocabulary size: 1000 (simulated)");
    println!("  - Embedding dimension: {}", config.input_dim);
    println!("  - Number of classes: 3 (positive, negative, neutral)");

    let transformer = TransformerEncoder::new(config, device.clone());

    // Simulate text sequences (in practice, these would be tokenized text)
    let num_samples = 8;
    let seq_length = 20;
    let input_dim = 128;

    // Create synthetic "embedded" text data
    let text_data: Vec<f32> = (0..num_samples * seq_length * input_dim)
        .map(|i| {
            let sample_idx = i / (seq_length * input_dim);
            let pos_idx = (i / input_dim) % seq_length;
            // Simulate different text patterns for different classes
            match sample_idx % 3 {
                0 => ((i as f32).sin() * 0.5 + pos_idx as f32 * 0.01), // Positive class pattern
                1 => ((i as f32).cos() * 0.3 - pos_idx as f32 * 0.01), // Negative class pattern
                _ => ((i as f32 * 0.7).sin() * 0.2),                   // Neutral class pattern
            }
        })
        .collect();

    let text_input = Tensor::from_floats(
        TensorData::new(text_data, [num_samples, seq_length, input_dim]),
        &device,
    );

    println!("\nProcessing {} text samples...", num_samples);
    println!("Text input shape: {:?}", text_input.dims());

    // Forward pass through transformer
    let text_features = transformer.forward(text_input);
    println!("Text features shape: {:?}", text_features.dims());

    // Simulate classification by pooling and projecting
    let pooled_features = text_features.mean_dim(1); // Pool over sequence dimension
    println!("Pooled features shape: {:?}", pooled_features.dims());

    // In a real scenario, you'd add a classification head here
    println!("📊 Classification results (simulated):");
    for i in 0..num_samples {
        let class_prediction = i % 3;
        let confidence = 0.75 + (i as f32 * 0.05) % 0.2;
        let class_name = match class_prediction {
            0 => "Positive",
            1 => "Negative",
            _ => "Neutral",
        };
        println!(
            "  Sample {}: {} (confidence: {:.2})",
            i + 1,
            class_name,
            confidence
        );
    }

    println!("✅ Text classification example completed!\n");
}

/// Demonstrates sequence modeling capabilities
fn sequence_modeling_example() {
    println!("🔄 5. Sequence Modeling and Generation");
    println!("-------------------------------------");

    let device = Device::<MyBackend>::default();

    // Create transformer for sequence modeling
    let config = TransformerEncoderConfig {
        input_dim: 256,
        hidden_dim: 1024,
        num_heads: 8,
        num_layers: 4,
        dropout_rate: 0.1,
        max_seq_length: 200,
    };

    println!("Creating sequence model with transformer:");
    println!("  - Context window: {} tokens", config.max_seq_length);
    println!("  - Model dimension: {}", config.input_dim);

    let transformer = TransformerEncoder::new(config, device.clone());

    // Create a sequence pattern to model (e.g., sine wave with noise)
    let batch_size = 1;
    let seq_length = 50;
    let input_dim = 256;

    let sequence_data: Vec<f32> = (0..batch_size * seq_length * input_dim)
        .map(|i| {
            let time_step = ((i / input_dim) % seq_length) as f32;
            let feature_idx = i % input_dim;

            // Create a learnable pattern: sine wave + harmonics
            let base_freq = 0.1;
            let signal = (time_step * base_freq).sin()
                + 0.3 * (time_step * base_freq * 2.0).sin()
                + 0.1 * (time_step * base_freq * 3.0).sin();

            // Add feature-dependent variation
            signal * (1.0 + 0.1 * (feature_idx as f32 / input_dim as f32))
        })
        .collect();

    let sequence_input = Tensor::from_floats(
        TensorData::new(sequence_data, [batch_size, seq_length, input_dim]),
        &device,
    );

    println!("\nModeling sequence pattern:");
    println!("Input sequence shape: {:?}", sequence_input.dims());

    // Process sequence through transformer
    let sequence_output = transformer.forward(sequence_input.clone());
    println!("Output sequence shape: {:?}", sequence_output.dims());

    // Simulate next-token prediction
    println!("\n🔮 Sequence Prediction Simulation:");

    // Take the last few time steps for prediction
    let context_length = 10;
    let start_pos = seq_length - context_length;

    for step in 0..5 {
        let current_pos = start_pos + step;
        if current_pos < seq_length {
            // In a real model, you'd use the output to predict the next token
            let predicted_confidence = 0.8 + (step as f32 * 0.03) % 0.15;
            println!(
                "  Step {}: Predicting token at position {} (confidence: {:.3})",
                step + 1,
                current_pos + 1,
                predicted_confidence
            );
        }
    }

    // Analyze sequence patterns
    let pattern_analysis = analyze_sequence_patterns(&sequence_input, &sequence_output);
    println!("\n📈 Pattern Analysis:");
    println!("  - Input signal variance: {:.6}", pattern_analysis.0);
    println!("  - Output signal variance: {:.6}", pattern_analysis.1);
    println!(
        "  - Pattern preservation: {:.3}%",
        pattern_analysis.2 * 100.0
    );

    println!("✅ Sequence modeling example completed!\n");
}

/// Helper function to analyze attention patterns
fn analyze_attention_patterns(attention_weights: &Tensor<MyBackend, 4>) -> (f32, f32, f32) {
    // Convert to data and analyze
    let data = attention_weights
        .to_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .unwrap();

    let sum: f32 = data.iter().sum();
    let count = data.len() as f32;
    let mean = sum / count;

    let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));

    // Simple entropy approximation
    let entropy = data
        .iter()
        .map(|&x| if x > 0.0 { -x * x.ln() } else { 0.0 })
        .sum::<f32>()
        / count;

    (entropy, max_val, min_val)
}

/// Helper function to analyze positional encoding effects  
fn analyze_positional_patterns(
    input: &Tensor<MyBackend, 3>,
    output: &Tensor<MyBackend, 3>,
) -> (f32, f32) {
    let input_data = input.to_data().convert::<f32>().to_vec::<f32>().unwrap();
    let output_data = output.to_data().convert::<f32>().to_vec::<f32>().unwrap();

    // Calculate magnitude change
    let magnitude_change: f32 = input_data
        .iter()
        .zip(output_data.iter())
        .map(|(i, o)| (o - i).abs())
        .sum::<f32>()
        / input_data.len() as f32;

    // Calculate position variance (simplified)
    let position_variance: f32 = output_data
        .iter()
        .enumerate()
        .map(|(idx, &val)| {
            let pos_factor = (idx % 512) as f32 / 512.0;
            (val - pos_factor).powi(2)
        })
        .sum::<f32>()
        / output_data.len() as f32;

    (magnitude_change, position_variance)
}

/// Helper function to analyze sequence patterns
fn analyze_sequence_patterns(
    input: &Tensor<MyBackend, 3>,
    output: &Tensor<MyBackend, 3>,
) -> (f32, f32, f32) {
    let input_data = input.to_data().convert::<f32>().to_vec::<f32>().unwrap();
    let output_data = output.to_data().convert::<f32>().to_vec::<f32>().unwrap();

    // Calculate variances
    let input_mean = input_data.iter().sum::<f32>() / input_data.len() as f32;
    let input_variance = input_data
        .iter()
        .map(|x| (x - input_mean).powi(2))
        .sum::<f32>()
        / input_data.len() as f32;

    let output_mean = output_data.iter().sum::<f32>() / output_data.len() as f32;
    let output_variance = output_data
        .iter()
        .map(|x| (x - output_mean).powi(2))
        .sum::<f32>()
        / output_data.len() as f32;

    // Calculate pattern preservation (correlation-like measure)
    let covariance = input_data
        .iter()
        .zip(output_data.iter())
        .map(|(i, o)| (i - input_mean) * (o - output_mean))
        .sum::<f32>()
        / input_data.len() as f32;

    let correlation = covariance / (input_variance.sqrt() * output_variance.sqrt()).max(1e-8);

    (input_variance, output_variance, correlation.abs())
}
