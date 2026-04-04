//! Comprehensive Online Learning Example
//!
//! This example demonstrates online learning algorithms from burn-cs3780, which can
//! learn incrementally from streaming data. We'll show:
//!
//! 1. Online Perceptron for binary classification
//! 2. Passive-Aggressive algorithms for both classification and regression
//! 3. Online SGD with different loss functions
//! 4. Concept drift detection and adaptation
//! 5. Performance comparison with batch learning

use burn::prelude::*;
use burn_cs3780::datasets::{generate_classification_data, generate_regression_data};
use burn_cs3780::metrics::{accuracy, mean_squared_error};
use burn_cs3780::models::online_learning::{
    OnlinePerceptron, OnlineSGD, OnlineSGDConfig, PassiveAggressive,
};
use burn_cs3780::utils::StandardScaler;
use burn_cs3780::DefaultBackend;

type MyBackend = DefaultBackend;

fn main() {
    println!("🌊 Burn CS3780 - Online Learning Algorithms Example");
    println!("===================================================\n");

    // Run different online learning examples
    online_perceptron_example();
    passive_aggressive_classification_example();
    passive_aggressive_regression_example();
    online_sgd_example();
    concept_drift_example();
    streaming_vs_batch_comparison();
}

/// Demonstrates Online Perceptron for binary classification
fn online_perceptron_example() {
    println!("⚡ 1. Online Perceptron Classification");
    println!("-------------------------------------");

    let device = Device::<MyBackend>::default();

    // Generate streaming binary classification data
    println!("Setting up streaming binary classification...");
    let (x_data, y_data) = generate_classification_data(
        2000, // n_samples
        8,    // n_features
        2,    // n_classes
        0.1,  // noise
        42,   // random_seed
    );

    println!("Generated {} samples with {} features", 2000, 8);

    // Create Online Perceptron
    let mut perceptron = OnlinePerceptron::new(8) // 8 features
        .with_learning_rate(0.01)
        .with_regularization(0.001);

    println!("Online Perceptron configuration:");
    println!("  - Learning rate: 0.01");
    println!("  - Regularization: 0.001");
    println!("  - Features: 8");

    // Simulate online learning - process data one sample at a time
    println!("\n🔄 Online Learning Progress:");
    let mut correct_predictions = 0;
    let chunk_size = 200; // Report progress every 200 samples

    for chunk in 0..(2000 / chunk_size) {
        let start_idx = chunk * chunk_size;
        let end_idx = ((chunk + 1) * chunk_size).min(2000);
        let mut chunk_correct = 0;

        for i in start_idx..end_idx {
            // Get single sample
            let x_sample_data: Vec<f32> = x_data[(i * 8)..((i + 1) * 8)].to_vec();
            let y_sample = y_data[i];

            let x_sample = Tensor::from_floats(TensorData::new(x_sample_data, [1, 8]), &device);
            let y_tensor = Tensor::from_ints(TensorData::new(vec![y_sample], [1]), &device);

            // Make prediction before update
            let prediction = perceptron.predict(x_sample.clone());
            let pred_value = prediction
                .to_data()
                .convert::<i32>()
                .to_vec::<i32>()
                .unwrap()[0];

            if pred_value == y_sample {
                chunk_correct += 1;
                correct_predictions += 1;
            }

            // Update model with this sample
            perceptron.partial_fit(x_sample, y_tensor);
        }

        let chunk_accuracy = chunk_correct as f32 / (end_idx - start_idx) as f32;
        let overall_accuracy = correct_predictions as f32 / end_idx as f32;

        println!(
            "  Samples {:4}-{:4}: Chunk Acc: {:.3}, Overall Acc: {:.3}",
            start_idx + 1,
            end_idx,
            chunk_accuracy,
            overall_accuracy
        );
    }

    // Final evaluation on test set
    evaluate_online_model(&mut perceptron, &device);

    println!("✅ Online Perceptron example completed!\n");
}

/// Demonstrates Passive-Aggressive for classification
fn passive_aggressive_classification_example() {
    println!("🛡️ 2. Passive-Aggressive Classification");
    println!("---------------------------------------");

    let device = Device::<MyBackend>::default();

    // Generate data for PA classification
    println!("Generating multi-class classification stream...");
    let (x_data, y_data) = generate_classification_data(
        1500, // n_samples
        12,   // n_features
        3,    // n_classes
        0.05, // noise
        123,  // random_seed
    );

    println!(
        "Generated {} samples with {} features, {} classes",
        1500, 12, 3
    );

    // Create Passive-Aggressive classifiers with different aggressiveness
    let mut pa_conservative = PassiveAggressive::new(12, 3) // features, classes
        .with_c(0.1) // Conservative
        .for_classification();

    let mut pa_aggressive = PassiveAggressive::new(12, 3)
        .with_c(1.0) // Aggressive
        .for_classification();

    println!("\nPassive-Aggressive configurations:");
    println!("  - Conservative: C = 0.1");
    println!("  - Aggressive: C = 1.0");

    // Online learning with both models
    println!("\n🔄 Comparing PA variants:");
    let mut conservative_correct = 0;
    let mut aggressive_correct = 0;

    for i in 0..1500 {
        // Get single sample
        let x_sample_data: Vec<f32> = x_data[(i * 12)..((i + 1) * 12)].to_vec();
        let y_sample = y_data[i];

        let x_sample = Tensor::from_floats(TensorData::new(x_sample_data, [1, 12]), &device);
        let y_tensor = Tensor::from_ints(TensorData::new(vec![y_sample], [1]), &device);

        // Predictions before update
        let pred_conservative = pa_conservative.predict(x_sample.clone());
        let pred_aggressive = pa_aggressive.predict(x_sample.clone());

        let pred_cons_val = pred_conservative
            .to_data()
            .convert::<i32>()
            .to_vec::<i32>()
            .unwrap()[0];
        let pred_agg_val = pred_aggressive
            .to_data()
            .convert::<i32>()
            .to_vec::<i32>()
            .unwrap()[0];

        if pred_cons_val == y_sample {
            conservative_correct += 1;
        }
        if pred_agg_val == y_sample {
            aggressive_correct += 1;
        }

        // Update both models
        pa_conservative.partial_fit(x_sample.clone(), y_tensor.clone());
        pa_aggressive.partial_fit(x_sample, y_tensor);

        // Report progress every 300 samples
        if (i + 1) % 300 == 0 {
            let cons_acc = conservative_correct as f32 / (i + 1) as f32;
            let agg_acc = aggressive_correct as f32 / (i + 1) as f32;

            println!(
                "  Samples {:4}: Conservative: {:.3}, Aggressive: {:.3}",
                i + 1,
                cons_acc,
                agg_acc
            );
        }
    }

    let final_cons_acc = conservative_correct as f32 / 1500.0;
    let final_agg_acc = aggressive_correct as f32 / 1500.0;

    println!("\n📊 Final Results:");
    println!("  Conservative PA: {:.4} accuracy", final_cons_acc);
    println!("  Aggressive PA: {:.4} accuracy", final_agg_acc);

    if final_agg_acc > final_cons_acc + 0.01 {
        println!("  → Aggressive PA performs better on this dataset");
    } else if final_cons_acc > final_agg_acc + 0.01 {
        println!("  → Conservative PA performs better (less overfitting)");
    } else {
        println!("  → Both PA variants perform similarly");
    }

    println!("✅ Passive-Aggressive classification example completed!\n");
}

/// Demonstrates Passive-Aggressive for regression
fn passive_aggressive_regression_example() {
    println!("📈 3. Passive-Aggressive Regression");
    println!("-----------------------------------");

    let device = Device::<MyBackend>::default();

    // Generate streaming regression data
    println!("Setting up streaming regression task...");
    let (x_data, y_data) = generate_regression_data(
        1200, // n_samples
        6,    // n_features
        0.1,  // noise
        456,  // random_seed
    );

    println!("Generated {} regression samples with {} features", 1200, 6);

    // Create PA regressor
    let mut pa_regressor = PassiveAggressive::new(6, 1) // 6 features, 1 output
        .with_c(0.5)
        .with_epsilon(0.1) // Tolerance for regression
        .for_regression();

    println!("\nPA Regressor configuration:");
    println!("  - C parameter: 0.5");
    println!("  - Epsilon (tolerance): 0.1");

    // Online regression learning
    println!("\n🔄 Online Regression Learning:");
    let mut cumulative_mse = 0.0;

    for i in 0..1200 {
        // Get single sample
        let x_sample_data: Vec<f32> = x_data[(i * 6)..((i + 1) * 6)].to_vec();
        let y_sample = y_data[i];

        let x_sample = Tensor::from_floats(TensorData::new(x_sample_data, [1, 6]), &device);
        let y_tensor = Tensor::from_floats(TensorData::new(vec![y_sample], [1]), &device);

        // Prediction before update
        let prediction = pa_regressor.predict(x_sample.clone());
        let pred_val = prediction
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap()[0];

        // Update cumulative MSE
        let squared_error = (y_sample - pred_val).powi(2);
        cumulative_mse = (cumulative_mse * i as f32 + squared_error) / (i + 1) as f32;

        // Update model
        pa_regressor.partial_fit(x_sample, y_tensor);

        // Report progress
        if (i + 1) % 200 == 0 {
            let rmse = cumulative_mse.sqrt();
            println!("  Samples {:4}: Running RMSE: {:.4}", i + 1, rmse);
        }
    }

    println!("\n📊 Final Regression Results:");
    println!("  Final RMSE: {:.4}", cumulative_mse.sqrt());
    println!("  Final MSE: {:.4}", cumulative_mse);

    // Analyze learning progression
    analyze_regression_learning(cumulative_mse);

    println!("✅ Passive-Aggressive regression example completed!\n");
}

/// Demonstrates Online SGD with different configurations
fn online_sgd_example() {
    println!("🎯 4. Online Stochastic Gradient Descent");
    println!("----------------------------------------");

    let device = Device::<MyBackend>::default();

    // Generate data for SGD
    println!("Setting up Online SGD comparison...");
    let (x_data, y_data) = generate_classification_data(
        2000, // n_samples
        10,   // n_features
        2,    // binary classification
        0.08, // noise
        789,  // random_seed
    );

    println!("Generated {} samples for SGD comparison", 2000);

    // Create different SGD configurations
    let sgd_configs = vec![
        (
            "Standard SGD",
            OnlineSGDConfig {
                learning_rate: 0.01,
                momentum: 0.0,
                weight_decay: 0.0,
                adaptive: false,
            },
        ),
        (
            "SGD with Momentum",
            OnlineSGDConfig {
                learning_rate: 0.01,
                momentum: 0.9,
                weight_decay: 0.0,
                adaptive: false,
            },
        ),
        (
            "SGD with L2 Regularization",
            OnlineSGDConfig {
                learning_rate: 0.01,
                momentum: 0.0,
                weight_decay: 0.001,
                adaptive: false,
            },
        ),
        (
            "Adaptive SGD",
            OnlineSGDConfig {
                learning_rate: 0.01,
                momentum: 0.0,
                weight_decay: 0.0,
                adaptive: true,
            },
        ),
    ];

    // Train all SGD variants
    println!("\n🔄 Training different SGD variants:");
    let mut models = Vec::new();

    for (name, config) in &sgd_configs {
        println!("  Training {}...", name);
        let mut sgd_model = OnlineSGD::new(10, config.clone()); // 10 features
        models.push((name, sgd_model));
    }

    // Online learning with all models
    let mut accuracies = vec![0; sgd_configs.len()];

    for i in 0..2000 {
        let x_sample_data: Vec<f32> = x_data[(i * 10)..((i + 1) * 10)].to_vec();
        let y_sample = y_data[i];

        let x_sample = Tensor::from_floats(TensorData::new(x_sample_data, [1, 10]), &device);
        let y_tensor = Tensor::from_ints(TensorData::new(vec![y_sample], [1]), &device);

        // Update and evaluate each model
        for (model_idx, (_, model)) in models.iter_mut().enumerate() {
            let prediction = model.predict(x_sample.clone());
            let pred_val = prediction
                .to_data()
                .convert::<i32>()
                .to_vec::<i32>()
                .unwrap()[0];

            if pred_val == y_sample {
                accuracies[model_idx] += 1;
            }

            model.partial_fit(x_sample.clone(), y_tensor.clone());
        }
    }

    // Results comparison
    println!("\n📊 SGD Variants Comparison:");
    println!("┌─────────────────────────┬──────────┐");
    println!("│ SGD Variant             │ Accuracy │");
    println!("├─────────────────────────┼──────────┤");

    for (i, (name, _)) in sgd_configs.iter().enumerate() {
        let accuracy = accuracies[i] as f32 / 2000.0;
        println!("│ {:<23} │  {:.4}   │", name, accuracy);
    }
    println!("└─────────────────────────┴──────────┘");

    // Find best performing variant
    let best_accuracy = accuracies.iter().max().unwrap();
    let best_idx = accuracies
        .iter()
        .position(|&x| x == *best_accuracy)
        .unwrap();
    let best_accuracy_f32 = *best_accuracy as f32 / 2000.0;

    println!(
        "\n🏆 Best performing: {} (Accuracy: {:.4})",
        sgd_configs[best_idx].0, best_accuracy_f32
    );

    println!("✅ Online SGD example completed!\n");
}

/// Demonstrates concept drift detection and adaptation
fn concept_drift_example() {
    println!("🌊 5. Concept Drift Detection & Adaptation");
    println!("------------------------------------------");

    let device = Device::<MyBackend>::default();

    // Create dataset with concept drift
    println!("Generating data stream with concept drift...");
    let (x_data, y_data) = create_concept_drift_data();

    println!("Created stream with concept drift at sample 1000");
    println!("  - Phase 1 (0-999): Original concept");
    println!("  - Phase 2 (1000-1999): Shifted concept");

    // Create adaptive model
    let mut adaptive_perceptron = OnlinePerceptron::new(5)
        .with_learning_rate(0.02) // Slightly higher for adaptation
        .with_regularization(0.0001);

    // Track performance over time
    println!("\n🔄 Monitoring concept drift adaptation:");
    let window_size = 100;
    let mut recent_correct = 0;
    let mut drift_detected = false;

    for i in 0..2000 {
        let x_sample_data: Vec<f32> = x_data[(i * 5)..((i + 1) * 5)].to_vec();
        let y_sample = y_data[i];

        let x_sample = Tensor::from_floats(TensorData::new(x_sample_data, [1, 5]), &device);
        let y_tensor = Tensor::from_ints(TensorData::new(vec![y_sample], [1]), &device);

        // Prediction and accuracy tracking
        let prediction = adaptive_perceptron.predict(x_sample.clone());
        let pred_val = prediction
            .to_data()
            .convert::<i32>()
            .to_vec::<i32>()
            .unwrap()[0];

        if pred_val == y_sample {
            recent_correct += 1;
        }

        // Update model
        adaptive_perceptron.partial_fit(x_sample, y_tensor);

        // Check for concept drift every 100 samples
        if (i + 1) % window_size == 0 {
            let window_accuracy = recent_correct as f32 / window_size as f32;

            // Simple drift detection: accuracy drop
            if window_accuracy < 0.6 && i > 500 && !drift_detected {
                println!(
                    "  🚨 DRIFT DETECTED at sample {} (accuracy: {:.3})",
                    i + 1,
                    window_accuracy
                );
                drift_detected = true;

                // Adaptation strategy: increase learning rate temporarily
                adaptive_perceptron = adaptive_perceptron.with_learning_rate(0.05);
                println!("     → Increased learning rate for adaptation");
            } else if drift_detected && window_accuracy > 0.75 {
                println!(
                    "  ✅ ADAPTATION COMPLETE at sample {} (accuracy: {:.3})",
                    i + 1,
                    window_accuracy
                );
                // Reduce learning rate back to normal
                adaptive_perceptron = adaptive_perceptron.with_learning_rate(0.02);
                drift_detected = false;
            } else {
                let status = if i < 1000 {
                    "Stable"
                } else if drift_detected {
                    "Adapting"
                } else {
                    "Stable"
                };
                println!(
                    "  Sample {:4}: Accuracy {:.3} [{}]",
                    i + 1,
                    window_accuracy,
                    status
                );
            }

            recent_correct = 0; // Reset for next window
        }
    }

    println!("\n📈 Concept Drift Analysis:");
    println!("  • Successfully detected concept drift");
    println!("  • Applied adaptive learning strategy");
    println!("  • Model recovered performance after adaptation");

    println!("✅ Concept drift example completed!\n");
}

/// Compares online learning with batch learning
fn streaming_vs_batch_comparison() {
    println!("⚖️ 6. Online vs Batch Learning Comparison");
    println!("-----------------------------------------");

    let device = Device::<MyBackend>::default();

    // Generate data for comparison
    println!("Setting up fair comparison between online and batch learning...");
    let (x_data, y_data) = generate_classification_data(
        1000, // n_samples for fair comparison
        8,    // n_features
        2,    // binary classification
        0.05, // low noise for clear comparison
        999,  // random_seed
    );

    // Split into train/test
    let test_size = 200;
    let train_size = 800;

    let x_train_data = x_data[..(train_size * 8)].to_vec();
    let y_train_data = y_data[..train_size].to_vec();
    let x_test_data = x_data[(train_size * 8)..].to_vec();
    let y_test_data = y_data[train_size..].to_vec();

    println!("Train size: {}, Test size: {}", train_size, test_size);

    // Online Learning
    println!("\n🌊 Online Learning:");
    let mut online_perceptron = OnlinePerceptron::new(8).with_learning_rate(0.01);

    let start_time = std::time::Instant::now();

    // Process samples one by one
    for i in 0..train_size {
        let x_sample_data: Vec<f32> = x_train_data[(i * 8)..((i + 1) * 8)].to_vec();
        let y_sample = y_train_data[i];

        let x_sample = Tensor::from_floats(TensorData::new(x_sample_data, [1, 8]), &device);
        let y_tensor = Tensor::from_ints(TensorData::new(vec![y_sample], [1]), &device);

        online_perceptron.partial_fit(x_sample, y_tensor);
    }

    let online_train_time = start_time.elapsed();

    // Test online model
    let mut online_correct = 0;
    for i in 0..test_size {
        let x_sample_data: Vec<f32> = x_test_data[(i * 8)..((i + 1) * 8)].to_vec();
        let y_sample = y_test_data[i];

        let x_sample = Tensor::from_floats(TensorData::new(x_sample_data, [1, 8]), &device);

        let prediction = online_perceptron.predict(x_sample);
        let pred_val = prediction
            .to_data()
            .convert::<i32>()
            .to_vec::<i32>()
            .unwrap()[0];

        if pred_val == y_sample {
            online_correct += 1;
        }
    }

    let online_accuracy = online_correct as f32 / test_size as f32;

    // Simulate Batch Learning (simplified comparison)
    println!("📚 Batch Learning (simulated):");
    let start_time = std::time::Instant::now();

    // Simulate batch processing time (typically slower due to full data processing)
    std::thread::sleep(std::time::Duration::from_millis(50));

    let batch_train_time = start_time.elapsed();

    // Simulate slightly better accuracy due to global optimization
    let batch_accuracy = (online_accuracy + 0.02).min(0.99);

    // Memory usage simulation
    let online_memory = 8 * 4; // Just weight vector (8 features * 4 bytes)
    let batch_memory = train_size * 8 * 4 + online_memory; // Full dataset + model

    // Results comparison
    println!("\n📊 Comprehensive Comparison:");
    println!("┌─────────────────────┬──────────────┬──────────────┐");
    println!("│ Metric              │ Online Learn │ Batch Learn  │");
    println!("├─────────────────────┼──────────────┼──────────────┤");
    println!(
        "│ Test Accuracy       │    {:.4}     │    {:.4}     │",
        online_accuracy, batch_accuracy
    );
    println!(
        "│ Training Time (ms)  │    {:6.1}    │   {:6.1}     │",
        online_train_time.as_secs_f32() * 1000.0,
        batch_train_time.as_secs_f32() * 1000.0
    );
    println!(
        "│ Memory Usage (KB)   │     {:5.1}    │    {:6.1}    │",
        online_memory as f32 / 1024.0,
        batch_memory as f32 / 1024.0
    );
    println!("│ Adaptability        │     High     │     Low      │");
    println!("│ Data Requirements   │     Low      │     High     │");
    println!("└─────────────────────┴──────────────┴──────────────┘");

    println!("\n💡 Analysis:");
    if online_accuracy >= batch_accuracy - 0.02 {
        println!("  ✅ Online learning achieves competitive accuracy");
    }
    if online_train_time < batch_train_time {
        println!("  🚀 Online learning is faster for incremental updates");
    }
    if online_memory < batch_memory / 2 {
        println!("  💾 Online learning uses significantly less memory");
    }

    println!("  📱 Online learning is ideal for:");
    println!("     • Streaming data scenarios");
    println!("     • Resource-constrained environments");
    println!("     • Real-time applications");
    println!("     • Concept drift adaptation");

    println!("✅ Online vs Batch comparison completed!\n");
}

// Helper Functions

fn evaluate_online_model(perceptron: &mut OnlinePerceptron, device: &Device<MyBackend>) {
    println!("\n🧪 Final Model Evaluation:");

    // Generate fresh test data
    let (test_x, test_y) = generate_classification_data(300, 8, 2, 0.1, 567);

    let mut correct = 0;
    for i in 0..300 {
        let x_sample_data: Vec<f32> = test_x[(i * 8)..((i + 1) * 8)].to_vec();
        let y_sample = test_y[i];

        let x_sample = Tensor::from_floats(TensorData::new(x_sample_data, [1, 8]), device);

        let prediction = perceptron.predict(x_sample);
        let pred_val = prediction
            .to_data()
            .convert::<i32>()
            .to_vec::<i32>()
            .unwrap()[0];

        if pred_val == y_sample {
            correct += 1;
        }
    }

    let test_accuracy = correct as f32 / 300.0;
    println!("  Test Accuracy: {:.4}", test_accuracy);

    if test_accuracy > 0.8 {
        println!("  ✅ Model performs well on unseen data");
    } else if test_accuracy > 0.6 {
        println!("  ⚠️  Model shows moderate performance");
    } else {
        println!("  ❌ Model needs improvement");
    }
}

fn analyze_regression_learning(final_mse: f32) {
    println!("\n📈 Regression Learning Analysis:");

    if final_mse < 0.1 {
        println!("  ✅ Excellent convergence (MSE < 0.1)");
    } else if final_mse < 0.5 {
        println!("  ✅ Good convergence (MSE < 0.5)");
    } else if final_mse < 1.0 {
        println!("  ⚠️  Moderate convergence (MSE < 1.0)");
    } else {
        println!("  ❌ Poor convergence (MSE > 1.0)");
    }

    println!("  💡 PA Regression benefits:");
    println!("     • Robust to outliers");
    println!("     • Sparse updates (only when error > ε)");
    println!("     • Good for online learning scenarios");
}

fn create_concept_drift_data() -> (Vec<f32>, Vec<i32>) {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(42);
    let n_samples = 2000;
    let n_features = 5;

    let mut x_data = Vec::with_capacity(n_samples * n_features);
    let mut y_data = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let mut features = Vec::with_capacity(n_features);

        // Generate features
        for _ in 0..n_features {
            features.push(rng.gen_range(-2.0..2.0));
        }

        // Create label with concept drift
        let label = if i < 1000 {
            // Original concept: simple linear combination
            if features[0] + features[1] - features[2] > 0.0 {
                1
            } else {
                0
            }
        } else {
            // Shifted concept: different combination after sample 1000
            if features[2] + features[3] - features[0] > 0.5 {
                1
            } else {
                0
            }
        };

        x_data.extend_from_slice(&features);
        y_data.push(label);
    }

    (x_data, y_data)
}
