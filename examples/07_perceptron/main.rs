/*!
# Project 2: The Perceptron Algorithm

This example demonstrates the classic Perceptron algorithm for binary classification,
adapted from CS3780 Python notebook. It showcases:

- Perceptron weight update rule
- Training loop with convergence checking
- Linear classification with decision boundaries
- Applications to both synthetic 2D data and real-world problems

## Educational Objectives

This implementation teaches:
- The fundamental perceptron algorithm and its geometric interpretation
- Linear separability and convergence guarantees
- Binary classification with decision hyperplanes
- Practical limitations of linear classifiers

## CS3780 Connection

This adapts Python Homework 2 from CS3780, replacing Python/NumPy operations
with Rust/Burn tensor operations while maintaining the same algorithmic approach.
Core functions implemented:
- `perceptron_update()` → Weight update step
- `perceptron_train()` → Full training algorithm
- `classify_linear()` → Prediction with learned weights

## Usage

```bash
cargo run --example 07_perceptron
```

The example will:
1. Demonstrate perceptron weight updates
2. Train on linearly separable 2D data
3. Apply to digit classification (binary: 0 vs 1)
4. Show decision boundary visualization (conceptual)
*/

use burn::tensor::{backend::Backend, Device, Distribution, Shape, Tensor};
use burn_cs3780::DefaultBackend;
use rand::seq::SliceRandom;
use std::time::Instant;

type MyBackend = DefaultBackend;

/// Single perceptron weight update step
/// If prediction is incorrect: w_new = w + y * x, b_new = b + y
fn perceptron_update<B: Backend<FloatElem = f32>>(
    x: Tensor<B, 1>, // Input vector [d]
    y: f32,          // True label (-1 or +1)
    w: Tensor<B, 1>, // Current weights [d]
    b: f32,          // Current bias
) -> (Tensor<B, 1>, f32) {
    // Updated weights and bias
    assert!(y == 1.0 || y == -1.0, "Label must be +1 or -1");

    // Compute prediction: sign(w^T x + b)
    let prediction = w.clone().mul(x.clone()).sum().into_scalar() + b;
    let pred_sign = if prediction > 0.0 { 1.0 } else { -1.0 };

    // Update only if prediction is incorrect
    if pred_sign != y {
        let w_new = w.add(x.mul_scalar(y));
        let b_new = b + y;
        (w_new, b_new)
    } else {
        (w, b)
    }
}

/// Full perceptron training algorithm
fn perceptron_train<B: Backend<FloatElem = f32>>(
    xs: Tensor<B, 2>, // Training data [n, d]
    ys: Vec<f32>,     // Training labels [n]
    max_iter: usize,  // Maximum iterations
    device: &Device<B>,
) -> (Tensor<B, 1>, f32) {
    // Learned weights and bias
    let [n, d] = xs.dims();
    assert_eq!(ys.len(), n, "Number of samples and labels must match");

    // Verify all labels are +1 or -1
    for &y in &ys {
        assert!(y == 1.0 || y == -1.0, "All labels must be +1 or -1");
    }

    // Initialize weights to zero and bias to zero
    let mut w = Tensor::zeros(Shape::new([d]), device);
    let mut b = 0.0;

    println!("Training perceptron:");
    println!("  Training samples: {}", n);
    println!("  Features: {}", d);
    println!("  Max iterations: {}", max_iter);

    let mut converged = false;

    for iteration in 0..max_iter {
        let mut errors = 0;

        // Create random permutation for this iteration
        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);

        // Process samples in random order
        for &i in &indices {
            let x_i = xs.clone().slice([i..i + 1]).squeeze::<1>();
            let y_i = ys[i];

            // Check if current prediction is correct
            let prediction = w.clone().mul(x_i.clone()).sum().into_scalar() + b;
            let pred_sign = if prediction > 0.0 { 1.0 } else { -1.0 };

            if pred_sign != y_i {
                errors += 1;
                // Update weights
                let (w_new, b_new) = perceptron_update(x_i, y_i, w, b);
                w = w_new;
                b = b_new;
            }
        }

        if errors == 0 {
            converged = true;
            println!("  Converged after {} iterations!", iteration + 1);
            break;
        } else if iteration % 10 == 9 {
            println!("  Iteration {}: {} errors", iteration + 1, errors);
        }
    }

    if !converged {
        println!(
            "  Did not converge after {} iterations. Final errors: {}",
            max_iter, "check final pass"
        );
    }

    (w, b)
}

/// Linear classification function
fn classify_linear<B: Backend<FloatElem = f32>>(
    xs: Tensor<B, 2>, // Test data [n, d]
    w: Tensor<B, 1>,  // Learned weights [d]
    b: f32,           // Learned bias
) -> Vec<f32> {
    // Predictions [n]
    let [n, _d] = xs.dims();
    let mut predictions = Vec::with_capacity(n);

    for i in 0..n {
        let x_i = xs.clone().slice([i..i + 1]).squeeze::<1>();
        let score = w.clone().mul(x_i).sum().into_scalar() + b;
        let prediction = if score > 0.0 { 1.0 } else { -1.0 };
        predictions.push(prediction);
    }

    predictions
}

/// Generate linearly separable 2D data
fn generate_linearly_separable_data(
    n: usize,
    device: &Device<MyBackend>,
) -> (Tensor<MyBackend, 2>, Vec<f32>) {
    println!("\nGenerating linearly separable 2D data:");
    println!("  Samples: {}", n);

    // Generate random points in [-5, 5] x [-5, 5]
    let xs = Tensor::random(Shape::new([n, 2]), Distribution::Uniform(-5.0, 5.0), device);

    // Define a random separating hyperplane
    let w_true = Tensor::random(Shape::new([2]), Distribution::Normal(0.0, 1.0), device);
    let b_true = rand::random::<f32>() * 2.0 - 1.0; // Random bias in [-1, 1]

    // Assign labels based on which side of hyperplane points lie
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let x_i = xs.clone().slice([i..i + 1]).squeeze::<1>();
        let score = w_true.clone().mul(x_i).sum().into_scalar() + b_true;
        let label = if score > 0.0 { 1.0 } else { -1.0 };
        ys.push(label);
    }

    // Count positive and negative examples
    let pos_count = ys.iter().filter(|&&y| y == 1.0).count();
    let neg_count = ys.len() - pos_count;
    println!("  Positive examples: {}", pos_count);
    println!("  Negative examples: {}", neg_count);

    (xs, ys)
}

/// Generate binary digit data (0 vs 1 classification)
fn generate_binary_digit_data(
    n: usize,
    device: &Device<MyBackend>,
) -> (Tensor<MyBackend, 2>, Vec<f32>) {
    println!("\nGenerating synthetic binary digit data (0 vs 1):");
    println!("  Samples: {}", n);

    let n_features = 64; // 8x8 pixel images

    // Generate digit '0' patterns
    let n_zeros = n / 2;
    let mut zero_patterns = Vec::new();
    for _ in 0..n_zeros {
        // Create a ring pattern for '0'
        let mut pattern = vec![0.0; n_features];
        for i in 0..8 {
            for j in 0..8 {
                let idx = i * 8 + j;
                // Create ring: edge pixels are bright, center is dark
                if i == 0 || i == 7 || j == 0 || j == 7 {
                    pattern[idx] = 0.8 + rand::random::<f32>() * 0.2; // 0.8-1.0
                } else if i == 1 || i == 6 || j == 1 || j == 6 {
                    pattern[idx] = rand::random::<f32>() * 0.3; // 0.0-0.3
                } else {
                    pattern[idx] = rand::random::<f32>() * 0.2; // 0.0-0.2
                }
            }
        }
        let pattern_tensor: Tensor<MyBackend, 1> = Tensor::from_floats(pattern.as_slice(), device);
        zero_patterns.push(pattern_tensor.unsqueeze::<2>());
    }

    // Generate digit '1' patterns
    let n_ones = n - n_zeros;
    let mut one_patterns = Vec::new();
    for _ in 0..n_ones {
        // Create a vertical line pattern for '1'
        let mut pattern = vec![0.0; n_features];
        for i in 0..8 {
            for j in 0..8 {
                let idx = i * 8 + j;
                // Create vertical line in middle columns
                if j >= 3 && j <= 4 {
                    pattern[idx] = 0.7 + rand::random::<f32>() * 0.3; // 0.7-1.0
                } else {
                    pattern[idx] = rand::random::<f32>() * 0.3; // 0.0-0.3
                }
            }
        }
        let pattern_tensor: Tensor<MyBackend, 1> = Tensor::from_floats(pattern.as_slice(), device);
        one_patterns.push(pattern_tensor.unsqueeze::<2>());
    }

    // Combine data
    let mut all_patterns = zero_patterns;
    all_patterns.extend(one_patterns);

    let xs = Tensor::cat(all_patterns, 0);

    // Create labels: -1 for '0', +1 for '1'
    let mut ys = vec![-1.0; n_zeros];
    ys.extend(vec![1.0; n_ones]);

    // Shuffle the data
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = rand::thread_rng();
    indices.shuffle(&mut rng);

    let shuffled_patterns: Vec<Tensor<MyBackend, 2>> = indices
        .iter()
        .map(|&i| xs.clone().slice([i..i + 1]))
        .collect();
    let shuffled_xs = Tensor::cat(shuffled_patterns, 0);

    let shuffled_ys: Vec<f32> = indices.iter().map(|&i| ys[i]).collect();

    println!("  Digit '0' samples: {} (label: -1)", n_zeros);
    println!("  Digit '1' samples: {} (label: +1)", n_ones);
    println!("  Feature dimension: {}", n_features);

    (shuffled_xs, shuffled_ys)
}

/// Compute classification accuracy
fn compute_accuracy(y_true: &[f32], y_pred: &[f32]) -> f32 {
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|&(a, b)| a == b)
        .count();
    correct as f32 / y_true.len() as f32
}

/// Demonstrate perceptron weight update
fn demonstrate_weight_update(device: &Device<MyBackend>) {
    println!("=== Perceptron Weight Update Demonstration ===");

    // Create a simple 2D example
    let x: Tensor<MyBackend, 1> = Tensor::from_floats([1.0, 2.0], device);
    let y = 1.0;
    let w: Tensor<MyBackend, 1> = Tensor::from_floats([0.5, -0.3], device);
    let b = 0.1;

    println!("Input vector x: [{:.2}, {:.2}]", 1.0, 2.0);
    println!("True label y: {:.0}", y);
    println!("Current weights w: [{:.2}, {:.2}]", 0.5, -0.3);
    println!("Current bias b: {:.2}", b);

    // Show prediction
    let score: f32 = w.clone().mul(x.clone()).sum().into_scalar() + b;
    let prediction = if score > 0.0 { 1.0 } else { -1.0 };

    println!("Score (w^T x + b): {:.3}", score);
    println!("Prediction: {:.0}", prediction);

    // Apply update
    let (w_new, b_new) = perceptron_update(x, y, w, b);
    let w_vals: Vec<f32> = (0..2)
        .map(|i| w_new.clone().slice([i..i + 1]).into_scalar())
        .collect();

    if prediction != y {
        println!("❌ Incorrect prediction! Applying update...");
        println!("New weights w: [{:.2}, {:.2}]", w_vals[0], w_vals[1]);
        println!("New bias b: {:.2}", b_new);
    } else {
        println!("✅ Correct prediction! No update needed.");
    }
}

/// Test perceptron on linearly separable data
fn test_linearly_separable_data(device: &Device<MyBackend>) {
    println!("\n=== Linearly Separable Data Test ===");

    let (xs, ys) = generate_linearly_separable_data(100, device);

    let start_time = Instant::now();
    let (w, b) = perceptron_train(xs.clone(), ys.clone(), 100, device);
    let elapsed = start_time.elapsed();

    // Test on training data (should be 100% accurate for linearly separable data)
    let predictions = classify_linear(xs, w, b);
    let accuracy = compute_accuracy(&ys, &predictions);

    println!("\nResults:");
    println!("  Training accuracy: {:.1}%", accuracy * 100.0);
    println!("  Training time: {:.4}s", elapsed.as_secs_f64());

    if accuracy == 1.0 {
        println!("✅ Perfect classification! Data was linearly separable.");
    } else {
        println!("⚠️  Imperfect classification. Data might not be linearly separable.");
    }
}

/// Test perceptron on binary digit classification
fn test_binary_digit_classification(device: &Device<MyBackend>) {
    println!("\n=== Binary Digit Classification (0 vs 1) ===");

    let total_samples = 200;
    let (xs, ys) = generate_binary_digit_data(total_samples, device);

    // Split into train/test
    let train_size = (0.8 * total_samples as f32) as usize;
    let test_size = total_samples - train_size;

    // Training data
    let train_patterns: Vec<Tensor<MyBackend, 2>> = (0..train_size)
        .map(|i| xs.clone().slice([i..i + 1]))
        .collect();
    let train_xs = Tensor::cat(train_patterns, 0);
    let train_ys = ys[0..train_size].to_vec();

    // Test data
    let test_patterns: Vec<Tensor<MyBackend, 2>> = (train_size..total_samples)
        .map(|i| xs.clone().slice([i..i + 1]))
        .collect();
    let test_xs = Tensor::cat(test_patterns, 0);
    let test_ys = ys[train_size..].to_vec();

    println!("Training set: {} samples", train_size);
    println!("Test set: {} samples", test_size);

    // Train perceptron
    let start_time = Instant::now();
    let (w, b) = perceptron_train(train_xs.clone(), train_ys.clone(), 50, device);
    let elapsed = start_time.elapsed();

    // Evaluate on training and test data
    let train_predictions = classify_linear(train_xs, w.clone(), b);
    let test_predictions = classify_linear(test_xs, w, b);

    let train_accuracy = compute_accuracy(&train_ys, &train_predictions);
    let test_accuracy = compute_accuracy(&test_ys, &test_predictions);

    println!("\nResults:");
    println!("  Training accuracy: {:.1}%", train_accuracy * 100.0);
    println!("  Test accuracy: {:.1}%", test_accuracy * 100.0);
    println!("  Training time: {:.4}s", elapsed.as_secs_f64());

    // Analyze results
    if train_accuracy > 0.9 && test_accuracy > 0.8 {
        println!("✅ Good performance! Perceptron successfully learned to distinguish digits.");
    } else if train_accuracy > test_accuracy + 0.1 {
        println!("⚠️  Possible overfitting. Consider simpler features or more data.");
    } else {
        println!("ℹ️  Note: Real digit data is often not linearly separable.");
    }
}

/// Educational insights about the perceptron algorithm
fn print_educational_insights() {
    println!("\n=== Educational Insights ===");
    println!("The Perceptron Algorithm:");
    println!("• Linear binary classifier using decision hyperplane");
    println!("• Update rule: w_new = w + y*x, b_new = b + y (when wrong)");
    println!("• Guaranteed to converge for linearly separable data");
    println!("• Number of mistakes is bounded by (R²/γ²) where:");
    println!("  - R is maximum distance from origin");
    println!("  - γ is margin (minimum distance to hyperplane)");

    println!("\nKey Properties:");
    println!("• Only updates weights on classification errors");
    println!("• Decision boundary: w^T x + b = 0");
    println!("• Cannot solve non-linearly separable problems (XOR)");
    println!("• Forms basis for neural networks and SVMs");

    println!("\nLimitations:");
    println!("• No convergence guarantee for non-separable data");
    println!("• Sensitive to feature scaling");
    println!("• No probabilistic outputs");
    println!("• Cannot handle multi-class problems directly");
}

/// Main demonstration function
fn main() {
    println!("=== CS3780 Project 2: The Perceptron Algorithm ===\n");

    // Initialize device
    let device = Default::default();

    // Run demonstrations
    demonstrate_weight_update(&device);
    test_linearly_separable_data(&device);
    test_binary_digit_classification(&device);
    print_educational_insights();

    println!("\n=== CS3780 Educational Connection ===");
    println!("This example adapts Python Homework 2 from CS3780:");
    println!("• Demonstrates perceptron weight update mechanism");
    println!("• Shows convergence on linearly separable data");
    println!("• Applies to binary classification problems");
    println!("• Illustrates geometric interpretation of linear classifiers");
    println!("• Rust/Burn implementation maintains same algorithmic principles");

    println!("\nFor comparison with original Python notebook:");
    println!("• perceptronUpdate() → perceptron_update()");
    println!("• perceptron() → perceptron_train()");
    println!("• classifyLinear() → classify_linear()");
    println!("• Interactive visualization replaced with numerical results");
    println!("• Synthetic digit data simulates real MNIST digit classification");
}
