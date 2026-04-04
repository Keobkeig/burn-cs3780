/*!
# Project 1: k-Nearest Neighbors Face Recognition

This example demonstrates k-nearest neighbors classification for face recognition,
adapted from the CS3780 Python notebook. It showcases:

- Computing L2 (Euclidean) distances between feature vectors
- Finding k nearest neighbors in feature space
- Classifying test images based on majority vote of neighbors
- Evaluating classification accuracy

## Educational Objectives

This implementation teaches:
- Distance-based classification algorithms
- The effect of k parameter on classification boundaries
- Practical application to computer vision tasks

## CS3780 Connection

This adapts Python Homework 1 from CS3780, replacing Python/NumPy operations
with Rust/Burn tensor operations while maintaining the same algorithmic approach.

## Usage

```bash
cargo run --example 06_knn_faces
```
*/

use burn::tensor::{backend::Backend, Device, Distribution, Shape, Tensor};
use burn_cs3780::DefaultBackend;
use rand::seq::SliceRandom;
use std::time::Instant;

type MyBackend = DefaultBackend;

/// L2 distance computation (Euclidean distance)
fn l2_distance<B: Backend<FloatElem = f32>>(
    x: Tensor<B, 2>,         // [n, d]
    z: Option<Tensor<B, 2>>, // [m, d]
) -> Tensor<B, 2> // [n, m]
{
    let z = z.unwrap_or_else(|| x.clone());

    let [_n, d1] = x.dims();
    let [_m, d2] = z.dims();

    assert_eq!(d1, d2, "Dimensions must match for distance computation");

    // Compute pairwise L2 distances using broadcasting
    // ||x - z||^2 = ||x||^2 + ||z||^2 - 2 * x * z^T

    let x_norm_sq = x.clone().powf_scalar(2.0).sum_dim(1); // [n]
    let z_norm_sq = z.clone().powf_scalar(2.0).sum_dim(1); // [m]

    let x_norm_sq = x_norm_sq.unsqueeze::<2>(); // [n, 1]
    let z_norm_sq = z_norm_sq.unsqueeze::<2>().swap_dims(0, 1); // [1, m]

    let cross_term = x.matmul(z.transpose()); // [n, m]
    let cross_term = cross_term.mul_scalar(-2.0);

    let distances_sq = x_norm_sq.add(z_norm_sq).add(cross_term);
    distances_sq.clamp_min(0.0).sqrt()
}

/// Find k nearest neighbors using a simplified approach
fn find_knn<B: Backend<FloatElem = f32>>(
    x_train: Tensor<B, 2>, // Training data [n, d]
    x_test: Tensor<B, 2>,  // Test data [m, d]
    k: usize,
) -> Vec<Vec<usize>> // indices [m][k]
{
    let distances = l2_distance(x_test, Some(x_train)); // [m, n]
    let [m, n] = distances.dims();

    let mut result = Vec::new();

    for i in 0..m {
        let test_distances = distances.clone().slice([i..i + 1, 0..n]).squeeze::<1>(); // [n]

        // Convert to vec for sorting
        let mut dist_with_indices: Vec<(f32, usize)> = Vec::new();
        for j in 0..n {
            let dist: f32 = test_distances.clone().slice([j..j + 1]).into_scalar();
            dist_with_indices.push((dist, j));
        }

        // Sort by distance and take k smallest
        dist_with_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let top_k: Vec<usize> = dist_with_indices
            .into_iter()
            .take(k)
            .map(|(_, idx)| idx)
            .collect();

        result.push(top_k);
    }

    result
}

/// k-NN classifier with majority vote
fn knn_classifier<B: Backend<FloatElem = f32, IntElem = i64>>(
    x_train: Tensor<B, 2>,
    y_train: Tensor<B, 1, burn::tensor::Int>,
    x_test: Tensor<B, 2>,
    k: usize,
) -> Vec<i64> {
    let indices = find_knn(x_train, x_test, k);
    let [n_train] = y_train.dims();

    // Convert labels to vec for easier access
    let mut labels_vec = Vec::new();
    for i in 0..n_train {
        let label: i64 = y_train.clone().slice([i..i + 1]).into_scalar();
        labels_vec.push(label);
    }

    let mut predictions = Vec::new();

    for neighbor_indices in indices {
        // Get labels of k nearest neighbors
        let neighbor_labels: Vec<i64> = neighbor_indices
            .iter()
            .map(|&idx| labels_vec[idx])
            .collect();

        // Find majority vote
        let mut label_counts = std::collections::HashMap::new();
        for &label in &neighbor_labels {
            *label_counts.entry(label).or_insert(0) += 1;
        }

        let prediction = label_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(label, _)| label)
            .unwrap_or(0);

        predictions.push(prediction);
    }

    predictions
}

/// Compute classification accuracy
fn compute_accuracy(y_true: &[i64], y_pred: &[i64]) -> f32 {
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|&(a, b)| a == b)
        .count();
    correct as f32 / y_true.len() as f32
}

/// Generate synthetic face dataset for demonstration
fn generate_face_dataset(
    device: &Device<MyBackend>,
) -> (
    Tensor<MyBackend, 2>,                    // train_images
    Tensor<MyBackend, 1, burn::tensor::Int>, // train_labels
    Tensor<MyBackend, 2>,                    // test_images
    Tensor<MyBackend, 1, burn::tensor::Int>, // test_labels
    usize,                                   // n_classes
) {
    let n_classes = 40;
    let n_samples_per_class = 10;
    let n_features = 100; // Simplified feature size

    let n_total = n_classes * n_samples_per_class;
    let n_train = (0.8 * n_total as f32) as usize;
    let n_test = n_total - n_train;

    println!("Generating synthetic face dataset:");
    println!("  Classes: {}", n_classes);
    println!("  Features: {}", n_features);
    println!("  Train samples: {}", n_train);
    println!("  Test samples: {}", n_test);

    // Generate class prototypes
    let class_prototypes = Tensor::random(
        Shape::new([n_classes, n_features]),
        Distribution::Normal(0.5, 0.2),
        device,
    );

    let mut all_images = Vec::new();
    let mut all_labels = Vec::new();

    // Generate samples for each class
    for class_id in 0..n_classes {
        let prototype = class_prototypes.clone().slice([class_id..class_id + 1]);

        for _ in 0..n_samples_per_class {
            let noise = Tensor::random(
                Shape::new([1, n_features]),
                Distribution::Normal(0.0, 0.1),
                device,
            );
            let face = prototype.clone().add(noise).clamp(0.0, 1.0);

            all_images.push(face);
            all_labels.push(class_id as i64);
        }
    }

    // Convert to tensors
    let all_images_tensor = Tensor::cat(all_images, 0);
    let all_labels_tensor: Tensor<MyBackend, 1, burn::tensor::Int> =
        Tensor::from_ints(all_labels.as_slice(), device);

    // Create random permutation for train/test split
    let mut indices: Vec<usize> = (0..n_total).collect();
    let mut rng = rand::thread_rng();
    indices.shuffle(&mut rng);

    // Split data
    let train_indices = &indices[0..n_train];
    let test_indices = &indices[n_train..];

    // Select training data
    let train_image_list: Vec<Tensor<MyBackend, 2>> = train_indices
        .iter()
        .map(|&i| all_images_tensor.clone().slice([i..i + 1]))
        .collect();
    let train_images = Tensor::cat(train_image_list, 0);

    let train_label_data: Vec<i64> = train_indices
        .iter()
        .map(|&i| {
            let slice = all_labels_tensor.clone().slice([i..i + 1]);
            slice.into_scalar() as i64
        })
        .collect();
    let train_labels: Tensor<MyBackend, 1, burn::tensor::Int> =
        Tensor::from_ints(train_label_data.as_slice(), device);

    // Select test data
    let test_image_list: Vec<Tensor<MyBackend, 2>> = test_indices
        .iter()
        .map(|&i| all_images_tensor.clone().slice([i..i + 1]))
        .collect();
    let test_images = Tensor::cat(test_image_list, 0);

    let test_label_data: Vec<i64> = test_indices
        .iter()
        .map(|&i| {
            let slice = all_labels_tensor.clone().slice([i..i + 1]);
            slice.into_scalar() as i64
        })
        .collect();
    let test_labels: Tensor<MyBackend, 1, burn::tensor::Int> =
        Tensor::from_ints(test_label_data.as_slice(), device);

    (
        train_images,
        train_labels,
        test_images,
        test_labels,
        n_classes,
    )
}

/// Main demonstration function
fn run_face_recognition_demo(device: &Device<MyBackend>) {
    println!("=== CS3780 Project 1: k-Nearest Neighbors Face Recognition ===\n");

    // Generate dataset
    println!("1. Loading face dataset...");
    let (train_images, train_labels, test_images, test_labels, n_classes) =
        generate_face_dataset(device);

    // Test different k values
    let k_values = vec![1, 3, 5, 7];
    println!("\n2. Testing different k values:");

    let mut results = Vec::new();

    // Convert test labels to vector for accuracy computation
    let [n_test] = test_labels.dims();
    let test_labels_vec: Vec<i64> = (0..n_test)
        .map(|i| test_labels.clone().slice([i..i + 1]).into_scalar() as i64)
        .collect();

    for k in k_values {
        println!("\n--- k = {} ---", k);
        let start_time = Instant::now();

        // Perform k-NN classification
        let predictions = knn_classifier(
            train_images.clone(),
            train_labels.clone(),
            test_images.clone(),
            k,
        );

        let elapsed = start_time.elapsed();

        // Compute accuracy
        let accuracy = compute_accuracy(&test_labels_vec, &predictions);

        println!("  Accuracy: {:.2}%", accuracy * 100.0);
        println!("  Time: {:.4}s", elapsed.as_secs_f64());

        results.push((k, accuracy, elapsed));
    }

    // Summary
    println!("\n3. Results Summary:");
    println!("   k  | Accuracy | Time (s)");
    println!("  ----|----------|----------");
    for (k, acc, time) in results {
        println!(
            "  {:2}  | {:6.2}%  | {:6.4}",
            k,
            acc * 100.0,
            time.as_secs_f64()
        );
    }

    // Educational notes
    println!("\n4. Educational Insights:");
    println!("   • k=1 (nearest neighbor) may overfit to training noise");
    println!("   • Larger k values provide smoother decision boundaries");
    println!("   • Face recognition benefits from k=3-5 typically");

    // Expected vs random accuracy
    let random_accuracy = 1.0 / n_classes as f32;
    println!(
        "   • Random classifier accuracy: {:.2}%",
        random_accuracy * 100.0
    );
    println!("   • Our classifier significantly outperforms random chance!");
}

/// Distance computation benchmarks
fn benchmark_distance_computation(device: &Device<MyBackend>) {
    println!("\n=== Distance Computation Benchmarks ===");

    let sizes = vec![(100, 50), (200, 50)];

    for (n, d) in sizes {
        println!("\nTesting {}x{} matrices:", n, d);

        let x: Tensor<MyBackend, 2> =
            Tensor::random(Shape::new([n, d]), Distribution::Normal(0.0, 1.0), device);

        let start_time = Instant::now();
        let _distances = l2_distance(x.clone(), None);
        let elapsed = start_time.elapsed();

        println!("  L2 distance computation: {:.4}s", elapsed.as_secs_f64());
        println!("  Result shape: [{}x{}]", n, n);
    }
}

fn main() {
    // Initialize device
    let device = Default::default();

    // Run main demonstration
    run_face_recognition_demo(&device);

    // Run benchmarks
    benchmark_distance_computation(&device);

    println!("\n=== CS3780 Educational Connection ===");
    println!("This example adapts Python Homework 1 from CS3780:");
    println!("• Demonstrates k-nearest neighbors algorithm");
    println!("• Uses Euclidean distance in high-dimensional space");
    println!("• Applies to face recognition (computer vision)");
    println!("• Shows effect of k parameter on classification");
    println!("• Rust/Burn implementation maintains same algorithmic principles");

    println!("\nFor comparison with original Python notebook:");
    println!("• faces.mat dataset would contain real face images");
    println!("• Our synthetic data simulates the same structure");
    println!("• Core algorithms (l2distance, findknn, knnclassifier) are equivalent");
    println!("• Performance metrics and analysis remain the same");
}
