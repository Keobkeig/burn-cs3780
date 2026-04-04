/*!
# Project 3: Support Vector Machines (Linear SVM)

This example demonstrates linear Support Vector Machines with margin maximization,
adapted from CS3780 concepts. It showcases:

- Linear SVM with primal formulation optimization
- Maximum margin classification with support vectors
- Hinge loss and regularization for soft-margin SVM
- Comparison of hard vs soft margin approaches
- Applications to linearly separable and non-separable data

## Educational Objectives

This implementation teaches:
- The geometric intuition behind SVMs and maximum margin classification
- Primal formulation of SVM optimization problem
- Support vectors and their role in decision boundaries
- Hinge loss for handling non-separable data
- Trade-off between margin maximization and classification errors (C parameter)

## CS3780 Connection

This implements the linear SVM concepts from CS3780 Project 3, focusing on:
- Understanding margin maximization principle
- Primal SVM formulation before kernel methods
- Comparison with perceptron and linear classifiers
- Foundation for kernelized SVMs (covered in 08_kernel_methods)

## Usage

```bash
cargo run --example 05_svm
```

The example will:
1. Implement linear SVM with gradient-based optimization
2. Generate linearly separable and non-separable datasets
3. Train SVMs with different regularization parameters
4. Demonstrate support vector identification
5. Compare decision boundaries with other linear classifiers
*/

use burn::tensor::{backend::Backend, Device, Shape, Tensor};
use burn_cs3780::DefaultBackend;
use rand::Rng;
use std::time::Instant;

type MyBackend = DefaultBackend;

/// Linear SVM implementation
#[derive(Debug, Clone)]
pub struct LinearSVM<B: Backend<FloatElem = f32>> {
    /// Weight vector [d]
    weights: Option<Tensor<B, 1>>,
    /// Bias term
    bias: f32,
    /// Regularization parameter
    c_param: f32,
    /// Maximum number of training epochs
    max_epochs: usize,
    /// Learning rate for gradient descent
    learning_rate: f32,
    /// Device for computations
    device: Device<B>,
}

impl<B: Backend<FloatElem = f32>> LinearSVM<B> {
    /// Create new linear SVM
    pub fn new(c_param: f32, learning_rate: f32, max_epochs: usize, device: &Device<B>) -> Self {
        Self {
            weights: None,
            bias: 0.0,
            c_param,
            max_epochs,
            learning_rate,
            device: device.clone(),
        }
    }

    /// Train the SVM using gradient descent on hinge loss + regularization
    pub fn fit(&mut self, x_train: Tensor<B, 2>, y_train: &[f32]) {
        let [n, d] = x_train.dims();
        assert_eq!(
            y_train.len(),
            n,
            "Number of samples must match number of labels"
        );

        println!("Training Linear SVM:");
        println!("  Training samples: {}", n);
        println!("  Feature dimension: {}", d);
        println!("  C parameter: {:.4}", self.c_param);
        println!("  Learning rate: {:.6}", self.learning_rate);
        println!("  Max epochs: {}", self.max_epochs);

        // Initialize weights and bias
        let mut weights = Tensor::random(
            Shape::new([d]),
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &self.device,
        );
        let mut bias = 0.0f32;

        let y_tensor: Tensor<B, 1> = Tensor::from_floats(y_train, &self.device);

        for epoch in 0..self.max_epochs {
            // Forward pass: compute predictions
            let predictions = x_train
                .clone()
                .matmul(weights.clone().unsqueeze_dim::<2>(1))
                .squeeze::<1>()
                .add_scalar(bias);

            // Compute hinge loss and gradients
            let margin = y_tensor.clone().mul(predictions.clone());
            let ones = Tensor::ones(Shape::new([n]), &self.device);
            let hinge_input = ones.sub(margin);

            // Hinge loss: max(0, 1 - y * (w^T x + b))
            let zero_tensor = Tensor::zeros(Shape::new([n]), &self.device);
            let hinge_loss = hinge_input.max_pair(zero_tensor.clone());

            // Total loss: regularization + C * hinge_loss
            let reg_loss = weights.clone().powf_scalar(2.0).sum().mul_scalar(0.5);
            let total_loss = reg_loss.add(hinge_loss.clone().mean().mul_scalar(self.c_param));

            // Compute gradients
            // For samples with hinge_loss > 0: gradient includes data term
            // For samples with hinge_loss = 0: gradient is just regularization

            // Weight gradient: w - C * sum(y_i * x_i) for support vectors
            let mut weight_grad = weights.clone(); // Regularization term
            let mut num_support_vectors = 0;

            // Add data term for support vectors
            for i in 0..n {
                let hinge_i = hinge_loss.clone().slice([i..i + 1]).into_scalar();
                if hinge_i > 1e-6 {
                    let y_i = y_train[i];
                    let x_i = x_train.clone().slice([i..i + 1]).squeeze::<1>();
                    let data_grad = x_i.mul_scalar(-self.c_param * y_i);
                    weight_grad = weight_grad.add(data_grad);
                    num_support_vectors += 1;
                }
            }

            // Bias gradient: -C * sum(y_i) for support vectors
            let mut bias_grad = 0.0f32;
            for i in 0..n {
                let hinge_i = hinge_loss.clone().slice([i..i + 1]).into_scalar();
                if hinge_i > 1e-6 {
                    bias_grad -= self.c_param * y_train[i];
                }
            }

            // Update parameters
            weights = weights.sub(weight_grad.mul_scalar(self.learning_rate));
            bias = bias - self.learning_rate * bias_grad;

            // Print progress
            if epoch % (self.max_epochs / 10) == 0 || epoch == self.max_epochs - 1 {
                let loss_val = total_loss.into_scalar();
                println!(
                    "  Epoch {}: Loss = {:.6}, Support Vectors = {}",
                    epoch, loss_val, num_support_vectors
                );
            }
        }

        self.weights = Some(weights);
        self.bias = bias;
    }

    /// Make predictions on new data
    pub fn predict(&self, x_test: Tensor<B, 2>) -> Vec<f32> {
        let weights = self.weights.as_ref().expect("SVM must be trained first");
        let predictions = x_test
            .matmul(weights.clone().unsqueeze_dim::<2>(1))
            .squeeze::<1>()
            .add_scalar(self.bias);

        let [m] = predictions.dims();
        let mut result = Vec::with_capacity(m);

        for i in 0..m {
            let pred_val = predictions.clone().slice([i..i + 1]).into_scalar();
            result.push(if pred_val > 0.0 { 1.0 } else { -1.0 });
        }

        result
    }

    /// Get decision function values (before applying sign)
    pub fn decision_function(&self, x_test: Tensor<B, 2>) -> Vec<f32> {
        let weights = self.weights.as_ref().expect("SVM must be trained first");
        let predictions = x_test
            .matmul(weights.clone().unsqueeze_dim::<2>(1))
            .squeeze::<1>()
            .add_scalar(self.bias);

        let [m] = predictions.dims();
        let mut result = Vec::with_capacity(m);

        for i in 0..m {
            let pred_val = predictions.clone().slice([i..i + 1]).into_scalar();
            result.push(pred_val);
        }

        result
    }

    /// Identify support vectors from training data
    pub fn find_support_vectors(&self, x_train: Tensor<B, 2>, y_train: &[f32]) -> Vec<usize> {
        let weights = self.weights.as_ref().expect("SVM must be trained first");
        let [n, _] = x_train.dims();

        let predictions = x_train
            .matmul(weights.clone().unsqueeze_dim::<2>(1))
            .squeeze::<1>()
            .add_scalar(self.bias);
        let y_tensor: Tensor<B, 1> = Tensor::from_floats(y_train, &self.device);

        let margin = y_tensor.mul(predictions);
        let ones = Tensor::ones(Shape::new([n]), &self.device);
        let hinge_input = ones.sub(margin);
        let zero_tensor = Tensor::zeros(Shape::new([n]), &self.device);
        let hinge_loss = hinge_input.max_pair(zero_tensor);

        let mut support_vectors = Vec::new();
        for i in 0..n {
            let loss_i = hinge_loss.clone().slice([i..i + 1]).into_scalar();
            if loss_i > 1e-6 {
                support_vectors.push(i);
            }
        }

        support_vectors
    }
}

/// Generate linearly separable 2D dataset
fn generate_separable_data(
    n_per_class: usize,
    device: &Device<MyBackend>,
) -> (Tensor<MyBackend, 2>, Vec<f32>) {
    println!("\nGenerating linearly separable dataset:");
    println!("  Samples per class: {}", n_per_class);

    let mut rng = rand::thread_rng();
    let mut data_points = Vec::new();
    let mut labels = Vec::new();

    // Class +1: centered at (2, 2) with some spread
    for _ in 0..n_per_class {
        let x = 2.0 + rng.gen::<f32>() * 1.5 - 0.75; // [1.25, 2.75]
        let y = 2.0 + rng.gen::<f32>() * 1.5 - 0.75;
        data_points.push([x, y]);
        labels.push(1.0);
    }

    // Class -1: centered at (-1, -1) with some spread
    for _ in 0..n_per_class {
        let x = -1.0 + rng.gen::<f32>() * 1.5 - 0.75; // [-1.75, -0.25]
        let y = -1.0 + rng.gen::<f32>() * 1.5 - 0.75;
        data_points.push([x, y]);
        labels.push(-1.0);
    }

    // Convert to tensor
    let total_samples = data_points.len();
    let data_vec: Vec<f32> = data_points.into_iter().flatten().collect();
    let data_tensor: Tensor<MyBackend, 2> =
        Tensor::<MyBackend, 1>::from_floats(data_vec.as_slice(), device)
            .reshape(Shape::new([total_samples, 2]));

    println!("  Total samples: {}", total_samples);
    println!("  Feature dimension: 2");

    (data_tensor, labels)
}

/// Generate non-linearly separable 2D dataset (overlapping classes)
fn generate_overlapping_data(
    n_per_class: usize,
    device: &Device<MyBackend>,
) -> (Tensor<MyBackend, 2>, Vec<f32>) {
    println!("\nGenerating overlapping dataset:");
    println!("  Samples per class: {}", n_per_class);

    let mut rng = rand::thread_rng();
    let mut data_points = Vec::new();
    let mut labels = Vec::new();

    // Class +1: centered at (1, 1) with larger spread
    for _ in 0..n_per_class {
        let x = 1.0 + rng.gen::<f32>() * 3.0 - 1.5; // [-0.5, 2.5]
        let y = 1.0 + rng.gen::<f32>() * 3.0 - 1.5;
        data_points.push([x, y]);
        labels.push(1.0);
    }

    // Class -1: centered at (-0.5, -0.5) with larger spread (overlapping)
    for _ in 0..n_per_class {
        let x = -0.5 + rng.gen::<f32>() * 3.0 - 1.5; // [-2.0, 1.0]
        let y = -0.5 + rng.gen::<f32>() * 3.0 - 1.5;
        data_points.push([x, y]);
        labels.push(-1.0);
    }

    // Convert to tensor
    let total_samples = data_points.len();
    let data_vec: Vec<f32> = data_points.into_iter().flatten().collect();
    let data_tensor: Tensor<MyBackend, 2> =
        Tensor::<MyBackend, 1>::from_floats(data_vec.as_slice(), device)
            .reshape(Shape::new([total_samples, 2]));

    println!("  Total samples: {}", total_samples);
    println!("  Feature dimension: 2");

    (data_tensor, labels)
}

/// Compute classification accuracy
fn compute_accuracy(y_pred: &[f32], y_true: &[f32]) -> f32 {
    assert_eq!(y_pred.len(), y_true.len());
    let correct = y_pred
        .iter()
        .zip(y_true.iter())
        .filter(|&(pred, true_label)| pred == true_label)
        .count();
    correct as f32 / y_pred.len() as f32
}

/// Test SVM on separable data with different C values
fn test_svm_separable_data(device: &Device<MyBackend>) {
    println!("\n=== Testing Linear SVM on Separable Data ===");

    let (data, labels) = generate_separable_data(50, device);

    // Train/test split
    let total_samples = data.dims()[0];
    let train_size = (total_samples as f32 * 0.8) as usize;

    let train_data: Vec<Tensor<MyBackend, 2>> = (0..train_size)
        .map(|i| data.clone().slice([i..i + 1]))
        .collect();
    let x_train = Tensor::cat(train_data, 0);
    let y_train = labels[0..train_size].to_vec();

    let test_data: Vec<Tensor<MyBackend, 2>> = (train_size..total_samples)
        .map(|i| data.clone().slice([i..i + 1]))
        .collect();
    let x_test = Tensor::cat(test_data, 0);
    let y_test = labels[train_size..].to_vec();

    // Test different C values
    let c_values = [0.1, 1.0, 10.0];

    for &c_val in &c_values {
        println!("\n--- Testing C = {:.1} ---", c_val);

        let start_time = Instant::now();
        let mut svm = LinearSVM::new(c_val, 0.01, 1000, device);
        svm.fit(x_train.clone(), &y_train);
        let training_time = start_time.elapsed();

        let train_pred = svm.predict(x_train.clone());
        let test_pred = svm.predict(x_test.clone());

        let train_accuracy = compute_accuracy(&train_pred, &y_train);
        let test_accuracy = compute_accuracy(&test_pred, &y_test);

        let support_vectors = svm.find_support_vectors(x_train.clone(), &y_train);

        println!("Results:");
        println!("  Training accuracy: {:.1}%", train_accuracy * 100.0);
        println!("  Test accuracy: {:.1}%", test_accuracy * 100.0);
        println!(
            "  Support vectors: {} / {}",
            support_vectors.len(),
            y_train.len()
        );
        println!("  Training time: {:.4}s", training_time.as_secs_f64());

        if train_accuracy > 0.95 && test_accuracy > 0.9 {
            println!("  ✓ SVM successfully separated linearly separable data!");
        }
    }
}

/// Test SVM on overlapping data to demonstrate soft margin
fn test_svm_overlapping_data(device: &Device<MyBackend>) {
    println!("\n=== Testing Linear SVM on Overlapping Data ===");

    let (data, labels) = generate_overlapping_data(75, device);

    // Train/test split
    let total_samples = data.dims()[0];
    let train_size = (total_samples as f32 * 0.8) as usize;

    let train_data: Vec<Tensor<MyBackend, 2>> = (0..train_size)
        .map(|i| data.clone().slice([i..i + 1]))
        .collect();
    let x_train = Tensor::cat(train_data, 0);
    let y_train = labels[0..train_size].to_vec();

    let test_data: Vec<Tensor<MyBackend, 2>> = (train_size..total_samples)
        .map(|i| data.clone().slice([i..i + 1]))
        .collect();
    let x_test = Tensor::cat(test_data, 0);
    let y_test = labels[train_size..].to_vec();

    // Test different C values to show soft margin effect
    let c_values = [0.01, 0.1, 1.0, 10.0];

    for &c_val in &c_values {
        println!("\n--- Testing C = {:.2} ---", c_val);

        let start_time = Instant::now();
        let mut svm = LinearSVM::new(c_val, 0.01, 1000, device);
        svm.fit(x_train.clone(), &y_train);
        let training_time = start_time.elapsed();

        let train_pred = svm.predict(x_train.clone());
        let test_pred = svm.predict(x_test.clone());

        let train_accuracy = compute_accuracy(&train_pred, &y_train);
        let test_accuracy = compute_accuracy(&test_pred, &y_test);

        let support_vectors = svm.find_support_vectors(x_train.clone(), &y_train);
        let sv_ratio = support_vectors.len() as f32 / y_train.len() as f32;

        println!("Results:");
        println!("  Training accuracy: {:.1}%", train_accuracy * 100.0);
        println!("  Test accuracy: {:.1}%", test_accuracy * 100.0);
        println!(
            "  Support vectors: {} / {} ({:.1}%)",
            support_vectors.len(),
            y_train.len(),
            sv_ratio * 100.0
        );
        println!("  Training time: {:.4}s", training_time.as_secs_f64());

        // Interpret results based on C value
        if c_val < 0.1 {
            println!("  → Low C: Emphasizes large margin, may underfit");
        } else if c_val > 1.0 {
            println!("  → High C: Emphasizes correct classification, may overfit");
        } else {
            println!("  → Moderate C: Good balance between margin and classification");
        }
    }
}

/// Educational insights about SVMs
fn print_educational_insights() {
    println!("\n=== Educational Insights ===");
    println!("Support Vector Machines (SVMs):");
    println!("• Maximum margin principle: find decision boundary with largest margin");
    println!("• Support vectors: training points that define the decision boundary");
    println!("• Margin: distance from decision boundary to nearest training points");
    println!("• Only support vectors affect the final classifier (sparsity)");

    println!("\nPrimal SVM Formulation:");
    println!("• Objective: minimize ½||w||² + C·Σξᵢ (margin + penalty)");
    println!("• Constraints: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0");
    println!("• Trade-off parameter C controls margin vs classification error");
    println!("• Hinge loss: ℓ(y, f(x)) = max(0, 1 - y·f(x))");

    println!("\nSoft vs Hard Margin:");
    println!("• Hard margin: requires perfect separation (C → ∞)");
    println!("• Soft margin: allows misclassification with penalty (finite C)");
    println!("• Low C: emphasizes large margin, may underfit");
    println!("• High C: emphasizes correct classification, may overfit");

    println!("\nGeometric Intuition:");
    println!("• Decision boundary: hyperplane w·x + b = 0");
    println!("• Margin boundaries: w·x + b = ±1");
    println!("• Support vectors lie on margin boundaries");
    println!("• Distance from point to hyperplane: |w·x + b| / ||w||");

    println!("\nComparison with Other Classifiers:");
    println!("• Perceptron: finds any separating hyperplane");
    println!("• SVM: finds the separating hyperplane with maximum margin");
    println!("• Logistic Regression: probabilistic, uses all data points");
    println!("• SVM: deterministic, uses only support vectors");
}

/// Main demonstration function
fn main() {
    println!("=== CS3780 Project 3: Linear Support Vector Machines ===\n");

    let device = Default::default();

    // Run demonstrations
    test_svm_separable_data(&device);
    test_svm_overlapping_data(&device);
    print_educational_insights();

    println!("\n=== CS3780 Educational Connection ===");
    println!("This example demonstrates fundamental SVM concepts from CS3780:");
    println!("• Maximum margin classification principle");
    println!("• Support vector identification and sparsity");
    println!("• Soft margin SVM for non-separable data");
    println!("• Effect of regularization parameter C");
    println!("• Foundation for kernel methods (see 08_kernel_methods)");

    println!("\nComparison with CS3780 concepts:");
    println!("• Primal formulation → minimize ½||w||² + C·Σξᵢ");
    println!("• Support vectors → training points that define decision boundary");
    println!("• Hinge loss → max(0, 1 - y(w·x + b))");
    println!("• Margin maximization → geometric interpretation of SVM");
    println!("• This linear SVM sets foundation for kernel methods");
}
