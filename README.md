#  Burn CS3780: Machine Learning Framework

A comprehensive machine learning library implementing all CS3780 concepts using the **Burn** deep learning framework. This project demonstrates how to build production-quality machine learning algorithms from scratch using Rust's powerful type system and Burn's flexible tensor operations.

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)
[![Burn](https://img.shields.io/badge/burn-0.20-red.svg)](https://burn.dev)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Features

### Classical Machine Learning
- **k-Nearest Neighbors (k-NN)** - Classification and regression with multiple distance metrics
- **Decision Trees** - Classification trees with multiple splitting criteria
- **Linear Models** - Linear and logistic regression with Ridge/Lasso/ElasticNet regularization
- **Perceptron Algorithm** - Single and multi-class perceptron with learning demonstrations
- **Support Vector Machines (SVM)** - With multiple kernel functions (RBF, polynomial, linear)
- **Kernel Methods** - Comprehensive kernel library for non-linear transformations

### Deep Learning
- **Neural Networks** - Multi-layer perceptrons with various activation functions
- **Convolutional Neural Networks (CNNs)** - For image classification tasks
- **Recurrent Neural Networks (RNNs)** - For sequence modeling
- **Transformers** - Attention mechanisms and transformer architectures
- **Autoencoders** - Dimensionality reduction and feature learning

### Ensemble Methods
- **Boosting** - AdaBoost and Gradient Boosting implementations
- **Random Forests** - Ensemble of decision trees

### Unsupervised Learning
- **k-Means Clustering** - Partitional clustering algorithm
- **Principal Component Analysis (PCA)** - Dimensionality reduction
- **t-SNE** - Non-linear dimensionality reduction

### Optimization & Training
- **Gradient Descent Variants** - SGD, Adam, AdaGrad with learning rate scheduling
- **Online Learning** - Streaming algorithms for large datasets
- **Cross-Validation** - Model selection and hyperparameter tuning
- **Regularization** - L1/L2 regularization and early stopping

## Installation

### Prerequisites
- Rust 1.75 or later
- Cargo package manager

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/burn-cs3780.git
cd burn-cs3780

# Run a specific example
cargo run --bin knn_example

# Run all tests
cargo test

# Run with GPU acceleration (if available)
cargo run --features wgpu --bin neural_nets_example

# Build for WebAssembly
cargo build --target wasm32-unknown-unknown --features wasm-web
```

## 🎯 Usage Examples

### k-Nearest Neighbors

```rust
use burn_cs3780::{DefaultBackend, datasets, models::KNearestNeighbors};

// Create dataset
let device = Default::default();
let dataset = datasets::make_linearly_separable::<DefaultBackend>(200, &device, Some(42));
let (train_data, test_data) = dataset.train_test_split(0.8, Some(42));

// Train k-NN classifier
let mut knn = KNearestNeighbors::new(5)
    .with_distance_metric(DistanceMetric::Euclidean)
    .with_weights(WeightFunction::Distance);

knn.fit(train_data.features, train_data.labels.squeeze(1));

// Make predictions
let predictions = knn.predict(&test_data.features);

// Evaluate
use burn_cs3780::metrics::ClassificationMetrics;
let accuracy = ClassificationMetrics::accuracy(&test_data.labels.squeeze(1), &predictions);
println!("Accuracy: {:.4}", accuracy);
```

### Linear Regression with Regularization

```rust
use burn_cs3780::models::{LinearRegression, Regularization, Solver};

let mut model = LinearRegression::new()
    .with_regularization(Regularization::Ridge { alpha: 0.1 })
    .with_solver(Solver::Adam)
    .with_intercept(true);

model.fit(&x_train, &y_train);
let predictions = model.predict(&x_test);

// Get model parameters
let coefficients = model.coef().unwrap();
let intercept = model.intercept().unwrap();
```

### Neural Network

```rust
use burn_cs3780::models::MLP;

let model = MLP::new()
    .add_layer(64, Activation::ReLU)
    .add_layer(32, Activation::ReLU) 
    .add_layer(10, Activation::Softmax)
    .with_optimizer(OptimizerType::Adam { lr: 0.001 });

let trained_model = model.fit(&train_data, 100, 32).unwrap();
let predictions = trained_model.predict(&test_data);
```

##  Running Examples

### Available Examples

```bash
# Classical ML
cargo run --bin knn_example              # k-Nearest Neighbors
cargo run --bin decision_trees_example   # Decision Trees  
cargo run --bin linear_regression_example # Linear Regression
cargo run --bin logistic_regression_example # Logistic Regression
cargo run --bin perceptron_example        # Perceptron Algorithm
cargo run --bin svm_example              # Support Vector Machines
cargo run --bin kernels_example          # Kernel Methods

# Deep Learning  
cargo run --bin neural_nets_example      # Neural Networks
cargo run --bin transformers_example     # Transformers
cargo run --bin autoencoders_example     # Autoencoders

# Ensemble Methods
cargo run --bin boosting_example         # Boosting
cargo run --bin clustering_example       # k-Means Clustering

# Unsupervised Learning
cargo run --bin pca_example             # Principal Component Analysis
cargo run --bin online_learning_example # Online Learning
cargo run --bin optimization_example    # Optimization Techniques
```

### Interactive Examples

Some examples generate visualizations and interactive plots:

```bash
# Generate decision boundary plots
cargo run --bin knn_example -- --plot --output plots/

# Show learning curves  
cargo run --bin perceptron_example -- --demo --visualize

# Compare algorithms
cargo run --bin comparison_example
```

##  Datasets

The library includes generators for common ML datasets:

- **Synthetic Classification**: Linearly separable, XOR, blobs
- **Synthetic Regression**: Polynomial, sinusoidal with noise
- **Real Datasets**: Iris, Boston Housing, Wine (via CSV loaders)

```rust
use burn_cs3780::datasets;

// Generate XOR dataset (non-linearly separable)
let xor_data = datasets::make_xor_dataset::<Backend>(200, 0.1, &device, Some(42));

// Generate polynomial regression data
let poly_data = datasets::make_polynomial_regression::<Backend>(100, 3, 0.2, &device, None);

// Generate clustering data
let cluster_data = datasets::make_blobs::<Backend>(300, 4, 1.5, &device, Some(123));
```

##  Configuration

### Backend Selection

```toml
[features]
default = ["ndarray"]
ndarray = ["burn/ndarray"]    # CPU backend
wgpu = ["burn/wgpu"]         # GPU backend  
wasm-web = ["burn/wgpu"]     # WebAssembly
```

### Custom Backends

```rust
// CPU Backend
type CpuBackend = burn::backend::NdArray<f32>;

// GPU Backend with autodifferentiation
type GpuBackend = burn::backend::Autodiff<burn::backend::Wgpu<f32>>;

// Use with any model
let mut knn = KNearestNeighbors::<GpuBackend>::new(5);
```

##  Mathematical Foundation

### Algorithms Implemented

| Algorithm | Type | Key Features |
|-----------|------|--------------|
| k-NN | Instance-based | Multiple distance metrics, weighted voting |
| Linear Regression | Parametric | Normal equation, gradient descent, regularization |
| Logistic Regression | Probabilistic | Sigmoid activation, cross-entropy loss |
| Perceptron | Linear | Online learning, convergence guarantees |
| SVM | Margin-based | Kernel trick, soft margins, SMO algorithm |
| Neural Networks | Connectionist | Backpropagation, various activations |
| k-Means | Clustering | Lloyd's algorithm, k-means++ initialization |
| PCA | Dimensionality reduction | Eigenvalue decomposition, variance explained |

### Optimization Methods

- **Gradient Descent**: Batch, stochastic, mini-batch variants
- **Advanced Optimizers**: Adam, AdaGrad, RMSprop with momentum
- **Learning Rate Scheduling**: Step decay, exponential decay, cosine annealing
- **Regularization**: L1 (Lasso), L2 (Ridge), ElasticNet, dropout

## Performance

### Benchmarks

Run benchmarks to compare algorithm performance:

```bash
cargo bench
```

### Memory Efficiency

Thanks to Burn's tensor operations and Rust's zero-cost abstractions:
- **Low memory footprint**: Minimal runtime overhead
- **Cache efficiency**: Optimized memory access patterns  
- **Vectorized operations**: SIMD instructions when available

##  Web Deployment

Deploy models to the browser using WebAssembly:

```bash
# Build for web
wasm-pack build --target web --out-dir pkg

# Serve demo
cd examples/web-demo
python -m http.server 8000
```

See `examples/mnist-inference-web/` for a complete browser demo.

##  Testing

Comprehensive test suite covering:

- **Unit tests**: Individual algorithm correctness
- **Integration tests**: End-to-end workflows
- **Property tests**: Mathematical invariants
- **Benchmarks**: Performance comparisons

```bash
cargo test                    # Run all tests
cargo test --doc             # Run documentation tests
cargo test models::knn       # Run specific module tests
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/your-username/burn-cs3780.git
cd burn-cs3780

# Install development dependencies
cargo install cargo-watch criterion

# Run tests on file changes  
cargo watch -x test

# Format code
cargo fmt

# Lint code
cargo clippy
```

### Adding New Algorithms

1. Create module in `src/models/`
2. Implement traits: `Fit`, `Predict`, `Transform` 
3. Add comprehensive tests
4. Create example in `examples/`
5. Update documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- **Burn Team** - For the excellent deep learning framework
- **CS3780 Course** - For the comprehensive ML curriculum  
- **Rust Community** - For the amazing ecosystem

---

**Happy Learning! **
