//! # Burn CS3780: Machine Learning Framework
//!
//! A comprehensive machine learning library implementing all CS3780 concepts using the Burn framework.
//! This library provides implementations of classical machine learning algorithms, deep learning models,
//! and optimization techniques with a focus on performance, correctness, and educational value.
//!
//! ## Features
//!
//! ### Classical Machine Learning
//! - k-Nearest Neighbors (k-NN)
//! - Decision Trees
//! - Linear and Logistic Regression
//! - Perceptron Algorithm
//! - Support Vector Machines (SVM)
//! - Kernel Methods
//!
//! ### Deep Learning
//! - Neural Networks (MLPs)
//! - Convolutional Neural Networks (CNNs)
//! - Recurrent Neural Networks (RNNs)
//! - Transformers
//! - Autoencoders
//!
//! ### Ensemble Methods
//! - Boosting (AdaBoost, Gradient Boosting)
//! - Random Forests
//!
//! ### Unsupervised Learning
//! - k-Means Clustering
//! - Principal Component Analysis (PCA)
//! - t-SNE
//!
//! ### Optimization & Training
//! - Gradient Descent variants (SGD, Adam, AdaGrad)
//! - Online Learning algorithms
//! - Cross-validation
//! - Regularization techniques
//!
//! ## Backend Support
//!
//! Thanks to Burn's flexible backend system, all algorithms can run on:
//! - CPU (NdArray backend)
//! - GPU (WGPU backend)  
//! - WebAssembly (for browser deployment)
//!
//! ## Example Usage
//!
//! ```rust
//! use burn_cs3780::models::KNearestNeighbors;
//! use burn::backend::NdArray;
//!
//! type Backend = NdArray<f32>;
//!
//! // Create and train a k-NN classifier
//! let knn = KNearestNeighbors::<Backend>::new(5);
//! // ... training and inference code
//! ```

#![warn(missing_docs)]

pub mod datasets;
pub mod kernels;
pub mod metrics;
pub mod models;
pub mod optimizers;
pub mod utils;

/// wasm-bindgen wrappers so the browser can drive these models directly.
#[cfg(target_arch = "wasm32")]
pub mod wasm;

// Re-export commonly used types
pub use burn;
#[cfg(not(target_arch = "wasm32"))]
pub use burn::backend::Wgpu;
pub use burn::backend::{Autodiff, NdArray};

/// Type alias for the default backend (CPU)
pub type DefaultBackend = NdArray<f32>;

/// Type alias for backend with autodifferentiation
pub type DefaultAutodiffBackend = Autodiff<NdArray<f32>>;

/// Type alias for GPU backend
#[cfg(not(target_arch = "wasm32"))]
pub type GpuBackend = Wgpu<f32, i32>;

/// Type alias for GPU backend with autodifferentiation
#[cfg(not(target_arch = "wasm32"))]
pub type GpuAutodiffBackend = Autodiff<Wgpu<f32, i32>>;
