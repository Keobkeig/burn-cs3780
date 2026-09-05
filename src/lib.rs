//! # burn-cs3780
//!
//! Machine learning algorithms from Cornell's CS 3780, implemented from
//! scratch on the [Burn](https://burn.dev) tensor framework and compilable to
//! WebAssembly.
//!
//! ## What's here
//!
//! **Supervised** — k-nearest neighbors, decision trees, linear and logistic
//! regression (ridge / lasso / elastic net), perceptron (binary and
//! one-vs-rest), support vector machines via SMO, naive Bayes for text,
//! AdaBoost, and gradient boosting with optional row subsampling.
//!
//! **Neural** — multilayer perceptrons, convolutional networks, autoencoders
//! (plain, variational, denoising, sparse), a transformer encoder and
//! classifier, and a character-level language model with causal masking.
//!
//! **Unsupervised** — k-means (random or k-means++ seeding), PCA by power
//! iteration, and kernel ridge regression.
//!
//! **Supporting** — linear / polynomial / RBF / sigmoid kernels, SGD / Adam /
//! AdaGrad with step and exponential schedules, three online learners,
//! classification and regression metrics, preprocessing, and synthetic
//! dataset generators.
//!
//! Not implemented, despite being adjacent to the syllabus: recurrent networks,
//! random forests, t-SNE, RMSprop, cosine annealing, early stopping, and
//! cross-validation beyond k-fold index generation.
//!
//! ## Backends
//!
//! Where a tensor computes is a type parameter, so every model is written once
//! and runs on CPU ([`DefaultBackend`]), GPU ([`GpuBackend`], via wgpu) or with
//! gradients ([`DefaultAutodiffBackend`]). The WebAssembly build is the same
//! source with the CPU backend selected.
//!
//! ## Example
//!
//! ```rust
//! use burn_cs3780::models::KNearestNeighbors;
//! use burn_cs3780::{datasets, DefaultBackend};
//!
//! let device = Default::default();
//! let data = datasets::make_blobs::<DefaultBackend>(60, 3, 0.8, &device, Some(42));
//!
//! let mut knn = KNearestNeighbors::<DefaultBackend>::new(5);
//! knn.fit(data.features.clone(), data.labels.squeeze_dims::<1>(&[1]));
//!
//! let predictions = knn.predict(&data.features);
//! assert_eq!(predictions.dims()[0], 60);
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
