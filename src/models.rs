//! Machine learning models implemented using Burn.
//!
//! This module contains implementations of all the machine learning algorithms
//! covered in CS3780, from classical algorithms like k-NN to deep learning models.

pub mod autoencoders;
pub mod boosting;
pub mod clustering;
pub mod cnn;
pub mod decision_tree;
pub mod knn;
pub mod linear_models;
/// Naive Bayes text classifiers
pub mod naive_bayes;
pub mod neural_networks;
pub mod online_learning;
pub mod pca;
pub mod perceptron;
pub mod svm;
pub mod transformers;

// Re-export all models for convenience
pub use autoencoders::*;
pub use boosting::*;
pub use clustering::*;
pub use cnn::*;
pub use decision_tree::*;
pub use knn::*;
pub use linear_models::*;
pub use naive_bayes::*;
pub use neural_networks::*;
pub use online_learning::*;
pub use pca::*;
pub use perceptron::*;
pub use svm::*;
pub use transformers::*;
