//! Browser bindings.
//!
//! Every function here is marshalling only: flat `Float32Array`s in, a
//! [`DemoResult`] out. The algorithms themselves live in [`crate::models`],
//! [`crate::kernels`] and [`crate::optimizers`] and are compiled unchanged —
//! the browser runs the same code paths as the native binaries.

#![allow(clippy::too_many_arguments)]

use burn::tensor::{backend::Backend, Tensor, TensorData};
use wasm_bindgen::prelude::*;

use crate::datasets;
use crate::kernels::{Kernel, LinearKernel, PolynomialKernel, RbfKernel, SigmoidKernel};
use crate::models::autoencoders::{ActivationType, Autoencoder, AutoencoderConfig};
use crate::models::boosting::{AdaBoostClassifier, AdaBoostConfig};
use crate::models::clustering::{InitMethod, KMeans, KMeansConfig};
use crate::models::cnn::{Cnn, CnnConfig};
use crate::models::decision_tree::{DecisionTree, SplitCriterion};
use crate::models::knn::{DistanceMetric, KNearestNeighbors, WeightFunction};
use crate::models::linear_models::{LinearRegression, LogisticRegression, Regularization, Solver};
use crate::models::naive_bayes::{
    extract_features, generate_synthetic_language_data, FeatureType, NaiveBayesClassifier,
};
use crate::models::neural_networks::NeuralNetClassifier;
use crate::models::online_learning::{
    OnlineLearner, OnlinePerceptron, OnlinePerceptronConfig, OnlineSGD, OnlineSGDConfig,
    PassiveAggressive, PassiveAggressiveConfig,
};
use crate::models::pca::PCA;
use crate::models::perceptron::Perceptron;
use crate::models::svm::{KernelType, SVM};
use crate::models::transformers::{
    make_harmony_words, make_letter_search_words, obeys_vowel_harmony, CharTokenizer,
    PositionEncoding, PositionEncodingConfig, TransformerClassifier, TransformerEncoderConfig,
    TransformerLanguageModel,
};
use crate::optimizers::{AdaGrad, Adam, Optimizer as ManualOptimizer, SGD};

/// Route Rust panics to `console.error` instead of a bare wasm trap.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// CPU backend used for every browser demo.
type Bk = burn::backend::NdArray<f32>;
/// Same backend wrapped in autodiff, for the demos that train by backprop.
type Ad = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

fn t2<B: Backend<FloatElem = f32>>(data: &[f32], cols: usize) -> Tensor<B, 2> {
    let rows = if cols == 0 { 0 } else { data.len() / cols };
    Tensor::from_data(
        TensorData::new(data[..rows * cols].to_vec(), [rows, cols]),
        &Default::default(),
    )
}

fn t1<B: Backend<FloatElem = f32>>(data: &[f32]) -> Tensor<B, 1> {
    Tensor::from_data(
        TensorData::new(data.to_vec(), [data.len()]),
        &Default::default(),
    )
}

fn flat<B: Backend<FloatElem = f32>, const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.to_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .unwrap_or_default()
}

/// Row-major grid of `res * res` sample points covering the given box.
/// Row `j` is a constant y, so JS can blit rows straight onto a canvas.
fn mesh<B: Backend<FloatElem = f32>>(
    res: usize,
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
) -> Tensor<B, 2> {
    let step = |a: f32, b: f32, i: usize| {
        if res <= 1 {
            a
        } else {
            a + (b - a) * (i as f32) / ((res - 1) as f32)
        }
    };
    let mut d = Vec::with_capacity(res * res * 2);
    for j in 0..res {
        for i in 0..res {
            d.push(step(x0, x1, i));
            d.push(step(y0, y1, j));
        }
    }
    Tensor::from_data(TensorData::new(d, [res * res, 2]), &Default::default())
}

/// Everything a demo can hand back to the page.
///
/// One shape for every algorithm — each demo fills the fields it has and
/// leaves the rest empty, so the JS side only ever learns one interface.
#[wasm_bindgen]
#[derive(Default)]
pub struct DemoResult {
    grid: Vec<f32>,
    points: Vec<f32>,
    labels: Vec<f32>,
    curve: Vec<f32>,
    frames: Vec<f32>,
    stats: Vec<f32>,
}

#[wasm_bindgen]
impl DemoResult {
    /// Scalar field sampled over a square mesh, row-major (decision values,
    /// class probabilities, kernel matrices, attention weights).
    #[wasm_bindgen(getter)]
    pub fn grid(&self) -> Vec<f32> {
        self.grid.clone()
    }
    /// Flat `x, y` pairs — data points, centroids, support vectors, axes.
    #[wasm_bindgen(getter)]
    pub fn points(&self) -> Vec<f32> {
        self.points.clone()
    }
    /// One value per point: class label, cluster id, or prediction.
    #[wasm_bindgen(getter)]
    pub fn labels(&self) -> Vec<f32> {
        self.labels.clone()
    }
    /// One value per training step — loss, error count, variance explained.
    #[wasm_bindgen(getter)]
    pub fn curve(&self) -> Vec<f32> {
        self.curve.clone()
    }
    /// Per-step snapshots for animation, concatenated end to end.
    #[wasm_bindgen(getter)]
    pub fn frames(&self) -> Vec<f32> {
        self.frames.clone()
    }
    /// Demo-specific scalars, documented at each call site.
    #[wasm_bindgen(getter)]
    pub fn stats(&self) -> Vec<f32> {
        self.stats.clone()
    }
}

// ---------------------------------------------------------------------------
// Sample data — the same generators the native examples use
// ---------------------------------------------------------------------------

/// Generate a toy dataset.
///
/// `kind`: 0 linearly separable, 1 XOR, 2 blobs, 3 polynomial regression.
/// `arg` is the XOR noise level, the blob count, or the polynomial degree.
///
/// Returns `points` (x, y pairs) and `labels`.
#[wasm_bindgen]
pub fn sample_data(kind: u8, n: usize, arg: f32, seed: u32) -> DemoResult {
    let device = Default::default();
    let seed = Some(seed as u64);
    let ds = match kind {
        1 => datasets::make_xor_dataset::<Bk>(n, arg, &device, seed),
        2 => datasets::make_blobs::<Bk>(n, arg.max(1.0) as usize, 0.8, &device, seed),
        3 => {
            datasets::make_polynomial_regression::<Bk>(n, arg.max(1.0) as usize, 0.3, &device, seed)
        }
        _ => datasets::make_linearly_separable::<Bk>(n, &device, seed),
    };
    DemoResult {
        points: flat(ds.features.clone()),
        labels: flat(ds.labels.clone().squeeze::<1>()),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// k-Nearest Neighbors
// ---------------------------------------------------------------------------

/// Fit k-NN and evaluate it over a mesh.
///
/// `metric`: 0 Euclidean, 1 Manhattan, 2 Cosine.
/// `weighting`: 0 uniform, 1 inverse distance, 2 exponential.
/// `grid` holds the predicted class at each mesh point.
#[wasm_bindgen]
pub fn knn_boundary(
    xs: Vec<f32>,
    ys: Vec<f32>,
    k: usize,
    metric: u8,
    weighting: u8,
    res: usize,
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
) -> DemoResult {
    let mut model = KNearestNeighbors::<Bk>::new(k.max(1))
        .with_distance_metric(match metric {
            1 => DistanceMetric::Manhattan,
            2 => DistanceMetric::Cosine,
            _ => DistanceMetric::Euclidean,
        })
        .with_weights(match weighting {
            1 => WeightFunction::Distance,
            2 => WeightFunction::Exponential,
            _ => WeightFunction::Uniform,
        });
    model.fit(t2(&xs, 2), t1(&ys));

    DemoResult {
        grid: flat(model.predict(&mesh(res, x0, x1, y0, y1))),
        labels: flat(model.predict(&t2(&xs, 2))),
        ..Default::default()
    }
}

/// Neighbors of a single query point, nearest first.
///
/// `points` holds the neighbor coordinates, `labels` their classes and
/// `curve` their distances — enough to draw the k spokes of the vote.
#[wasm_bindgen]
pub fn knn_neighbors(xs: Vec<f32>, ys: Vec<f32>, k: usize, qx: f32, qy: f32) -> DemoResult {
    let mut model = KNearestNeighbors::<Bk>::new(k.max(1));
    model.fit(t2(&xs, 2), t1(&ys));

    let neighbors = model.get_neighbors(&t1(&[qx, qy]));
    let mut out = DemoResult::default();
    for (dist, label, idx) in neighbors.into_iter().take(k.max(1)) {
        out.points.push(xs[idx * 2]);
        out.points.push(xs[idx * 2 + 1]);
        out.labels.push(label);
        out.curve.push(dist);
    }
    out
}

// ---------------------------------------------------------------------------
// Perceptron
// ---------------------------------------------------------------------------

/// Train a perceptron, capturing the separating line after each epoch.
///
/// `frames` holds `[w0, w1, bias]` per epoch and `curve` the
/// misclassification *rate* per epoch (which is what `Perceptron::fit`
/// returns), with `stats` the final `[w0, w1, bias, epochs_run]`.
#[wasm_bindgen]
pub fn perceptron_epochs(xs: Vec<f32>, ys: Vec<f32>, lr: f32, epochs: usize) -> DemoResult {
    let x = t2::<Bk>(&xs, 2);
    let y = t1::<Bk>(&ys);
    let mut out = DemoResult::default();

    // The perceptron exposes weights only after fit, so replay the fit with a
    // growing iteration cap to get a snapshot per epoch. Cheap at these sizes.
    for epoch in 1..=epochs.max(1) {
        let mut model = Perceptron::<Bk>::new()
            .with_learning_rate(lr)
            .with_max_iter(epoch)
            .with_shuffle(false);
        let errors = model.fit(&x, &y);

        // `weights()` is [bias, w_x, w_y] when an intercept is fitted, so the
        // normal vector is `coef()` — not the first two weights.
        let w = model.coef().map(flat).unwrap_or_else(|| vec![0.0, 0.0]);
        out.frames.push(*w.first().unwrap_or(&0.0));
        out.frames.push(*w.get(1).unwrap_or(&0.0));
        out.frames.push(model.bias().unwrap_or(0.0));
        out.curve.push(*errors.last().unwrap_or(&0.0));

        if epoch == epochs.max(1) {
            out.labels = flat(model.predict(&x));
            out.stats = vec![
                *w.first().unwrap_or(&0.0),
                *w.get(1).unwrap_or(&0.0),
                model.bias().unwrap_or(0.0),
                errors.len() as f32,
            ];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Decision trees
// ---------------------------------------------------------------------------

/// Fit a classification tree and evaluate it over a mesh.
///
/// `criterion`: 0 Gini, 1 entropy. `stats` is `[depth, nodes, leaves]`.
#[wasm_bindgen]
pub fn decision_tree_boundary(
    xs: Vec<f32>,
    ys: Vec<f32>,
    max_depth: usize,
    min_samples_split: usize,
    criterion: u8,
    res: usize,
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
) -> Result<DemoResult, JsError> {
    let mut tree = DecisionTree::<Bk>::classifier(Default::default())
        .with_max_depth(max_depth.max(1))
        .with_min_samples_split(min_samples_split.max(2))
        .with_criterion(if criterion == 1 {
            SplitCriterion::Entropy
        } else {
            SplitCriterion::Gini
        });

    tree.fit(t2(&xs, 2), t1(&ys))
        .map_err(|e| JsError::new(&e))?;
    let grid = tree
        .predict(mesh(res, x0, x1, y0, y1))
        .map_err(|e| JsError::new(&e))?;
    let labels = tree.predict(t2(&xs, 2)).map_err(|e| JsError::new(&e))?;

    Ok(DemoResult {
        grid: flat(grid),
        labels: flat(labels),
        stats: vec![
            tree.get_depth() as f32,
            tree.get_n_nodes() as f32,
            tree.get_n_leaves() as f32,
        ],
        ..Default::default()
    })
}

// ---------------------------------------------------------------------------
// Linear and logistic regression
// ---------------------------------------------------------------------------

fn regularization(kind: u8, alpha: f32, l1_ratio: f32) -> Regularization {
    match kind {
        1 => Regularization::Ridge { alpha },
        2 => Regularization::Lasso { alpha },
        3 => Regularization::ElasticNet { alpha, l1_ratio },
        _ => Regularization::None,
    }
}

/// Fit 1-D linear regression and sample the fitted line.
///
/// `reg`: 0 none, 1 ridge, 2 lasso, 3 elastic net.
/// `curve` is the prediction at `res` evenly spaced x values across
/// `[x0, x1]`; `stats` is `[slope, intercept, train_mse]`.
#[wasm_bindgen]
pub fn linear_regression_fit(
    xs: Vec<f32>,
    ys: Vec<f32>,
    reg: u8,
    alpha: f32,
    l1_ratio: f32,
    res: usize,
    x0: f32,
    x1: f32,
) -> DemoResult {
    let x = t2::<Bk>(&xs, 1);
    let y = t1::<Bk>(&ys);
    let mut model = LinearRegression::<Bk>::new()
        .with_regularization(regularization(reg, alpha, l1_ratio))
        .with_solver(if reg == 0 || reg == 1 {
            Solver::Normal
        } else {
            Solver::SGD
        });
    model.fit(&x, &y);

    let line_x: Vec<f32> = (0..res.max(2))
        .map(|i| x0 + (x1 - x0) * (i as f32) / ((res.max(2) - 1) as f32))
        .collect();
    let preds = model.predict(&t2(&line_x, 1));
    let residual = model.predict(&x) - y;
    let mse = residual.clone().powf_scalar(2.0).mean().into_scalar();
    let coef = model.coef().map(flat).unwrap_or_default();

    DemoResult {
        curve: flat(preds),
        stats: vec![
            *coef.first().unwrap_or(&0.0),
            model.intercept().unwrap_or(0.0),
            mse,
        ],
        ..Default::default()
    }
}

/// Fit logistic regression and sample `P(y = 1)` over a mesh.
///
/// `reg` follows [`linear_regression_fit`]. `stats` is `[w0, w1, bias]`.
#[wasm_bindgen]
pub fn logistic_regression_grid(
    xs: Vec<f32>,
    ys: Vec<f32>,
    reg: u8,
    alpha: f32,
    lr: f32,
    max_iter: usize,
    res: usize,
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
) -> DemoResult {
    let mut model = LogisticRegression::<Bk>::new()
        .with_regularization(regularization(reg, alpha, 0.5))
        .with_learning_rate(lr)
        .with_max_iter(max_iter.max(1));
    model.fit(&t2(&xs, 2), &t1(&ys));

    let coef = model.coef().map(flat).unwrap_or_default();
    DemoResult {
        grid: flat(model.predict_proba(&mesh(res, x0, x1, y0, y1))),
        labels: flat(model.predict(&t2(&xs, 2))),
        stats: vec![
            *coef.first().unwrap_or(&0.0),
            *coef.get(1).unwrap_or(&0.0),
            model.intercept().unwrap_or(0.0),
        ],
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Kernels and SVM
// ---------------------------------------------------------------------------

fn kernel_type(kind: u8, gamma: f32, degree: u32, coef0: f32) -> KernelType {
    match kind {
        1 => KernelType::RBF { gamma },
        2 => KernelType::Polynomial {
            degree,
            gamma,
            coef0,
        },
        3 => KernelType::Sigmoid { gamma, coef0 },
        _ => KernelType::Linear,
    }
}

/// Train an SVM with SMO and sample its decision function over a mesh.
///
/// Labels must be -1 or +1. `kernel`: 0 linear, 1 RBF, 2 polynomial,
/// 3 sigmoid. `points` holds the support vectors and `stats` is
/// `[n_support_vectors, train_accuracy]`.
#[wasm_bindgen]
pub fn svm_boundary(
    xs: Vec<f32>,
    ys: Vec<f32>,
    kernel: u8,
    gamma: f32,
    degree: u32,
    coef0: f32,
    c: f32,
    max_iter: usize,
    res: usize,
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
) -> Result<DemoResult, JsError> {
    let mut model = SVM::<Bk>::new(
        c,
        kernel_type(kernel, gamma, degree, coef0),
        1e-3,
        max_iter.max(1),
        Default::default(),
    );
    model
        .fit(t2(&xs, 2), t1(&ys))
        .map_err(|e| JsError::new(&e))?;

    let grid = model
        .decision_function(mesh(res, x0, x1, y0, y1))
        .map_err(|e| JsError::new(&e))?;
    let preds = model.predict(t2(&xs, 2)).map_err(|e| JsError::new(&e))?;
    let preds = flat(preds);
    let correct = preds
        .iter()
        .zip(ys.iter())
        .filter(|(p, y)| (*p - *y).abs() < 1e-6)
        .count();

    Ok(DemoResult {
        grid: flat(grid),
        points: model
            .get_support_vectors()
            .map(|sv| flat(sv.clone()))
            .unwrap_or_default(),
        stats: vec![
            model.n_support_vectors() as f32,
            correct as f32 / ys.len().max(1) as f32,
        ],
        labels: preds,
        ..Default::default()
    })
}

/// Compute the full kernel matrix of a dataset.
///
/// `kernel` follows [`svm_boundary`]. `grid` is the `n x n` Gram matrix in
/// row-major order — the thing kernel methods actually see instead of the
/// raw coordinates.
#[wasm_bindgen]
pub fn kernel_matrix(
    xs: Vec<f32>,
    cols: usize,
    kernel: u8,
    gamma: f32,
    degree: u32,
    coef0: f32,
) -> DemoResult {
    let x = t2::<Bk>(&xs, cols.max(1));
    let grid = match kernel {
        1 => RbfKernel::new(gamma).kernel_matrix(&x, &x),
        2 => PolynomialKernel::new(degree, gamma, coef0).kernel_matrix(&x, &x),
        3 => SigmoidKernel::new(gamma, coef0).kernel_matrix(&x, &x),
        _ => LinearKernel.kernel_matrix(&x, &x),
    };
    DemoResult {
        grid: flat(grid),
        stats: vec![x.dims()[0] as f32],
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// k-Means
// ---------------------------------------------------------------------------

/// Run k-means, capturing the centroids after each iteration.
///
/// `init`: 0 random, 1 k-means++. `frames` holds `k` centroid pairs per
/// iteration, `curve` the inertia per iteration, `labels` the final
/// assignments, `grid` the Voronoi cell id at each mesh point and `stats`
/// `[iterations, final_inertia]`.
#[wasm_bindgen]
pub fn kmeans_steps(
    xs: Vec<f32>,
    k: usize,
    max_iter: usize,
    init: u8,
    seed: u32,
    res: usize,
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
) -> Result<DemoResult, JsError> {
    let x = t2::<Bk>(&xs, 2);
    let mut out = DemoResult::default();

    // KMeans reports only its final state, so re-run it with a growing
    // iteration cap. Deterministic because the seed is fixed and n_init = 1.
    for iter in 1..=max_iter.max(1) {
        let config = KMeansConfig::new(k.max(1))
            .with_max_iterations(iter)
            .with_n_init(1)
            .with_random_seed(seed as u64)
            .with_init_method(if init == 1 {
                InitMethod::KMeansPlusPlus
            } else {
                InitMethod::Random
            });
        let mut model = KMeans::<Bk>::new(config, Default::default());
        model.fit(&x).map_err(|e| JsError::new(&e))?;

        if let Some(centroids) = model.centroids() {
            out.frames.extend(flat(centroids.clone()));
        }
        out.curve.push(model.inertia().unwrap_or(f32::NAN));

        if iter == max_iter.max(1) {
            out.labels = model.labels().map(|l| flat(l.clone())).unwrap_or_default();
            out.points = model
                .centroids()
                .map(|c| flat(c.clone()))
                .unwrap_or_default();
            out.grid = model
                .predict(&mesh(res, x0, x1, y0, y1))
                .map(flat)
                .unwrap_or_default();
            out.stats = vec![
                model.n_iterations() as f32,
                model.inertia().unwrap_or(f32::NAN),
            ];
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// PCA
// ---------------------------------------------------------------------------

/// Fit PCA and project the data.
///
/// `points` holds the principal component vectors (row-major, one row per
/// component), `frames` the projected coordinates, `curve` the explained
/// variance ratio per component and `stats` `[total_variance, mean_x, mean_y]`.
#[wasm_bindgen]
pub fn pca_fit(
    xs: Vec<f32>,
    cols: usize,
    n_components: usize,
    scale: bool,
) -> Result<DemoResult, JsError> {
    let x = t2::<Bk>(&xs, cols.max(1));
    let mut model = PCA::<Bk>::new(Some(n_components.max(1)), true, scale);
    let projected = model.fit_transform(&x).map_err(|e| JsError::new(&e))?;
    let mean = model.mean().map(|m| flat(m.clone())).unwrap_or_default();

    Ok(DemoResult {
        points: model
            .components()
            .map(|c| flat(c.clone()))
            .unwrap_or_default(),
        frames: flat(projected),
        curve: model
            .explained_variance_ratio()
            .map(|v| flat(v.clone()))
            .unwrap_or_default(),
        stats: vec![
            model.total_variance().unwrap_or(0.0),
            *mean.first().unwrap_or(&0.0),
            *mean.get(1).unwrap_or(&0.0),
        ],
        ..Default::default()
    })
}

// ---------------------------------------------------------------------------
// Neural network (backprop, autodiff backend)
// ---------------------------------------------------------------------------

/// Train an MLP by backprop and sample its decision surface.
///
/// `hidden` is the hidden layer widths. `curve` is the cross-entropy loss
/// per epoch, `grid` the predicted class at each mesh point and `stats`
/// `[final_loss, train_accuracy, parameter_count]`.
#[wasm_bindgen]
pub fn mlp_train(
    xs: Vec<f32>,
    ys: Vec<f32>,
    hidden: Vec<u32>,
    n_classes: usize,
    epochs: usize,
    lr: f64,
    res: usize,
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
) -> Result<DemoResult, JsError> {
    let hidden: Vec<usize> = hidden.iter().map(|&h| h as usize).collect();
    let x = t2::<Ad>(&xs, 2);
    let y = t1::<Ad>(&ys);

    let mut model = NeuralNetClassifier::<Ad>::new(2, hidden, n_classes.max(2), Default::default());
    let history = model
        .fit_backprop(&x, &y, epochs.max(1), lr)
        .map_err(|e| JsError::new(&e))?;

    let grid = model
        .predict(&mesh::<Ad>(res, x0, x1, y0, y1))
        .map_err(|e| JsError::new(&e))?;
    let preds = flat(model.predict(&x).map_err(|e| JsError::new(&e))?);
    let correct = preds
        .iter()
        .zip(ys.iter())
        .filter(|(p, y)| (*p - *y).abs() < 0.5)
        .count();

    Ok(DemoResult {
        grid: flat(grid),
        labels: preds,
        stats: vec![
            *history.last().unwrap_or(&f32::NAN),
            correct as f32 / ys.len().max(1) as f32,
        ],
        curve: history,
        ..Default::default()
    })
}

// ---------------------------------------------------------------------------
// Autoencoder (stateful — the page trains once, then probes the model)
// ---------------------------------------------------------------------------

/// A trained autoencoder the page can keep around and query.
#[wasm_bindgen]
pub struct AutoencoderDemo {
    model: Autoencoder<Ad>,
    input_dim: usize,
    history: Vec<f32>,
}

#[wasm_bindgen]
impl AutoencoderDemo {
    /// Train an autoencoder on `xs` (row-major, `cols` features per row).
    #[wasm_bindgen(constructor)]
    pub fn new(
        xs: Vec<f32>,
        cols: usize,
        hidden: Vec<u32>,
        latent_dim: usize,
        epochs: usize,
        lr: f64,
    ) -> AutoencoderDemo {
        let input_dim = cols.max(1);
        let config = AutoencoderConfig {
            input_dim,
            hidden_dims: hidden.iter().map(|&h| h as usize).collect(),
            latent_dim: latent_dim.max(1),
            activation: ActivationType::Tanh,
            dropout_rate: 0.0,
            use_batch_norm: false,
            tied_weights: false,
        };
        let (model, history) = Autoencoder::<Ad>::new(config, Default::default())
            .train_reconstruction(t2(&xs, input_dim), ActivationType::Tanh, epochs.max(1), lr);

        AutoencoderDemo {
            model,
            input_dim,
            history,
        }
    }

    /// Reconstruction loss after each training epoch.
    #[wasm_bindgen(getter)]
    pub fn history(&self) -> Vec<f32> {
        self.history.clone()
    }

    /// Latent codes for the given rows, row-major.
    pub fn encode(&self, xs: Vec<f32>) -> Vec<f32> {
        flat(
            self.model
                .encode(t2(&xs, self.input_dim), ActivationType::Tanh),
        )
    }

    /// Reconstructions of the given rows, row-major.
    pub fn reconstruct(&self, xs: Vec<f32>) -> Vec<f32> {
        flat(
            self.model
                .forward(t2(&xs, self.input_dim), ActivationType::Tanh),
        )
    }

    /// Decode arbitrary latent codes — walk the latent space directly.
    pub fn decode(&self, zs: Vec<f32>, latent_dim: usize) -> Vec<f32> {
        flat(
            self.model
                .decode(t2(&zs, latent_dim.max(1)), ActivationType::Tanh),
        )
    }
}

// ---------------------------------------------------------------------------
// AdaBoost
// ---------------------------------------------------------------------------

/// Fit AdaBoost with `n_estimators` stumps and sample its margin.
///
/// Labels must be -1 or +1. Call it repeatedly with a growing
/// `n_estimators` to watch the boundary sharpen stump by stump.
/// `curve` holds the per-feature importances and `stats` is
/// `[stumps_used, train_accuracy]`.
#[wasm_bindgen]
pub fn adaboost_boundary(
    xs: Vec<f32>,
    ys: Vec<f32>,
    n_estimators: usize,
    lr: f32,
    res: usize,
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
) -> Result<DemoResult, JsError> {
    let x = t2::<Bk>(&xs, 2);
    let y = t1::<Bk>(&ys);
    let mut model = AdaBoostClassifier::<Bk>::new(AdaBoostConfig {
        n_estimators: n_estimators.max(1),
        learning_rate: lr,
    });
    model.fit(&x, &y).map_err(|e| JsError::new(&e))?;

    let grid = model
        .decision_function(&mesh(res, x0, x1, y0, y1))
        .map_err(|e| JsError::new(&e))?;
    let preds = flat(model.predict(&x).map_err(|e| JsError::new(&e))?);
    let correct = preds
        .iter()
        .zip(ys.iter())
        .filter(|(p, y)| (*p - *y).abs() < 1e-6)
        .count();

    Ok(DemoResult {
        grid: flat(grid),
        labels: preds,
        curve: model.feature_importances().unwrap_or_default(),
        stats: vec![
            model.n_estimators_used() as f32,
            correct as f32 / ys.len().max(1) as f32,
        ],
        ..Default::default()
    })
}

// ---------------------------------------------------------------------------
// Naive Bayes language classifier
// ---------------------------------------------------------------------------

/// Naive Bayes trained on the built-in German/English word lists.
#[wasm_bindgen]
pub struct NaiveBayesDemo {
    model: NaiveBayesClassifier<Bk>,
    feature_type: FeatureType,
    train_accuracy: f32,
}

#[wasm_bindgen]
impl NaiveBayesDemo {
    /// Train on the synthetic word lists.
    ///
    /// `bigrams` selects letter-pair features (676) instead of single
    /// letters (26). Positive class is German.
    #[wasm_bindgen(constructor)]
    pub fn new(bigrams: bool, smoothing: bool) -> NaiveBayesDemo {
        let feature_type = if bigrams {
            FeatureType::LetterPairs
        } else {
            FeatureType::Letters
        };
        let (german, english) = generate_synthetic_language_data();

        let mut words = german.clone();
        words.extend(english.clone());
        let mut labels = vec![1.0f32; german.len()];
        labels.extend(vec![-1.0f32; english.len()]);

        let device = Default::default();
        let x = extract_features::<Bk>(&words, feature_type, &device);
        let y = t1::<Bk>(&labels);

        let mut model = NaiveBayesClassifier::<Bk>::new(feature_type, device);
        model.train(&x, &y, smoothing);

        let preds = flat(model.predict(&x));
        let correct = preds
            .iter()
            .zip(labels.iter())
            .filter(|(p, y)| (*p - *y).abs() < 1e-6)
            .count();

        NaiveBayesDemo {
            model,
            feature_type,
            train_accuracy: correct as f32 / labels.len().max(1) as f32,
        }
    }

    /// Accuracy on the training words.
    #[wasm_bindgen(getter)]
    pub fn train_accuracy(&self) -> f32 {
        self.train_accuracy
    }

    /// Score one word.
    ///
    /// `stats` is `[log_probability_ratio, predicted_class]` where +1 is
    /// German; `curve` holds the log-ratio of every letter in the word, so
    /// the page can show which letters carried the decision.
    pub fn score(&self, word: String) -> DemoResult {
        let device = Default::default();
        let x = extract_features::<Bk>(&[word.clone()], self.feature_type, &device);
        let ratio = flat(self.model.log_probability_ratio(&x));
        let pred = flat(self.model.predict(&x));

        // Per-letter contribution: score each letter on its own.
        let letters: Vec<String> = word
            .to_lowercase()
            .chars()
            .filter(|c| c.is_ascii_lowercase())
            .map(|c| c.to_string())
            .collect();
        let curve = if letters.is_empty() {
            Vec::new()
        } else {
            let per_letter = extract_features::<Bk>(&letters, self.feature_type, &device);
            flat(self.model.log_probability_ratio(&per_letter))
        };

        DemoResult {
            curve,
            stats: vec![
                *ratio.first().unwrap_or(&0.0),
                *pred.first().unwrap_or(&0.0),
            ],
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Optimizers
// ---------------------------------------------------------------------------

/// Descend an anisotropic quadratic bowl with one of the hand-written
/// optimizers.
///
/// The loss is `0.5 * (curvature_x * x^2 + curvature_y * y^2)`, whose
/// gradient is exact — so the path shows the optimizer's behaviour and
/// nothing else. `kind`: 0 SGD, 1 Adam, 2 AdaGrad.
/// `frames` is the `(x, y)` position after each step and `curve` the loss.
#[wasm_bindgen]
pub fn optimizer_path(
    kind: u8,
    x_start: f32,
    y_start: f32,
    lr: f32,
    steps: usize,
    curvature_x: f32,
    curvature_y: f32,
) -> DemoResult {
    let mut optimizer: Box<dyn ManualOptimizer<Bk>> = match kind {
        1 => Box::new(Adam::new(lr)),
        2 => Box::new(AdaGrad::new(lr)),
        _ => Box::new(SGD::new(lr)),
    };

    let mut params = t2::<Bk>(&[x_start, y_start], 2);
    let mut out = DemoResult::default();

    for _ in 0..steps.max(1) {
        let p = flat(params.clone());
        let (x, y) = (p[0], p[1]);
        out.frames.push(x);
        out.frames.push(y);
        out.curve
            .push(0.5 * (curvature_x * x * x + curvature_y * y * y));

        let grads = t2::<Bk>(&[curvature_x * x, curvature_y * y], 2);
        optimizer.step(&mut params, &grads);
    }

    let p = flat(params);
    out.stats = vec![p[0], p[1], *out.curve.last().unwrap_or(&f32::NAN)];
    out
}

// ---------------------------------------------------------------------------
// Online learning
// ---------------------------------------------------------------------------

/// Stream samples through an online learner one at a time.
///
/// `algo`: 0 online perceptron, 1 passive-aggressive, 2 online SGD (hinge).
/// Labels must be -1 or +1. `curve` is the cumulative mistake count after
/// each sample, `frames` the `[w0, w1, bias]` boundary after each sample and
/// `stats` `[total_mistakes, updates, final_error_rate]`.
#[wasm_bindgen]
pub fn online_stream(
    xs: Vec<f32>,
    ys: Vec<f32>,
    algo: u8,
    lr: f32,
    c: f32,
    alpha: f32,
) -> Result<DemoResult, JsError> {
    let mut learner: Box<dyn OnlineLearner<Bk>> = match algo {
        1 => Box::new(PassiveAggressive::<Bk>::new(PassiveAggressiveConfig {
            c,
            loss: "hinge".to_string(),
            fit_intercept: true,
            random_seed: Some(0),
        })),
        2 => Box::new(OnlineSGD::<Bk>::new(OnlineSGDConfig {
            learning_rate: lr,
            learning_rate_schedule: "constant".to_string(),
            power_t: 0.5,
            loss: "hinge".to_string(),
            alpha,
            fit_intercept: true,
            random_seed: Some(0),
        })),
        _ => Box::new(OnlinePerceptron::<Bk>::new(OnlinePerceptronConfig {
            learning_rate: lr,
            fit_intercept: true,
            random_seed: Some(0),
        })),
    };

    let mut out = DemoResult::default();
    let mut mistakes = 0usize;
    let mut updates = 0usize;

    for (i, &label) in ys.iter().enumerate() {
        let sample = t1::<Bk>(&xs[i * 2..i * 2 + 2]);

        // Predict before learning — that is what makes the count a regret.
        // The raw score, not the thresholded class: a correct negative
        // prediction and a zero score are not the same thing.
        if learner.is_initialized() {
            let score = learner
                .decision_score(&sample)
                .map_err(|e| JsError::new(&e))?;
            if score * label <= 0.0 {
                mistakes += 1;
            }
        } else {
            mistakes += 1;
        }

        learner
            .partial_fit(&sample, label)
            .map_err(|e| JsError::new(&e))?;
        updates += 1;

        let w = learner
            .get_weights()
            .map(|w| flat(w.clone()))
            .unwrap_or_default();
        out.frames.push(*w.first().unwrap_or(&0.0));
        out.frames.push(*w.get(1).unwrap_or(&0.0));
        out.frames.push(learner.bias());
        out.curve.push(mistakes as f32);
    }

    out.stats = vec![
        mistakes as f32,
        updates as f32,
        mistakes as f32 / ys.len().max(1) as f32,
    ];
    Ok(out)
}

// ---------------------------------------------------------------------------
// Transformers
// ---------------------------------------------------------------------------

/// Sinusoidal positional encodings.
///
/// `grid` is the `seq_len x d_model` matrix, row-major — the encoding a
/// transformer adds to its token embeddings, with no training involved.
#[wasm_bindgen]
pub fn positional_encoding(seq_len: usize, d_model: usize) -> DemoResult {
    let device = Default::default();
    let config = PositionEncodingConfig {
        d_model: d_model.max(2),
        max_len: seq_len.max(1),
        dropout: 0.0,
    };
    let encoder = PositionEncoding::<Bk>::new(&config, &device);
    let zeros = Tensor::<Bk, 3>::zeros([1, seq_len.max(1), d_model.max(2)], &device);
    let encoded = encoder.forward(zeros).squeeze::<2>();

    DemoResult {
        grid: flat(encoded),
        stats: vec![seq_len.max(1) as f32, d_model.max(2) as f32],
        ..Default::default()
    }
}

/// Scaled dot-product attention over the given token vectors.
///
/// `softmax(Q K^T / sqrt(d))` with `Q = K = embeddings`, i.e. the attention
/// mechanism itself with no learned projections — move a token and watch the
/// weights move. `grid` is the `n x n` attention matrix, row-major.
#[wasm_bindgen]
pub fn attention_weights(embeddings: Vec<f32>, d_model: usize) -> DemoResult {
    let d = d_model.max(1);
    let q = t2::<Bk>(&embeddings, d);
    let n = q.dims()[0];
    let scores = q
        .clone()
        .matmul(q.transpose())
        .div_scalar((d as f32).sqrt());
    let weights = burn::tensor::activation::softmax(scores, 1);

    DemoResult {
        grid: flat(weights),
        stats: vec![n as f32, d as f32],
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Images: convolution and CNNs
// ---------------------------------------------------------------------------

/// Convolve a grayscale image with an arbitrary kernel.
///
/// This is Burn's functional `conv2d` with weights supplied from the page
/// rather than learned — the operation a convolution layer performs, with the
/// filter under your control. `grid` is the response, same size as the input
/// (zero-padded), row-major.
#[wasm_bindgen]
pub fn apply_kernel(
    pixels: Vec<f32>,
    width: usize,
    height: usize,
    kernel: Vec<f32>,
    kernel_size: usize,
) -> Result<DemoResult, JsError> {
    if width * height != pixels.len() {
        return Err(JsError::new("pixel count does not match width * height"));
    }
    if kernel_size * kernel_size != kernel.len() {
        return Err(JsError::new("kernel is not kernel_size squared"));
    }

    let device = Default::default();
    let image = Tensor::<Bk, 1>::from_data(TensorData::new(pixels, [width * height]), &device)
        .reshape([1, 1, height, width]);
    let weight = Tensor::<Bk, 1>::from_data(
        TensorData::new(kernel, [kernel_size * kernel_size]),
        &device,
    )
    .reshape([1, 1, kernel_size, kernel_size]);

    let padding = kernel_size / 2;
    let response = burn::tensor::module::conv2d(
        image,
        weight,
        None,
        burn::tensor::ops::ConvOptions::new([1, 1], [padding, padding], [1, 1], 1),
    );

    let dims = response.dims();
    Ok(DemoResult {
        grid: flat(response),
        stats: vec![dims[2] as f32, dims[3] as f32],
        ..Default::default()
    })
}

/// The built-in shape dataset: discs, squares, rings and triangles.
///
/// `frames` holds the flat pixel rows (`n * size * size`, values in `[0, 1]`),
/// `labels` the class index of each image, and `stats` `[n_images, size]`.
#[wasm_bindgen]
pub fn shape_images(n_per_class: usize, size: usize, seed: u32) -> DemoResult {
    let device = Default::default();
    let data = datasets::make_shape_images::<Bk>(
        n_per_class.max(1),
        size.max(4),
        &device,
        Some(seed as u64),
    );

    DemoResult {
        frames: flat(data.features.clone()),
        labels: flat(data.labels.clone().squeeze::<1>()),
        stats: vec![data.features.dims()[0] as f32, size.max(4) as f32],
        ..Default::default()
    }
}

/// A CNN over flat grayscale image rows, trained in slices so the page can
/// stay responsive between them.
#[wasm_bindgen]
pub struct CnnDemo {
    model: Option<Cnn<Ad>>,
    history: Vec<f32>,
    train_images: Tensor<Ad, 2>,
    train_labels: Vec<f32>,
    image_size: usize,
    n_classes: usize,
}

#[wasm_bindgen]
impl CnnDemo {
    /// Build an untrained classifier over flat grayscale image rows.
    ///
    /// `conv_channels` is the output channel count of each conv block; every
    /// block halves the image, so `image_size` must divide by
    /// `2^conv_channels.length`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        images: Vec<f32>,
        image_size: usize,
        labels: Vec<f32>,
        n_classes: usize,
        conv_channels: Vec<u32>,
        kernel_size: usize,
    ) -> Result<CnnDemo, JsError> {
        let image_size = image_size.max(4);
        let pixels = image_size * image_size;
        if images.len() != labels.len() * pixels {
            return Err(JsError::new("image data does not match the label count"));
        }

        let config = CnnConfig {
            image_size,
            in_channels: 1,
            conv_channels: conv_channels.iter().map(|&c| c as usize).collect(),
            kernel_size: kernel_size.max(1) | 1,
            n_classes: n_classes.max(2),
        };
        if image_size % (1 << config.conv_channels.len()) != 0 {
            return Err(JsError::new(
                "image size must be divisible by 2^(number of conv blocks)",
            ));
        }

        Ok(CnnDemo {
            model: Some(Cnn::<Ad>::new(&config, &Default::default())),
            history: Vec::new(),
            train_images: t2::<Ad>(&images, pixels),
            train_labels: labels,
            image_size,
            n_classes: config.n_classes,
        })
    }

    /// Train for `epochs` more epochs, appending to the loss history.
    pub fn train(&mut self, epochs: usize, lr: f64, batch_size: usize) {
        let Some(model) = self.model.take() else {
            return;
        };
        let (model, mut losses) = model.train_classifier(
            self.train_images.clone(),
            t1::<Ad>(&self.train_labels),
            epochs.max(1),
            lr,
            batch_size.max(1),
        );
        self.model = Some(model);
        self.history.append(&mut losses);
    }

    /// Epochs trained so far.
    #[wasm_bindgen(getter)]
    pub fn epochs_trained(&self) -> usize {
        self.history.len()
    }

    /// Cross-entropy loss averaged over each training epoch.
    #[wasm_bindgen(getter)]
    pub fn history(&self) -> Vec<f32> {
        self.history.clone()
    }

    /// Accuracy on the images it trained on.
    #[wasm_bindgen(getter)]
    pub fn train_accuracy(&self) -> f32 {
        self.accuracy(flat(self.train_images.clone()), self.train_labels.clone())
    }

    /// Accuracy on a held-out set of flat image rows.
    ///
    /// Training accuracy on a few hundred 16x16 images saturates almost
    /// immediately, so this is the number worth showing.
    pub fn accuracy(&self, images: Vec<f32>, labels: Vec<f32>) -> f32 {
        let Some(model) = self.model.as_ref() else {
            return f32::NAN;
        };
        let pixels = self.image_size * self.image_size;
        if labels.is_empty() || images.len() != labels.len() * pixels {
            return f32::NAN;
        }
        let logits = model.forward(model.as_images(t2::<Ad>(&images, pixels)));
        let predictions = flat(logits.argmax(1).squeeze::<1>().float());
        let correct = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(p, y)| (*p - *y).abs() < 0.5)
            .count();
        correct as f32 / labels.len() as f32
    }

    /// First-layer filters, concatenated row-major: `count` kernels of
    /// `kernel_size^2` values each.
    pub fn filters(&self) -> Vec<f32> {
        self.model
            .as_ref()
            .and_then(|model| model.filters())
            .map(flat)
            .unwrap_or_default()
    }

    /// Class probabilities for one image, as a softmax over the logits.
    pub fn predict(&self, image: Vec<f32>) -> Vec<f32> {
        let pixels = self.image_size * self.image_size;
        let Some(model) = self.model.as_ref() else {
            return vec![f32::NAN; self.n_classes];
        };
        if image.len() != pixels {
            return vec![f32::NAN; self.n_classes];
        }
        let logits = model.forward(model.as_images(t2::<Ad>(&image, pixels)));
        flat(burn::tensor::activation::softmax(logits, 1))
    }

    /// Feature maps from the first conv block for one image, concatenated
    /// row-major: one `size x size` map per filter.
    pub fn feature_maps(&self, image: Vec<f32>) -> Vec<f32> {
        let pixels = self.image_size * self.image_size;
        let Some(model) = self.model.as_ref() else {
            return Vec::new();
        };
        if image.len() != pixels {
            return Vec::new();
        }
        model
            .feature_maps(model.as_images(t2::<Ad>(&image, pixels)))
            .into_iter()
            .next()
            .map(flat)
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Text: a transformer classifier over characters
// ---------------------------------------------------------------------------

/// A character-level transformer over the built-in word lists or a generated
/// task, trained in slices so the page can stay responsive.
///
/// Construction builds an untrained model and holds the data; [`Self::train`]
/// advances it a few epochs at a time. Long training runs block the browser's
/// main thread for as long as they run, so handing the caller control of the
/// slice size is what lets the page paint between them.
#[wasm_bindgen]
pub struct TransformerTextDemo {
    model: Option<TransformerClassifier<Ad>>,
    history: Vec<f32>,
    train_ids: Tensor<Ad, 2>,
    train_labels: Vec<f32>,
    test_ids: Tensor<Ad, 2>,
    test_labels: Vec<f32>,
    seq_len: usize,
    n_heads: usize,
    n_layers: usize,
}

#[wasm_bindgen]
impl TransformerTextDemo {
    /// Build an untrained classifier and prepare its data.
    ///
    /// `task` 0 is German versus English on the built-in word lists (class 1
    /// is German); task 1 is "does this string contain a k" on generated
    /// strings (class 1 contains one). Every fifth example is held out.
    /// `d_model` must divide by `n_heads`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        task: u8,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        seq_len: usize,
    ) -> Result<TransformerTextDemo, JsError> {
        let n_heads = n_heads.max(1);
        let d_model = d_model.max(n_heads);
        if d_model % n_heads != 0 {
            return Err(JsError::new(&format!(
                "d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )));
        }
        let seq_len = seq_len.clamp(2, 32);

        let (words, labels) = if task == 1 {
            make_letter_search_words(700, 'k', 17)
        } else {
            let (german, english) = generate_synthetic_language_data();
            let mut words = Vec::new();
            let mut labels = Vec::new();
            for (list, label) in [(&german, 1.0f32), (&english, 0.0f32)] {
                for word in list.iter() {
                    words.push(word.clone());
                    labels.push(label);
                }
            }
            (words, labels)
        };

        // Deterministic interleave, then every fifth word is held out. The
        // lists arrive grouped by language, so a contiguous split would put
        // one whole class in the test set.
        let mut order: Vec<usize> = (0..words.len()).collect();
        order.sort_by_key(|&i| (i * 7919) % words.len().max(1));

        let mut train_words = Vec::new();
        let mut train_labels = Vec::new();
        let mut test_words = Vec::new();
        let mut test_labels = Vec::new();
        for (position, &i) in order.iter().enumerate() {
            if position % 5 == 0 {
                test_words.push(words[i].clone());
                test_labels.push(labels[i]);
            } else {
                train_words.push(words[i].clone());
                train_labels.push(labels[i]);
            }
        }

        let device = Default::default();
        let config = TransformerEncoderConfig {
            d_model,
            n_heads,
            n_layers: n_layers.clamp(1, 4),
            d_ff: d_model * 2,
            max_len: seq_len,
            vocab_size: CharTokenizer::VOCAB,
            dropout: 0.0,
        };

        Ok(TransformerTextDemo {
            model: Some(TransformerClassifier::<Ad>::new(&config, 2, &device)),
            history: Vec::new(),
            train_ids: CharTokenizer::encode_batch::<Ad>(&train_words, seq_len, &device),
            train_labels,
            test_ids: CharTokenizer::encode_batch::<Ad>(&test_words, seq_len, &device),
            test_labels,
            seq_len,
            n_heads,
            n_layers: config.n_layers,
        })
    }

    /// Train for `epochs` more epochs, appending to the loss history.
    ///
    /// Call it repeatedly with a small `epochs` to keep the page interactive.
    pub fn train(&mut self, epochs: usize, lr: f64, batch_size: usize) {
        let Some(model) = self.model.take() else {
            return;
        };
        let (model, mut losses) = model.train_classifier(
            self.train_ids.clone(),
            t1::<Ad>(&self.train_labels),
            epochs.max(1),
            lr,
            batch_size.max(1),
        );
        self.model = Some(model);
        self.history.append(&mut losses);
    }

    /// Epochs trained so far.
    #[wasm_bindgen(getter)]
    pub fn epochs_trained(&self) -> usize {
        self.history.len()
    }

    /// Cross-entropy loss averaged over each training epoch.
    #[wasm_bindgen(getter)]
    pub fn history(&self) -> Vec<f32> {
        self.history.clone()
    }

    fn score(&self, ids: &Tensor<Ad, 2>, labels: &[f32]) -> f32 {
        let Some(model) = self.model.as_ref() else {
            return f32::NAN;
        };
        if labels.is_empty() {
            return f32::NAN;
        }
        let predictions = flat(
            model
                .forward(ids.clone(), None)
                .argmax(1)
                .squeeze::<1>()
                .float(),
        );
        let correct = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(p, y)| (*p - *y).abs() < 0.5)
            .count();
        correct as f32 / labels.len() as f32
    }

    /// Accuracy on the words it trained on.
    #[wasm_bindgen(getter)]
    pub fn train_accuracy(&self) -> f32 {
        self.score(&self.train_ids, &self.train_labels)
    }

    /// Accuracy on the held-out fifth.
    #[wasm_bindgen(getter)]
    pub fn test_accuracy(&self) -> f32 {
        self.score(&self.test_ids, &self.test_labels)
    }

    /// Sequence length the model was built for.
    #[wasm_bindgen(getter)]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Number of attention heads per layer.
    #[wasm_bindgen(getter)]
    pub fn n_heads(&self) -> usize {
        self.n_heads
    }

    /// Number of encoder layers.
    #[wasm_bindgen(getter)]
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// The tokens one word turns into, `[CLS]` first, padding as `·`.
    pub fn tokens(&self, word: String) -> Vec<String> {
        CharTokenizer::tokens(&word, self.seq_len)
    }

    /// Class probabilities for one word: `[P(class 0), P(class 1)]`.
    pub fn classify(&self, word: String) -> Vec<f32> {
        let Some(model) = self.model.as_ref() else {
            return vec![f32::NAN; 2];
        };
        let ids = CharTokenizer::encode_batch::<Ad>(
            std::slice::from_ref(&word),
            self.seq_len,
            &Default::default(),
        );
        flat(burn::tensor::activation::softmax(
            model.forward(ids, None),
            1,
        ))
    }

    /// Attention weights for one word, concatenated row-major:
    /// `n_layers * n_heads` matrices of `seq_len * seq_len`.
    pub fn attention(&self, word: String) -> Vec<f32> {
        let Some(model) = self.model.as_ref() else {
            return Vec::new();
        };
        let ids = CharTokenizer::encode_batch::<Ad>(
            std::slice::from_ref(&word),
            self.seq_len,
            &Default::default(),
        );
        model
            .encoder()
            .attention_weights(ids, None)
            .into_iter()
            .flat_map(flat)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Text generation
// ---------------------------------------------------------------------------

/// A character-level language model trained on one of the built-in word
/// lists, which then invents new words in the same style.
#[wasm_bindgen]
pub struct TransformerGenDemo {
    model: Option<TransformerLanguageModel<Ad>>,
    history: Vec<f32>,
    inputs: Tensor<Ad, 2>,
    targets: Tensor<Ad, 2>,
    corpus: Vec<String>,
    seq_len: usize,
}

#[wasm_bindgen]
impl TransformerGenDemo {
    /// Build an untrained model over `corpus` 0 (German), 1 (English) or
    /// 2 (a generated language with vowel harmony).
    #[wasm_bindgen(constructor)]
    pub fn new(
        corpus: u8,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        seq_len: usize,
    ) -> Result<TransformerGenDemo, JsError> {
        let n_heads = n_heads.max(1);
        let d_model = d_model.max(n_heads);
        if d_model % n_heads != 0 {
            return Err(JsError::new(&format!(
                "d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )));
        }
        let seq_len = seq_len.clamp(4, 24);

        let words = match corpus {
            1 => generate_synthetic_language_data().1,
            2 => make_harmony_words(900, 41),
            _ => generate_synthetic_language_data().0,
        };

        let device = Default::default();
        let mut input_data = Vec::with_capacity(words.len() * seq_len);
        let mut target_data = Vec::with_capacity(words.len() * seq_len);
        for word in &words {
            input_data.extend(CharTokenizer::encode(word, seq_len));
            target_data.extend(CharTokenizer::encode_target(word, seq_len));
        }

        let shape = [words.len(), seq_len];
        let config = TransformerEncoderConfig {
            d_model,
            n_heads,
            n_layers: n_layers.clamp(1, 4),
            d_ff: d_model * 2,
            max_len: seq_len,
            vocab_size: CharTokenizer::VOCAB,
            dropout: 0.0,
        };

        Ok(TransformerGenDemo {
            model: Some(TransformerLanguageModel::<Ad>::new(&config, &device)),
            history: Vec::new(),
            inputs: Tensor::from_data(TensorData::new(input_data, shape), &device),
            targets: Tensor::from_data(TensorData::new(target_data, shape), &device),
            corpus: words,
            seq_len,
        })
    }

    /// Train for `epochs` more epochs, appending to the loss history.
    pub fn train(&mut self, epochs: usize, lr: f64, batch_size: usize) {
        let Some(model) = self.model.take() else {
            return;
        };
        let (model, mut losses) = model.train_lm(
            self.inputs.clone(),
            self.targets.clone(),
            epochs.max(1),
            lr,
            batch_size.max(1),
        );
        self.model = Some(model);
        self.history.append(&mut losses);
    }

    /// Epochs trained so far.
    #[wasm_bindgen(getter)]
    pub fn epochs_trained(&self) -> usize {
        self.history.len()
    }

    /// Cross-entropy loss averaged over each training epoch.
    #[wasm_bindgen(getter)]
    pub fn history(&self) -> Vec<f32> {
        self.history.clone()
    }

    /// Number of words the model was trained on.
    #[wasm_bindgen(getter)]
    pub fn corpus_size(&self) -> usize {
        self.corpus.len()
    }

    /// Next-character probabilities given a prefix, one per vocabulary entry.
    ///
    /// Index 0 is the start symbol, 1 is end-of-word, and 2..28 are a-z.
    pub fn next_probabilities(&self, prefix: String) -> Vec<f32> {
        let Some(model) = self.model.as_ref() else {
            return Vec::new();
        };
        let ids = CharTokenizer::encode(&prefix, self.seq_len);
        // The prefix occupies [CLS] plus its own characters; read the logits
        // at the last of those, which is the position predicting what's next.
        let position = prefix
            .chars()
            .filter(|c| c.is_ascii_lowercase())
            .count()
            .min(self.seq_len - 1);

        let logits = model.forward(t2::<Ad>(&ids, self.seq_len));
        let row = logits
            .slice([0..1, position..position + 1])
            .reshape([CharTokenizer::VOCAB]);
        flat(burn::tensor::activation::softmax(row, 0))
    }

    /// Sample a word, optionally continuing `prompt`.
    ///
    /// `temperature` flattens (>1) or sharpens (<1) the distribution before
    /// sampling. Generation stops at the end-of-word symbol.
    pub fn generate(&self, prompt: String, temperature: f32, seed: u32) -> String {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed as u64);
        let temperature = temperature.max(0.01);

        let mut word: String = prompt
            .to_lowercase()
            .chars()
            .filter(|c| c.is_ascii_lowercase())
            .collect();

        while word.chars().count() < self.seq_len - 1 {
            let probabilities = self.next_probabilities(word.clone());
            if probabilities.len() < CharTokenizer::VOCAB {
                break;
            }

            // Reweight by temperature in log space, skipping the start symbol
            // which is never a legal continuation. Computing p^(1/T) directly
            // underflows: at T = 0.01 the exponent is 100, every weight
            // becomes zero, and sampling degenerates to "emit end-of-word".
            // Subtracting the max before exponentiating keeps low temperatures
            // sharpening toward greedy, which is the correct limit.
            let mut logits = vec![f32::NEG_INFINITY; CharTokenizer::VOCAB];
            let mut max_logit = f32::NEG_INFINITY;
            for id in 1..CharTokenizer::VOCAB {
                let logit = probabilities[id].max(1e-12).ln() / temperature;
                logits[id] = logit;
                max_logit = max_logit.max(logit);
            }

            let mut weights = vec![0.0f32; CharTokenizer::VOCAB];
            let mut total = 0.0;
            for id in 1..CharTokenizer::VOCAB {
                let w = (logits[id] - max_logit).exp();
                weights[id] = w;
                total += w;
            }

            let mut target = rng.gen_range(0.0..total.max(f32::MIN_POSITIVE));
            let mut chosen = CharTokenizer::PAD as usize;
            for (id, &w) in weights.iter().enumerate() {
                target -= w;
                if target <= 0.0 {
                    chosen = id;
                    break;
                }
            }

            if chosen <= CharTokenizer::PAD as usize {
                break;
            }
            word.push((b'a' + (chosen - 2) as u8) as char);
        }

        word
    }

    /// Whether `word` obeys the vowel-harmony rule of corpus 2.
    ///
    /// Meaningless for the real-language corpora, which have no such rule.
    pub fn obeys_harmony(&self, word: String) -> bool {
        obeys_vowel_harmony(&word)
    }

    /// Whether a generated word appears verbatim in the training corpus.
    ///
    /// A model this small can memorize, and a demo that quietly replays its
    /// training data would be a lie.
    pub fn is_memorized(&self, word: String) -> bool {
        self.corpus.iter().any(|w| *w == word)
    }
}
