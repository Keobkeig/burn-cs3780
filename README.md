# burn-cs3780

Machine learning algorithms from Cornell's CS 3780, implemented from scratch on
the [Burn](https://burn.dev) tensor framework — and compiled to WebAssembly, so
they run in a browser as well as natively.

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)
[![Burn](https://img.shields.io/badge/burn-0.20-red.svg)](https://burn.dev)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Twenty-one of these run live at
[/projects/burn-cs3780](https://github.com/Keobkeig/richie-portfolio) — training
in the page, with no server involved.

## What's here

Everything below is implemented and covered by a unit test or a browser demo.
If something isn't in this list, it isn't in the crate.

### Supervised

| Model | Module | Notes |
|---|---|---|
| k-nearest neighbors | `models::knn` | Classification and regression; Euclidean, Manhattan, cosine; uniform, inverse-distance, exponential weighting |
| Decision trees | `models::decision_tree` | Classification and regression; Gini, entropy, MSE |
| Linear regression | `models::linear_models` | Normal equation via Gauss-Jordan, or SGD/Adam; ridge, lasso, elastic net |
| Logistic regression | `models::linear_models` | Gradient descent with the same regularizers |
| Perceptron | `models::perceptron` | Single and one-vs-rest multi-class |
| Support vector machine | `models::svm` | Sequential Minimal Optimization, four kernels |
| Naive Bayes | `models::naive_bayes` | Text classification over letter or letter-pair features |
| AdaBoost | `models::boosting` | Stumps fitted by exhaustive weighted-error search |
| Gradient boosting | `models::boosting` | Binary logistic loss, regression trees on residuals, optional row subsampling |

### Neural

| Model | Module | Notes |
|---|---|---|
| Multilayer perceptron | `models::neural_networks` | `fit_backprop` trains with Adam on an autodiff backend |
| Convolutional network | `models::cnn` | Conv2d / ReLU / max-pool with a linear head; exposes learned filters and feature maps |
| Autoencoder | `models::autoencoders` | Plus variational, denoising (3 noise types), and sparse (KL penalty on hidden units) |
| Transformer encoder | `models::transformers` | Multi-head attention, sinusoidal positional encoding, embedding table |
| Transformer classifier | `models::transformers` | Reads `[CLS]`; exposes per-layer, per-head attention weights |
| Character language model | `models::transformers` | Causal mask, next-token head, `CharTokenizer` |

### Unsupervised

| Model | Module | Notes |
|---|---|---|
| k-means | `models::clustering` | Random or k-means++ (D² seeding); reproducible under a seed |
| PCA | `models::pca` | Power iteration with deflation on the Gram matrix — Burn has no SVD |
| Kernel ridge regression | `kernels::utils` | Dual-form solve of `(K + λI)α = y` |

### Supporting

- **Kernels** (`kernels`) — linear, polynomial, RBF, sigmoid. Each builds its
  whole Gram matrix in a few tensor ops rather than a pair at a time.
- **Optimizers** (`optimizers`) — SGD (momentum, Nesterov), Adam (AMSGrad),
  AdaGrad. Schedules: step and exponential.
- **Online learning** (`models::online_learning`) — online perceptron,
  passive-aggressive, and online SGD with hinge/log/squared/Huber losses and
  constant, optimal or inverse-scaling rates.
- **Metrics** (`metrics`) — accuracy, precision, recall, F1, confusion matrix;
  MSE, RMSE, MAE, R², explained variance; k-fold index generation.
- **Preprocessing** (`utils`) — standardization, min-max scaling, polynomial
  powers, bias columns, vectorized pairwise distances.
- **Datasets** (`datasets`) — linearly separable, XOR, Gaussian blobs,
  polynomial regression, shape images, and a synthetic language with vowel
  harmony.

### Not here

Named so nobody goes looking: no RNN/LSTM/GRU, no random forests, no t-SNE, no
RMSprop, no cosine annealing, no early stopping. Cross-validation is
`k_fold_indices` only — there is no scoring loop or hyperparameter search on
top of it. `polynomial_features` emits pure powers, not interaction terms.
Multi-class gradient boosting, and multi-class probabilities for both boosting
models, return an error rather than a wrong answer.

## Backends

Where a tensor computes is a type parameter, so every model above is written
once and runs on all of these:

```rust
pub type DefaultBackend = NdArray<f32>;                   // CPU
pub type GpuBackend = Wgpu<f32, i32>;                     // GPU, via wgpu
pub type DefaultAutodiffBackend = Autodiff<NdArray<f32>>; // CPU + gradients
```

Autodiff is a backend wrapping another backend, so `Autodiff<Wgpu>` is gradients
on the GPU and `Autodiff<NdArray>` is gradients in a browser tab — same source.

## Usage

```rust
use burn_cs3780::models::KNearestNeighbors;
use burn_cs3780::{datasets, DefaultBackend};

let device = Default::default();
let data = datasets::make_blobs::<DefaultBackend>(200, 3, 0.8, &device, Some(42));
let (train, test) = data.train_test_split(0.8, Some(42));

let mut knn = KNearestNeighbors::<DefaultBackend>::new(5);
knn.fit(train.features, train.labels.squeeze_dims::<1>(&[1]));

let predictions = knn.predict(&test.features);
```

Training a network needs an autodiff backend, and the compiler enforces it:
`backward()` is not a method that exists on a plain `Backend`.

```rust
use burn_cs3780::models::NeuralNetClassifier;
use burn_cs3780::DefaultAutodiffBackend as B;

let mut net = NeuralNetClassifier::<B>::new(2, vec![16, 16], 2, Default::default());
let losses = net.fit_backprop(&x, &y, 300, 0.05)?;
```

## Building

```bash
cargo build                 # native
cargo test --lib            # unit tests
```

Use `--lib`. Plain `cargo test` also builds `examples/`, and three of those
target an API that was renamed and no longer compile; the library and its tests
are unaffected.

### WebAssembly

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

./build-wasm.sh                 # -> ../richie-portfolio/static/wasm/burn
./build-wasm.sh /path/to/out    # or anywhere
bun test-wasm.mjs               # 61 assertions against the built bundle
```

The browser build drops `wgpu`, `burn-train`, `plotters` and `clap`, keeping
`burn` with `ndarray`, `std` and `autodiff`. Exactly one module is excluded —
`utils::visualization`, which draws with plotters. Output is about 1.2 MB of
`.wasm`, 260 KB brotli-compressed.

`src/wasm.rs` is the binding layer: flat `Float32Array` in, one `DemoResult`
struct out, so every demo shares one interface. It holds no algorithm logic, so
the browser runs the same code paths the native build does.

`make-thumbnail.mjs` renders a project thumbnail from real model output.

## Layout

```
src/
├── datasets.rs      synthetic data generators
├── kernels/         linear, polynomial, RBF, sigmoid, kernel ridge
├── metrics/         classification, regression, k-fold indices
├── models/          every model in the tables above
├── optimizers/      SGD, Adam, AdaGrad, LR schedules
├── utils/           distance, math, preprocessing, plotting (native only)
└── wasm.rs          wasm-bindgen layer (wasm32 only)

examples/            8 of 11 compile; see the note under Building
```

## License

MIT — see [LICENSE](LICENSE).
