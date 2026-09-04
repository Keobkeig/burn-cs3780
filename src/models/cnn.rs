//! Convolutional neural networks for image classification.
//!
//! A convolution is the one architectural idea on the syllabus that is about
//! the *shape* of the input rather than the model: weights are shared across
//! positions, so a filter that detects an edge detects it anywhere in the
//! image. This module provides a small stack of conv blocks with a linear
//! head, plus accessors for the learned filters and the intermediate feature
//! maps — the parts worth looking at.

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::{activation::relu, backend::Backend, Device, Tensor};

/// Configuration for a [`Cnn`].
#[derive(Debug, Clone)]
pub struct CnnConfig {
    /// Side length of the (square) input images.
    pub image_size: usize,
    /// Channels in the input — 1 for grayscale.
    pub in_channels: usize,
    /// Output channels of each convolution block. Each block halves the
    /// spatial dimensions, so `image_size` must be divisible by
    /// `2^conv_channels.len()`.
    pub conv_channels: Vec<usize>,
    /// Side length of every convolution kernel. Must be odd.
    pub kernel_size: usize,
    /// Number of output classes.
    pub n_classes: usize,
}

impl Default for CnnConfig {
    fn default() -> Self {
        Self {
            image_size: 12,
            in_channels: 1,
            conv_channels: vec![8],
            kernel_size: 3,
            n_classes: 4,
        }
    }
}

/// A convolution / ReLU / max-pool stack with a linear classifier head.
#[derive(Module, Debug)]
pub struct Cnn<B: Backend> {
    convs: Vec<Conv2d<B>>,
    pool: MaxPool2d,
    head: Linear<B>,
    image_size: usize,
    in_channels: usize,
}

impl<B: Backend<FloatElem = f32>> Cnn<B> {
    /// Build a new network with randomly initialized filters.
    pub fn new(config: &CnnConfig, device: &Device<B>) -> Self {
        let mut convs = Vec::new();
        let mut channels = config.in_channels;

        for &out_channels in &config.conv_channels {
            convs.push(
                Conv2dConfig::new(
                    [channels, out_channels],
                    [config.kernel_size, config.kernel_size],
                )
                // Same padding keeps the arithmetic simple: only the pooling
                // changes the spatial size, and it always halves it.
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            );
            channels = out_channels;
        }

        let pool = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        let spatial = config.image_size >> config.conv_channels.len();
        let flattened = channels * spatial * spatial;

        Self {
            convs,
            pool,
            head: LinearConfig::new(flattened, config.n_classes).init(device),
            image_size: config.image_size,
            in_channels: config.in_channels,
        }
    }

    /// Class logits for a batch of images shaped `[batch, channels, h, w]`.
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut x = images;
        for conv in &self.convs {
            x = self.pool.forward(relu(conv.forward(x)));
        }
        self.head.forward(x.flatten(1, 3))
    }

    /// Activations after each conv block, before pooling.
    ///
    /// One tensor per block, shaped `[batch, channels, h, w]` — the feature
    /// maps a page can draw.
    pub fn feature_maps(&self, images: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        let mut maps = Vec::with_capacity(self.convs.len());
        let mut x = images;
        for conv in &self.convs {
            let activated = relu(conv.forward(x));
            maps.push(activated.clone());
            x = self.pool.forward(activated);
        }
        maps
    }

    /// First-layer filters, shaped `[out_channels, in_channels, k, k]`.
    ///
    /// These are the only weights in the network with a direct visual
    /// reading: each one is a small image the layer looks for.
    pub fn filters(&self) -> Option<Tensor<B, 4>> {
        self.convs.first().map(|conv| conv.weight.val())
    }

    /// Reshape a flat batch of grayscale pixels into a conv input.
    pub fn as_images(&self, pixels: Tensor<B, 2>) -> Tensor<B, 4> {
        let batch = pixels.dims()[0];
        pixels.reshape([batch, self.in_channels, self.image_size, self.image_size])
    }

    /// Side length of the images this network expects.
    pub fn image_size(&self) -> usize {
        self.image_size
    }
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

impl<B: burn::tensor::backend::AutodiffBackend<FloatElem = f32>> Cnn<B> {
    /// Train the classifier on `images` with integer class labels `y`.
    ///
    /// Mini-batch Adam, batches taken in order. Returns the trained network
    /// and the mean loss of each epoch. Consumes and returns `self` because
    /// that is how Burn's optimizers hand parameters back.
    ///
    /// # Arguments
    /// * `images` - Flat pixel rows, `[n_samples, channels * h * w]`
    /// * `y` - Class indices as floats, `[n_samples]`
    /// * `epochs` - Passes over the data
    /// * `lr` - Adam learning rate
    /// * `batch_size` - Samples per gradient step
    pub fn train_classifier(
        mut self,
        images: Tensor<B, 2>,
        y: Tensor<B, 1>,
        epochs: usize,
        lr: f64,
        batch_size: usize,
    ) -> (Self, Vec<f32>) {
        use burn::nn::loss::CrossEntropyLossConfig;
        use burn::optim::{AdamConfig, GradientsParams, Optimizer};

        let device = y.device();
        let n_samples = images.dims()[0];
        let batch_size = batch_size.clamp(1, n_samples.max(1));
        let targets = y.int();
        let loss_fn = CrossEntropyLossConfig::new().init(&device);
        let mut optimizer = AdamConfig::new().init();
        let mut history = Vec::with_capacity(epochs);

        for _ in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut steps = 0;

            let mut start = 0;
            while start < n_samples {
                let end = (start + batch_size).min(n_samples);
                let batch = self.as_images(images.clone().slice([start..end]));
                let batch_targets = targets.clone().slice([start..end]);

                let loss = loss_fn.forward(self.forward(batch), batch_targets);
                epoch_loss += loss.clone().into_scalar();
                steps += 1;

                let grads = GradientsParams::from_grads(loss.backward(), &self);
                self = optimizer.step(lr, self, grads);
                start = end;
            }

            history.push(if steps > 0 {
                epoch_loss / steps as f32
            } else {
                f32::NAN
            });
        }

        (self, history)
    }
}
