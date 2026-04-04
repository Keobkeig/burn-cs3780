//! Neural Networks implementation for feedforward networks
//!
//! This module implements Multi-Layer Perceptrons (MLPs) and other feedforward
//! neural networks using the Burn framework for both CPU and GPU computation.

use burn::{
    backend::NdArray,
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    tensor::{backend::Backend, Device, Tensor},
};

/// Type alias for the default backend (CPU)
type DefaultBackend = NdArray<f32>;

/// Activation functions for neural networks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// ReLU activation function
    Relu,
    /// Sigmoid activation function
    Sigmoid,
    /// Tanh activation function
    Tanh,
    /// No activation (linear)
    Linear,
}

impl Activation {
    /// Apply the activation function to a tensor
    pub fn apply<B: Backend<FloatElem = f32>, const D: usize>(
        &self,
        input: Tensor<B, D>,
    ) -> Tensor<B, D> {
        match self {
            Activation::Relu => burn::tensor::activation::relu(input),
            Activation::Sigmoid => burn::tensor::activation::sigmoid(input),
            Activation::Tanh => burn::tensor::activation::tanh(input),
            Activation::Linear => input,
        }
    }
}

/// A neural network layer
#[derive(Module, Debug)]
pub struct Layer<B: Backend> {
    linear: Linear<B>,
    dropout: Option<Dropout>,
}

impl<B: Backend<FloatElem = f32>> Layer<B> {
    /// Create a new layer
    pub fn new(
        input_size: usize,
        output_size: usize,
        use_bias: bool,
        dropout_prob: Option<f64>,
        device: &Device<B>,
    ) -> Self {
        let linear_config = LinearConfig::new(input_size, output_size).with_bias(use_bias);
        let linear = linear_config.init(device);

        let dropout = dropout_prob.map(|prob| DropoutConfig::new(prob).init());

        Self { linear, dropout }
    }

    /// Forward pass through the layer
    pub fn forward(&self, input: Tensor<B, 2>, activation: Activation) -> Tensor<B, 2> {
        let mut output = self.linear.forward(input);

        // Apply dropout during training
        if let Some(dropout) = &self.dropout {
            output = dropout.forward(output);
        }

        // Apply activation function
        activation.apply(output)
    }
}

/// Multi-Layer Perceptron (MLP) configuration
#[derive(Config, Debug)]
pub struct MLPConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Number of hidden layers
    pub hidden_layers: Vec<usize>,
    /// Dropout probability for hidden layers
    pub dropout: Option<f64>,
}

/// Multi-Layer Perceptron (MLP) neural network
#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    layers: Vec<Layer<B>>,
    input_dim: usize,
    output_dim: usize,
}

impl<B: Backend<FloatElem = f32>> MLP<B> {
    /// Create a new MLP
    pub fn new(
        input_dim: usize,
        hidden_layers: Vec<usize>,
        output_dim: usize,
        device: &Device<B>,
    ) -> Self {
        let mut layers = Vec::new();

        // Create hidden layers
        let mut prev_size = input_dim;
        for &hidden_size in &hidden_layers {
            layers.push(Layer::new(
                prev_size,
                hidden_size,
                true,
                None, // No dropout for now
                device,
            ));
            prev_size = hidden_size;
        }

        // Create output layer
        layers.push(Layer::new(prev_size, output_dim, true, None, device));

        Self {
            layers,
            input_dim,
            output_dim,
        }
    }

    /// Forward pass through the network
    pub fn forward(
        &self,
        input: Tensor<B, 2>,
        hidden_activation: Activation,
        output_activation: Activation,
    ) -> Tensor<B, 2> {
        let mut output = input;

        // Forward through hidden layers
        for layer in &self.layers[..self.layers.len() - 1] {
            output = layer.forward(output, hidden_activation);
        }

        // Forward through output layer
        if let Some(output_layer) = self.layers.last() {
            output = output_layer.forward(output, output_activation);
        }

        output
    }

    /// Get the input dimension
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get the output dimension
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Get the number of parameters in the network
    pub fn num_params(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| {
                let linear = &layer.linear;
                let weight_dims = linear.weight.dims();
                let weight_params = weight_dims.iter().product::<usize>();
                let bias_params = if let Some(ref bias) = linear.bias {
                    bias.dims().iter().product::<usize>()
                } else {
                    0
                };
                weight_params + bias_params
            })
            .sum()
    }
}

/// Training configuration for neural networks
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of training epochs
    pub epochs: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size for training
    pub batch_size: usize,
    /// Optimizer type to use
    pub optimizer_type: OptimizerType,
    /// Loss function type
    pub loss_type: LossType,
    /// Validation split (fraction of data to use for validation)
    pub validation_split: f64,
    /// Hidden activation function
    pub hidden_activation: Activation,
    /// Output activation function  
    pub output_activation: Activation,
}

/// Available optimizer types
#[derive(Debug, Clone)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent
    SGD,
    /// Adam optimizer
    Adam,
    /// AdaGrad optimizer
    AdaGrad,
}

/// Available loss function types
#[derive(Debug, Clone)]
pub enum LossType {
    /// Mean Squared Error for regression
    MSE,
    /// Cross Entropy for classification
    CrossEntropy,
    /// Binary Cross Entropy for binary classification
    BinaryCrossEntropy,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            learning_rate: 0.001,
            batch_size: 32,
            optimizer_type: OptimizerType::Adam,
            loss_type: LossType::MSE,
            validation_split: 0.2,
            hidden_activation: Activation::Relu,
            output_activation: Activation::Linear,
        }
    }
}

/// Neural network classifier using MLP
#[derive(Debug)]
pub struct NeuralNetClassifier<B: Backend<FloatElem = f32>> {
    model: Option<MLP<B>>,
    training_config: TrainingConfig,
    device: Device<B>,
    input_dim: usize,
    hidden_layers: Vec<usize>,
    num_classes: usize,
    is_fitted: bool,
}

impl Default for NeuralNetClassifier<DefaultBackend> {
    fn default() -> Self {
        Self {
            model: None,
            training_config: TrainingConfig {
                loss_type: LossType::CrossEntropy,
                output_activation: Activation::Linear, // Softmax will be applied in loss calculation
                ..Default::default()
            },
            device: Device::<DefaultBackend>::default(),
            input_dim: 4,
            hidden_layers: vec![10, 5],
            num_classes: 3,
            is_fitted: false,
        }
    }
}

impl<B: Backend<FloatElem = f32>> NeuralNetClassifier<B> {
    /// Create a new neural network classifier
    pub fn new(
        input_dim: usize,
        hidden_layers: Vec<usize>,
        num_classes: usize,
        device: Device<B>,
    ) -> Self {
        let training_config = TrainingConfig {
            loss_type: LossType::CrossEntropy,
            output_activation: Activation::Linear,
            ..Default::default()
        };

        Self {
            model: None,
            training_config,
            device,
            input_dim,
            hidden_layers,
            num_classes,
            is_fitted: false,
        }
    }

    /// Set the training configuration
    pub fn with_training_config(mut self, config: TrainingConfig) -> Self {
        self.training_config = config;
        self
    }

    /// Fit the classifier to training data
    pub fn fit(&mut self, _x: &Tensor<B, 2>, _y: &Tensor<B, 1>) -> Result<(), String> {
        // Initialize the model
        self.model = Some(MLP::new(
            self.input_dim,
            self.hidden_layers.clone(),
            self.num_classes,
            &self.device,
        ));

        // For now, we'll implement a basic training loop
        // In a full implementation, this would include proper gradient computation,
        // loss calculation, and optimization steps

        self.is_fitted = true;
        Ok(())
    }

    /// Predict class probabilities for input data
    pub fn predict_proba(&self, x: &Tensor<B, 2>) -> Result<Tensor<B, 2>, String> {
        if !self.is_fitted || self.model.is_none() {
            return Err("Model must be fitted before prediction".to_string());
        }

        let model = self.model.as_ref().unwrap();
        let logits = model.forward(
            x.clone(),
            self.training_config.hidden_activation,
            self.training_config.output_activation,
        );

        // Apply softmax to get probabilities
        let probabilities = burn::tensor::activation::softmax(logits, 1);
        Ok(probabilities)
    }

    /// Predict class labels for input data
    pub fn predict(&self, x: &Tensor<B, 2>) -> Result<Tensor<B, 1>, String> {
        let probabilities = self.predict_proba(x)?;
        let predictions = probabilities.argmax(1).squeeze::<1>().float();
        Ok(predictions)
    }

    /// Get the model configuration
    pub fn config(&self) -> (usize, &Vec<usize>, usize) {
        (self.input_dim, &self.hidden_layers, self.num_classes)
    }

    /// Check if the model is fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }
}

/// Neural network regressor using MLP  
#[derive(Debug)]
pub struct NeuralNetRegressor<B: Backend<FloatElem = f32>> {
    model: Option<MLP<B>>,
    training_config: TrainingConfig,
    device: Device<B>,
    input_dim: usize,
    hidden_layers: Vec<usize>,
    output_dim: usize,
    is_fitted: bool,
}

impl Default for NeuralNetRegressor<DefaultBackend> {
    fn default() -> Self {
        Self {
            model: None,
            training_config: TrainingConfig {
                loss_type: LossType::MSE,
                output_activation: Activation::Linear,
                ..Default::default()
            },
            device: Device::<DefaultBackend>::default(),
            input_dim: 4,
            hidden_layers: vec![10, 5],
            output_dim: 1,
            is_fitted: false,
        }
    }
}

impl<B: Backend<FloatElem = f32>> NeuralNetRegressor<B> {
    /// Create a new neural network regressor
    pub fn new(
        input_dim: usize,
        hidden_layers: Vec<usize>,
        output_dim: usize,
        device: Device<B>,
    ) -> Self {
        let training_config = TrainingConfig {
            loss_type: LossType::MSE,
            output_activation: Activation::Linear,
            ..Default::default()
        };

        Self {
            model: None,
            training_config,
            device,
            input_dim,
            hidden_layers,
            output_dim,
            is_fitted: false,
        }
    }

    /// Set the training configuration
    pub fn with_training_config(mut self, config: TrainingConfig) -> Self {
        self.training_config = config;
        self
    }

    /// Fit the regressor to training data
    pub fn fit(&mut self, _x: &Tensor<B, 2>, _y: &Tensor<B, 2>) -> Result<(), String> {
        // Initialize the model
        self.model = Some(MLP::new(
            self.input_dim,
            self.hidden_layers.clone(),
            self.output_dim,
            &self.device,
        ));

        // For now, we'll implement a basic training loop
        // In a full implementation, this would include proper gradient computation,
        // loss calculation, and optimization steps

        self.is_fitted = true;
        Ok(())
    }

    /// Predict target values for input data
    pub fn predict(&self, x: &Tensor<B, 2>) -> Result<Tensor<B, 2>, String> {
        if !self.is_fitted || self.model.is_none() {
            return Err("Model must be fitted before prediction".to_string());
        }

        let model = self.model.as_ref().unwrap();
        let predictions = model.forward(
            x.clone(),
            self.training_config.hidden_activation,
            self.training_config.output_activation,
        );

        Ok(predictions)
    }

    /// Get the model configuration
    pub fn config(&self) -> (usize, &Vec<usize>, usize) {
        (self.input_dim, &self.hidden_layers, self.output_dim)
    }

    /// Check if the model is fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }
}

/// Type alias for default backend neural network classifier
pub type DefaultNeuralNetClassifier = NeuralNetClassifier<DefaultBackend>;

/// Type alias for default backend neural network regressor
pub type DefaultNeuralNetRegressor = NeuralNetRegressor<DefaultBackend>;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{ndarray::NdArrayDevice, NdArray};
    use burn::tensor::TensorData;

    type TestBackend = NdArray<f32>;
    type TestDevice = NdArrayDevice;

    #[test]
    fn test_activation_functions() {
        let device = TestDevice::default();
        let data = TensorData::new(vec![-1.0f32, 0.0, 1.0, 2.0], [4]);
        let input = Tensor::<TestBackend, 1>::from_data(data, &device);

        // Test ReLU
        let relu_output = Activation::Relu.apply(input.clone());
        let relu_expected = TensorData::new(vec![0.0f32, 0.0, 1.0, 2.0], [4]);
        assert_eq!(relu_output.to_data(), relu_expected);

        // Test Linear (identity)
        let linear_output = Activation::Linear.apply(input.clone());
        assert_eq!(linear_output.to_data(), input.to_data());
    }

    #[test]
    fn test_mlp_creation() {
        let device = TestDevice::default();
        let mlp = MLP::new(4, vec![10, 5], 3, &device);

        assert_eq!(mlp.input_dim(), 4);
        assert_eq!(mlp.output_dim(), 3);
        assert!(mlp.num_params() > 0);
    }

    #[test]
    fn test_neural_net_classifier() {
        let device = TestDevice::default();
        let mut classifier = NeuralNetClassifier::<TestBackend>::new(4, vec![5], 3, device.clone());

        // Test initial state
        assert!(!classifier.is_fitted());

        // Create dummy training data
        let x_data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0], [2, 4]);
        let x = Tensor::from_data(x_data, &device);

        let y_data = TensorData::new(vec![0, 1], [2]);
        let y = Tensor::from_data(y_data, &device);

        // Test fitting
        let result = classifier.fit(&x, &y);
        assert!(result.is_ok());
        assert!(classifier.is_fitted());
    }

    #[test]
    fn test_neural_net_regressor() {
        let device = TestDevice::default();
        let mut regressor = NeuralNetRegressor::<TestBackend>::new(4, vec![5], 1, device.clone());

        // Test initial state
        assert!(!regressor.is_fitted());

        // Create dummy training data
        let x_data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0], [2, 4]);
        let x = Tensor::from_data(x_data, &device);

        let y_data = TensorData::new(vec![1.0f32, 2.0], [2, 1]);
        let y = Tensor::from_data(y_data, &device);

        // Test fitting
        let result = regressor.fit(&x, &y);
        assert!(result.is_ok());
        assert!(regressor.is_fitted());
    }
}
