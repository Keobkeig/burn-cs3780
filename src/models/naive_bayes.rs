use burn::backend::NdArray;
use burn::prelude::*;
use std::marker::PhantomData;

/// Type alias for the default backend used in examples
type DefaultBackend = NdArray<f32>;

/// Naive Bayes classifier for text classification using Burn tensors
pub struct NaiveBayesClassifier<B: Backend<FloatElem = f32>> {
    /// Weight vector for linear classifier representation
    pub weights: Option<Tensor<B, 1>>,
    /// Bias term  
    pub bias: f32,
    /// Feature dimension (26 for letters, 676 for letter pairs)
    pub feature_dim: usize,
    /// Class priors: P(y=1) and P(y=-1)
    pub pos_prior: f32,
    /// Prior probability for the negative class
    pub neg_prior: f32,
    /// Conditional probabilities: P(x_i|y=1) and P(x_i|y=-1)
    pub pos_probs: Option<Tensor<B, 1>>,
    /// Conditional feature probabilities for the negative class
    pub neg_probs: Option<Tensor<B, 1>>,
    /// Device for tensor operations
    pub device: Device<B>,
    /// Phantom data for backend type
    _phantom: PhantomData<B>,
}

/// Feature extraction types
#[derive(Clone, Copy)]
pub enum FeatureType {
    /// Single letters (26 features)
    Letters,
    /// Letter pairs/bigrams (676 features)
    LetterPairs,
}

impl FeatureType {
    /// Return the number of features for this feature type
    pub fn dimension(&self) -> usize {
        match self {
            FeatureType::Letters => 26,
            FeatureType::LetterPairs => 676,
        }
    }
}

impl<B: Backend<FloatElem = f32>> NaiveBayesClassifier<B> {
    /// Create a new Naive Bayes classifier
    pub fn new(feature_type: FeatureType, device: Device<B>) -> Self {
        let feature_dim = feature_type.dimension();

        Self {
            weights: None,
            bias: 0.0,
            feature_dim,
            pos_prior: 0.5,
            neg_prior: 0.5,
            pos_probs: None,
            neg_probs: None,
            device,
            _phantom: PhantomData,
        }
    }

    /// Train the Naive Bayes classifier
    ///
    /// # Arguments
    /// * `x` - Feature matrix of shape [n_samples, n_features]
    /// * `y` - Labels of shape [n_samples] where -1 = negative class, +1 = positive class
    /// * `use_smoothing` - Whether to use Laplace smoothing
    pub fn train(&mut self, x: &Tensor<B, 2>, y: &Tensor<B, 1>, use_smoothing: bool) {
        let [n_samples, n_features] = x.dims();
        assert_eq!(n_features, self.feature_dim, "Feature dimension mismatch");
        assert_eq!([n_samples], y.dims(), "Label dimension mismatch");

        // Convert tensors to data for easier manipulation
        let x_data = x.to_data().as_slice::<f32>().unwrap().to_vec();
        let y_data = y.to_data().as_slice::<f32>().unwrap().to_vec();

        // Reshape x_data into matrix format
        let mut x_matrix = Vec::new();
        for i in 0..n_samples {
            let start_idx = i * n_features;
            let end_idx = start_idx + n_features;
            x_matrix.push(x_data[start_idx..end_idx].to_vec());
        }

        // Convert y_data to i32
        let y_labels: Vec<i32> = y_data.iter().map(|&val| val as i32).collect();

        // Calculate using Vec-based computation then convert back to tensors
        self.calculate_class_priors_from_vec(&y_labels);

        if use_smoothing {
            self.calculate_conditional_probs_smoothing_from_vec(&x_matrix, &y_labels);
        } else {
            self.calculate_conditional_probs_mle_from_vec(&x_matrix, &y_labels);
        }

        // Convert to linear classifier: w_i = log(P(x_i|y=1)) - log(P(x_i|y=-1))
        let pos_probs = self.pos_probs.as_ref().unwrap();
        let neg_probs = self.neg_probs.as_ref().unwrap();

        let weights = pos_probs.clone().log() - neg_probs.clone().log();
        self.weights = Some(weights);

        // Bias: log(P(y=1)) - log(P(y=-1))
        self.bias = (self.pos_prior / self.neg_prior).ln();
    }

    /// Predict class labels for input data
    ///
    /// # Arguments
    /// * `x` - Feature matrix of shape [n_samples, n_features]
    ///
    /// # Returns
    /// * Tensor of predictions of shape [n_samples] with values -1 or +1
    pub fn predict(&self, x: &Tensor<B, 2>) -> Tensor<B, 1> {
        let scores = self.predict_scores(x);

        // Convert to Vec for easier manipulation
        let scores_data = scores.to_data().as_slice::<f32>().unwrap().to_vec();

        // Convert scores to predictions: score >= 0 -> 1, score < 0 -> -1
        let predictions: Vec<f32> = scores_data
            .iter()
            .map(|&score| if score >= 0.0 { 1.0 } else { -1.0 })
            .collect();

        Tensor::<B, 1>::from_floats(predictions.as_slice(), &self.device)
    }

    /// Calculate prediction scores for input data
    ///
    /// # Arguments
    /// * `x` - Feature matrix of shape [n_samples, n_features]
    ///
    /// # Returns
    /// * Tensor of scores of shape [n_samples]
    pub fn predict_scores(&self, x: &Tensor<B, 2>) -> Tensor<B, 1> {
        let weights = self.weights.as_ref().expect("Model not trained");

        // scores = X @ weights + bias
        let scores = x
            .clone()
            .matmul(weights.clone().unsqueeze_dim(1))
            .squeeze::<1>();
        let [n_samples] = scores.dims();
        let bias_tensor =
            Tensor::<B, 1>::from_floats(vec![self.bias; n_samples].as_slice(), &self.device);

        scores + bias_tensor
    }

    /// Calculate log probability ratio: log(P(Y=1|X) / P(Y=-1|X))
    pub fn log_probability_ratio(&self, x: &Tensor<B, 2>) -> Tensor<B, 1> {
        let pos_probs = self.pos_probs.as_ref().expect("Model not trained");
        let neg_probs = self.neg_probs.as_ref().expect("Model not trained");

        // log_ratio = log(P(y=1)/P(y=-1)) + sum(x_i * (log(P(x_i|y=1)) - log(P(x_i|y=-1))))
        let prior_ratio = self.bias; // Already computed as log(pos_prior / neg_prior)
        let feature_ratio = pos_probs.clone().log() - neg_probs.clone().log();

        let weighted_features = x.clone() * feature_ratio.clone().unsqueeze_dim(0);
        let feature_sum = weighted_features.sum_dim(1).squeeze(); // Squeeze to get 1D tensor

        let [n_samples] = feature_sum.dims();
        let prior_tensor =
            Tensor::<B, 1>::from_floats(vec![prior_ratio; n_samples].as_slice(), &self.device);

        feature_sum + prior_tensor
    }

    /// Calculate class priors P(Y) from Vec data
    fn calculate_class_priors_from_vec(&mut self, y: &[i32]) {
        let n_samples = y.len() as f32;
        let pos_count = y.iter().filter(|&&label| label == 1).count() as f32;
        let neg_count = y.iter().filter(|&&label| label == -1).count() as f32;

        self.pos_prior = pos_count / n_samples;
        self.neg_prior = neg_count / n_samples;
    }

    /// Calculate conditional probabilities P(X|Y) using Maximum Likelihood Estimation from Vec data
    fn calculate_conditional_probs_mle_from_vec(&mut self, x: &[Vec<f32>], y: &[i32]) {
        let n_features = self.feature_dim;
        let mut pos_feature_counts = vec![0.0; n_features];
        let mut neg_feature_counts = vec![0.0; n_features];
        let mut pos_total = 0.0;
        let mut neg_total = 0.0;

        // Count feature occurrences for each class
        for (feature_vec, &label) in x.iter().zip(y.iter()) {
            let is_positive = label == 1;

            for (i, &feature_count) in feature_vec.iter().enumerate() {
                if is_positive {
                    pos_feature_counts[i] += feature_count;
                    pos_total += feature_count;
                } else {
                    neg_feature_counts[i] += feature_count;
                    neg_total += feature_count;
                }
            }
        }

        // Calculate probabilities
        let pos_probs: Vec<f32> = pos_feature_counts
            .iter()
            .map(|&count| count / pos_total)
            .collect();

        let neg_probs: Vec<f32> = neg_feature_counts
            .iter()
            .map(|&count| count / neg_total)
            .collect();

        // Convert to tensors
        self.pos_probs = Some(Tensor::<B, 1>::from_floats(
            pos_probs.as_slice(),
            &self.device,
        ));
        self.neg_probs = Some(Tensor::<B, 1>::from_floats(
            neg_probs.as_slice(),
            &self.device,
        ));
    }

    /// Calculate conditional probabilities P(X|Y) using Laplace smoothing from Vec data
    fn calculate_conditional_probs_smoothing_from_vec(&mut self, x: &[Vec<f32>], y: &[i32]) {
        let n_features = self.feature_dim;
        let mut pos_feature_counts = vec![0.0; n_features];
        let mut neg_feature_counts = vec![0.0; n_features];
        let mut pos_total = 0.0;
        let mut neg_total = 0.0;

        // Count feature occurrences for each class
        for (feature_vec, &label) in x.iter().zip(y.iter()) {
            let is_positive = label == 1;

            for (i, &feature_count) in feature_vec.iter().enumerate() {
                if is_positive {
                    pos_feature_counts[i] += feature_count;
                    pos_total += feature_count;
                } else {
                    neg_feature_counts[i] += feature_count;
                    neg_total += feature_count;
                }
            }
        }

        // Apply Laplace smoothing: (count + 1) / (total + |vocabulary|)
        let smoothing_factor = n_features as f32;

        let pos_probs: Vec<f32> = pos_feature_counts
            .iter()
            .map(|&count| (count + 1.0) / (pos_total + smoothing_factor))
            .collect();

        let neg_probs: Vec<f32> = neg_feature_counts
            .iter()
            .map(|&count| (count + 1.0) / (neg_total + smoothing_factor))
            .collect();

        // Convert to tensors
        self.pos_probs = Some(Tensor::<B, 1>::from_floats(
            pos_probs.as_slice(),
            &self.device,
        ));
        self.neg_probs = Some(Tensor::<B, 1>::from_floats(
            neg_probs.as_slice(),
            &self.device,
        ));
    }
}

/// Extract features from a word using letter frequencies
pub fn extract_letter_features(word: &str) -> Vec<f32> {
    let mut features = vec![0.0; 26];

    for ch in word.chars() {
        if ch.is_ascii_lowercase() {
            let idx = (ch as u8 - b'a') as usize;
            if idx < 26 {
                features[idx] += 1.0;
            }
        }
    }

    features
}

/// Extract features from a word using letter pair (bigram) frequencies  
pub fn extract_letter_pair_features(word: &str) -> Vec<f32> {
    let mut features = vec![0.0; 676]; // 26 * 26
    let chars: Vec<char> = word.chars().filter(|c| c.is_ascii_lowercase()).collect();

    for i in 0..(chars.len().saturating_sub(1)) {
        let first_idx = (chars[i] as u8 - b'a') as usize;
        let second_idx = (chars[i + 1] as u8 - b'a') as usize;

        if first_idx < 26 && second_idx < 26 {
            let pair_idx = first_idx * 26 + second_idx;
            features[pair_idx] += 1.0;
        }
    }

    features
}

/// Extract features from a list of words and convert to tensor
pub fn extract_features<B: Backend<FloatElem = f32>>(
    words: &[String],
    feature_type: FeatureType,
    device: &Device<B>,
) -> Tensor<B, 2> {
    let n_words = words.len();
    let feature_dim = feature_type.dimension();

    let mut feature_matrix = Vec::with_capacity(n_words * feature_dim);

    for word in words {
        let word_lower = word.to_lowercase();
        let features = match feature_type {
            FeatureType::Letters => extract_letter_features(&word_lower),
            FeatureType::LetterPairs => extract_letter_pair_features(&word_lower),
        };
        feature_matrix.extend(features);
    }

    Tensor::<B, 1>::from_floats(feature_matrix.as_slice(), device).reshape([n_words, feature_dim])
}

/// Convert labels vector to tensor
pub fn labels_to_tensor<B: Backend<FloatElem = f32>>(
    labels: &[i32],
    device: &Device<B>,
) -> Tensor<B, 1> {
    let data: Vec<f32> = labels.iter().map(|&x| x as f32).collect();
    Tensor::<B, 1>::from_floats(data.as_slice(), device)
}

/// Convert tensor predictions back to Vec<i32> for compatibility
pub fn tensor_to_labels<B: Backend<FloatElem = f32>>(tensor: &Tensor<B, 1>) -> Vec<i32> {
    let data = tensor.to_data();
    data.as_slice::<f32>()
        .unwrap()
        .iter()
        .map(|&x| x as i32)
        .collect()
}

/// Calculate classification accuracy
pub fn calculate_accuracy(predictions: &[i32], true_labels: &[i32]) -> f32 {
    if predictions.len() != true_labels.len() {
        return 0.0;
    }

    let correct = predictions
        .iter()
        .zip(true_labels.iter())
        .filter(|(pred, true_label)| pred == true_label)
        .count();

    correct as f32 / predictions.len() as f32
}

/// Generate synthetic German and English words for testing
pub fn generate_synthetic_language_data() -> (Vec<String>, Vec<String>) {
    // Synthetic German words (tend to have 'sch', 'ie', 'ei', 'ung', etc.)
    let german_words = vec![
        "hallo",
        "welt",
        "schoen",
        "deutsch",
        "sprechen",
        "verstehen",
        "arbeiten",
        "studieren",
        "universitat",
        "bibliothek",
        "wissenschaft",
        "mathematik",
        "geschichte",
        "gesellschaft",
        "wirtschaft",
        "politik",
        "entwicklung",
        "forschung",
        "technologie",
        "engineering",
        "maschine",
        "fahrzeug",
        "gebaeude",
        "landschaft",
        "freundschaft",
        "gemeinschaft",
        "verantwortung",
        "erfahrung",
        "bedeutung",
        "entscheidung",
        "diskussion",
        "schatten",
        "zwischen",
        "waehrend",
        "verschiedene",
        "besonders",
        "moegliche",
        "wichtige",
        "oeffentliche",
        "internationale",
        "europaeische",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    // Synthetic English words (tend to have 'th', 'ing', 'tion', etc.)
    let english_words = vec![
        "hello",
        "world",
        "beautiful",
        "english",
        "speaking",
        "understanding",
        "working",
        "studying",
        "university",
        "library",
        "science",
        "mathematics",
        "history",
        "society",
        "economy",
        "politics",
        "development",
        "research",
        "technology",
        "engineering",
        "machine",
        "vehicle",
        "building",
        "landscape",
        "friendship",
        "community",
        "responsibility",
        "experience",
        "meaning",
        "decision",
        "discussion",
        "shadow",
        "between",
        "during",
        "different",
        "especially",
        "possible",
        "important",
        "public",
        "international",
        "european",
        "thinking",
        "learning",
        "teaching",
        "reading",
        "writing",
        "listening",
        "watching",
        "playing",
        "running",
        "walking",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    (german_words, english_words)
}

/// Convenience alias for the default NdArray backend
pub type NaiveBayesClassifierDefault = NaiveBayesClassifier<DefaultBackend>;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArrayDevice;

    #[test]
    fn test_letter_features() {
        let features = extract_letter_features("hello");
        assert_eq!(features.len(), 26);
        assert_eq!(features[7], 1.0); // 'h'
        assert_eq!(features[4], 1.0); // 'e'
        assert_eq!(features[11], 2.0); // 'l' appears twice
        assert_eq!(features[14], 1.0); // 'o'
    }

    #[test]
    fn test_letter_pair_features() {
        let features = extract_letter_pair_features("mama");
        assert_eq!(features.len(), 676);

        // "ma" pair should occur twice
        let ma_idx = (b'm' - b'a') as usize * 26 + (b'a' - b'a') as usize; // 12*26 + 0 = 312
        assert_eq!(features[ma_idx], 2.0);

        // "am" pair should occur once
        let am_idx = (b'a' - b'a') as usize * 26 + (b'm' - b'a') as usize; // 0*26 + 12 = 12
        assert_eq!(features[am_idx], 1.0);
    }

    #[test]
    fn test_feature_extraction() {
        let device = NdArrayDevice::default();
        let words = vec!["hello".to_string(), "world".to_string()];
        let features = extract_features::<DefaultBackend>(&words, FeatureType::Letters, &device);
        assert_eq!(features.dims(), [2, 26]);
    }

    #[test]
    fn test_naive_bayes_training() {
        let device = NdArrayDevice::default();
        let words = vec![
            "hello".to_string(),
            "world".to_string(),
            "hallo".to_string(),
            "welt".to_string(),
        ];
        let features = extract_features::<DefaultBackend>(&words, FeatureType::Letters, &device);
        let labels = labels_to_tensor::<DefaultBackend>(&[1, 1, -1, -1], &device);

        let mut nb = NaiveBayesClassifier::<DefaultBackend>::new(FeatureType::Letters, device);
        nb.train(&features, &labels, true);

        // Basic sanity checks
        assert!(nb.weights.is_some());
        assert!(nb.pos_probs.is_some());
        assert!(nb.neg_probs.is_some());

        // Should be able to make predictions
        let predictions_tensor = nb.predict(&features);
        let predictions = tensor_to_labels(&predictions_tensor);
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_synthetic_data() {
        let (german_words, english_words) = generate_synthetic_language_data();
        assert!(german_words.len() >= 20);
        assert!(english_words.len() >= 20);

        // Should contain typical German/English characteristics
        assert!(german_words
            .iter()
            .any(|w| w.contains("sch") || w.contains("ung")));
        assert!(english_words
            .iter()
            .any(|w| w.contains("ing") || w.contains("tion")));
    }

    #[test]
    fn test_accuracy_calculation() {
        let pred = vec![1, -1, 1, -1];
        let true_labels = vec![1, -1, -1, -1]; // One incorrect
        let accuracy = calculate_accuracy(&pred, &true_labels);
        assert_eq!(accuracy, 0.75);
    }
}
