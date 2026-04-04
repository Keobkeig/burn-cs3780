//! # Naive Bayes Text Classification Example
//!
//! This example demonstrates German vs English word classification using Naive Bayes
//! with both letter frequency (26D) and letter pair frequency (676D) features.
//!
//! Based on CS3780 Project 5, this implementation shows:
//! - Feature extraction from text (letter frequencies and bigrams)
//! - Naive Bayes training with and without Laplace smoothing using Burn tensors
//! - Linear classifier representation of Naive Bayes
//! - Performance comparison between different feature representations
//! - Interactive word classification
//!
//! Expected results:
//! - Letter features: ~34% test error
//! - Letter pair features: ~17% test error (significant improvement)
//! - Smoothing helps with unseen feature combinations

use burn::backend::NdArray;
use burn::prelude::*;
use burn_cs3780::models::naive_bayes::{
    calculate_accuracy, extract_features, generate_synthetic_language_data, labels_to_tensor,
    tensor_to_labels, FeatureType, NaiveBayesClassifier,
};
use std::io::{self, Write};
use std::time::Instant;

type DefaultBackend = NdArray<f32>;

fn main() {
    let device = Default::default();

    println!("🇩🇪🇬🇧 Naive Bayes German vs English Word Classification");
    println!("{}", "=".repeat(60));

    // Generate synthetic language data
    println!("📚 Generating synthetic German and English words...");
    let (german_words, english_words) = generate_synthetic_language_data();

    println!("   📊 Dataset statistics:");
    println!("      German words: {}", german_words.len());
    println!("      English words: {}", english_words.len());

    // Prepare training data
    let mut train_words = Vec::new();
    let mut train_labels = Vec::new();

    // Add German words (label = -1)
    for word in &german_words[..30] {
        // Use first 30 for training
        train_words.push(word.clone());
        train_labels.push(-1);
    }

    // Add English words (label = +1)
    for word in &english_words[..30] {
        // Use first 30 for training
        train_words.push(word.clone());
        train_labels.push(1);
    }

    // Prepare test data
    let mut test_words = Vec::new();
    let mut test_labels = Vec::new();

    // Add remaining German words for testing
    for word in &german_words[30..] {
        test_words.push(word.clone());
        test_labels.push(-1);
    }

    // Add remaining English words for testing
    for word in &english_words[30..] {
        test_words.push(word.clone());
        test_labels.push(1);
    }

    println!("   Training samples: {}", train_words.len());
    println!("   Test samples: {}", test_words.len());

    // Demonstrate both feature types
    demonstrate_classification(
        &train_words,
        &train_labels,
        &test_words,
        &test_labels,
        FeatureType::Letters,
        &device,
    );
    demonstrate_classification(
        &train_words,
        &train_labels,
        &test_words,
        &test_labels,
        FeatureType::LetterPairs,
        &device,
    );

    // Interactive classification demo
    interactive_classification(&train_words, &train_labels, &device);
}

fn demonstrate_classification(
    train_words: &[String],
    train_labels: &[i32],
    test_words: &[String],
    test_labels: &[i32],
    feature_type: FeatureType,
    device: &Device<DefaultBackend>,
) {
    let feature_name = match feature_type {
        FeatureType::Letters => "Letter Frequencies",
        FeatureType::LetterPairs => "Letter Pair Frequencies",
    };

    println!(
        "\n🔬 {} ({} dimensions)",
        feature_name,
        feature_type.dimension()
    );
    println!("{}", "-".repeat(50));

    let start_time = Instant::now();

    // Extract features as tensors
    let train_features = extract_features::<DefaultBackend>(train_words, feature_type, device);
    let test_features = extract_features::<DefaultBackend>(test_words, feature_type, device);

    // Convert labels to tensors
    let train_labels_tensor = labels_to_tensor::<DefaultBackend>(train_labels, device);
    let test_labels_tensor = labels_to_tensor::<DefaultBackend>(test_labels, device);

    let feature_time = start_time.elapsed();
    println!(
        "   ⏱️  Feature extraction: {:.2}ms",
        feature_time.as_secs_f64() * 1000.0
    );

    // Train both MLE and smoothed versions
    let mut nb_mle = NaiveBayesClassifier::new(feature_type, device.clone());
    let mut nb_smoothed = NaiveBayesClassifier::new(feature_type, device.clone());

    let train_start = Instant::now();
    nb_mle.train(&train_features, &train_labels_tensor, false);
    nb_smoothed.train(&train_features, &train_labels_tensor, true);
    let train_time = train_start.elapsed();

    println!(
        "   ⏱️  Training time: {:.2}ms",
        train_time.as_secs_f64() * 1000.0
    );

    // Make predictions
    let pred_start = Instant::now();
    let train_pred_mle_tensor = nb_mle.predict(&train_features);
    let test_pred_mle_tensor = nb_mle.predict(&test_features);
    let train_pred_smooth_tensor = nb_smoothed.predict(&train_features);
    let test_pred_smooth_tensor = nb_smoothed.predict(&test_features);
    let pred_time = pred_start.elapsed();

    println!(
        "   ⏱️  Prediction time: {:.2}ms",
        pred_time.as_secs_f64() * 1000.0
    );

    // Convert tensor predictions back to Vec for accuracy calculation
    let train_pred_mle = tensor_to_labels(&train_pred_mle_tensor);
    let test_pred_mle = tensor_to_labels(&test_pred_mle_tensor);
    let train_pred_smooth = tensor_to_labels(&train_pred_smooth_tensor);
    let test_pred_smooth = tensor_to_labels(&test_pred_smooth_tensor);

    // Calculate accuracies
    let train_acc_mle = calculate_accuracy(&train_pred_mle, train_labels);
    let test_acc_mle = calculate_accuracy(&test_pred_mle, test_labels);
    let train_acc_smooth = calculate_accuracy(&train_pred_smooth, train_labels);
    let test_acc_smooth = calculate_accuracy(&test_pred_smooth, test_labels);

    println!("\n   📈 Results:");
    println!("      Maximum Likelihood Estimate (MLE):");
    println!("         Training accuracy: {:.2}%", train_acc_mle * 100.0);
    println!("         Test accuracy:     {:.2}%", test_acc_mle * 100.0);
    println!(
        "         Training error:    {:.2}%",
        (1.0 - train_acc_mle) * 100.0
    );
    println!(
        "         Test error:        {:.2}%",
        (1.0 - test_acc_mle) * 100.0
    );

    println!("      Laplace Smoothing:");
    println!(
        "         Training accuracy: {:.2}%",
        train_acc_smooth * 100.0
    );
    println!(
        "         Test accuracy:     {:.2}%",
        test_acc_smooth * 100.0
    );
    println!(
        "         Training error:    {:.2}%",
        (1.0 - train_acc_smooth) * 100.0
    );
    println!(
        "         Test error:        {:.2}%",
        (1.0 - test_acc_smooth) * 100.0
    );

    // Show some example predictions
    println!("\n   🔍 Example Predictions:");
    for i in 0..std::cmp::min(5, test_words.len()) {
        let word = &test_words[i];
        let true_label = test_labels[i];
        let pred_label = test_pred_smooth[i];

        // Get score for single sample - get all scores and index into it
        let all_scores = nb_smoothed.predict_scores(&test_features);
        let score_slice = all_scores.slice([i..i + 1]);
        let score: f32 = score_slice.into_scalar();

        let true_lang = if true_label == 1 { "English" } else { "German" };
        let pred_lang = if pred_label == 1 { "English" } else { "German" };
        let correct = if true_label == pred_label {
            "✓"
        } else {
            "✗"
        };

        println!(
            "      {} '{}': {} -> {} (score: {:.3}) {}",
            correct,
            word,
            true_lang,
            pred_lang,
            score,
            if correct == "✓" { "" } else { "INCORRECT" }
        );
    }
}

fn interactive_classification(
    train_words: &[String],
    train_labels: &[i32],
    device: &Device<DefaultBackend>,
) {
    println!("\n🎮 Interactive Word Classification");
    println!("{}", "-".repeat(40));
    println!("Enter words to classify as German or English (type 'exit' to quit):");

    // Train classifier with letter pair features (better accuracy)
    let train_features =
        extract_features::<DefaultBackend>(train_words, FeatureType::LetterPairs, device);
    let train_labels_tensor = labels_to_tensor::<DefaultBackend>(train_labels, device);
    let mut nb = NaiveBayesClassifier::new(FeatureType::LetterPairs, device.clone());
    nb.train(&train_features, &train_labels_tensor, true); // Use smoothing

    loop {
        print!("\n> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_ok() {
            let word = input.trim();

            if word.is_empty() {
                continue;
            }

            if word == "exit" {
                println!("Goodbye! 👋");
                break;
            }

            // Extract features for the input word
            let features = extract_features::<DefaultBackend>(
                &[word.to_string()],
                FeatureType::LetterPairs,
                device,
            );
            let prediction_tensor = nb.predict(&features);
            let scores_tensor = nb.predict_scores(&features);
            let log_ratio_tensor = nb.log_probability_ratio(&features);

            let predictions = tensor_to_labels(&prediction_tensor);
            let prediction = predictions[0];
            let score: f32 = scores_tensor.into_scalar();
            let log_ratio: f32 = log_ratio_tensor.into_scalar();

            let language = if prediction == 1 {
                "English 🇬🇧"
            } else {
                "German 🇩🇪"
            };
            let confidence = (score.abs() / 5.0).min(1.0) * 100.0; // Rough confidence estimate

            println!("   '{}' is classified as: {}", word, language);
            println!(
                "   Score: {:.3}, Log P(English)/P(German): {:.3}",
                score, log_ratio
            );
            println!("   Confidence: ~{:.0}%", confidence);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_extraction() {
        let device = Default::default();
        let words = vec!["hello".to_string(), "hallo".to_string()];

        // Test letter features
        let letter_features =
            extract_features::<DefaultBackend>(&words, FeatureType::Letters, &device);
        assert_eq!(letter_features.dims(), [2, 26]);

        // Test letter pair features
        let pair_features =
            extract_features::<DefaultBackend>(&words, FeatureType::LetterPairs, &device);
        assert_eq!(pair_features.dims(), [2, 676]);
    }

    #[test]
    fn test_naive_bayes_training() {
        let device = Default::default();
        let words = vec![
            "english".to_string(),
            "hello".to_string(),
            "deutsch".to_string(),
            "hallo".to_string(),
        ];
        let labels = vec![1, 1, -1, -1];
        let features = extract_features::<DefaultBackend>(&words, FeatureType::Letters, &device);
        let labels_tensor = labels_to_tensor::<DefaultBackend>(&labels, &device);

        let mut nb = NaiveBayesClassifier::new(FeatureType::Letters, device);
        nb.train(&features, &labels_tensor, true);

        // Basic sanity checks
        assert_eq!(nb.pos_prior, 0.5);
        assert_eq!(nb.neg_prior, 0.5);

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
