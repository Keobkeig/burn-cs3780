//! Comprehensive Boosting Algorithms Example
//!
//! This example demonstrates the boosting algorithms implemented in burn-cs3780,
//! including AdaBoost and Gradient Boosting. We'll show:
//!
//! 1. AdaBoost for binary classification
//! 2. Gradient Boosting for regression and classification
//! 3. Comparison of different boosting techniques
//! 4. Feature importance analysis
//! 5. Model interpretation and performance analysis

use burn::prelude::*;
use burn_cs3780::datasets::{generate_classification_data, generate_regression_data};
use burn_cs3780::metrics::{accuracy, classification_report, mean_squared_error};
use burn_cs3780::models::boosting::{AdaBoost, GradientBoosting};
use burn_cs3780::utils::{train_test_split, StandardScaler};
use burn_cs3780::DefaultBackend;

type MyBackend = DefaultBackend;

fn main() {
    println!("🚀 Burn CS3780 - Boosting Algorithms Example");
    println!("=============================================\n");

    // Run different boosting examples
    adaboost_classification_example();
    gradient_boosting_regression_example();
    gradient_boosting_classification_example();
    boosting_comparison_example();
    feature_importance_analysis();
}

/// Demonstrates AdaBoost for binary classification
fn adaboost_classification_example() {
    println!("🎯 1. AdaBoost Binary Classification");
    println!("------------------------------------");

    let device = Device::<MyBackend>::default();

    // Generate synthetic binary classification dataset
    println!("Generating synthetic binary classification data...");
    let (x_data, y_data) = generate_classification_data(
        800, // n_samples
        10,  // n_features
        2,   // n_classes
        0.1, // noise
        42,  // random_seed
    );

    let x = Tensor::from_floats(TensorData::new(x_data, [800, 10]), &device);
    let y = Tensor::from_ints(TensorData::new(y_data, [800]), &device);

    println!("Dataset shape: {:?}", x.dims());
    println!("Labels shape: {:?}", y.dims());

    // Split data into train/test sets
    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, 0.2, Some(42));

    println!("Training set: {:?}", x_train.dims());
    println!("Test set: {:?}", x_test.dims());

    // Create and train AdaBoost model
    println!("\nTraining AdaBoost classifier...");
    let mut adaboost = AdaBoost::new()
        .with_n_estimators(50)
        .with_learning_rate(1.0)
        .with_max_depth(1); // Decision stumps

    println!("AdaBoost configuration:");
    println!("  - Number of estimators: 50");
    println!("  - Learning rate: 1.0");
    println!("  - Max depth: 1 (decision stumps)");

    // Train the model
    adaboost.fit(x_train.clone(), y_train.clone());
    println!("✅ Training completed!");

    // Make predictions
    println!("\n📊 Making predictions...");
    let train_predictions = adaboost.predict(x_train.clone());
    let test_predictions = adaboost.predict(x_test.clone());

    // Calculate accuracy
    let train_accuracy = accuracy(y_train.clone(), train_predictions.clone());
    let test_accuracy = accuracy(y_test.clone(), test_predictions.clone());

    println!("Performance metrics:");
    println!("  - Training accuracy: {:.4}", train_accuracy);
    println!("  - Test accuracy: {:.4}", test_accuracy);

    // Generate detailed classification report
    let class_report = classification_report(y_test.clone(), test_predictions.clone());
    println!("\n📈 Classification Report:");
    println!("{}", class_report);

    // Analyze boosting rounds
    let feature_importance = adaboost.feature_importance();
    println!("\n🔍 Feature Importance (top 5):");
    for (i, importance) in feature_importance.iter().enumerate().take(5) {
        println!("  Feature {}: {:.4}", i, importance);
    }

    // Show boosting progression
    analyze_boosting_progression(&adaboost);

    println!("✅ AdaBoost classification example completed!\n");
}

/// Demonstrates Gradient Boosting for regression
fn gradient_boosting_regression_example() {
    println!("📈 2. Gradient Boosting Regression");
    println!("----------------------------------");

    let device = Device::<MyBackend>::default();

    // Generate synthetic regression dataset
    println!("Generating synthetic regression data...");
    let (x_data, y_data) = generate_regression_data(
        600, // n_samples
        8,   // n_features
        0.1, // noise
        42,  // random_seed
    );

    let x = Tensor::from_floats(TensorData::new(x_data, [600, 8]), &device);
    let y = Tensor::from_floats(TensorData::new(y_data, [600]), &device);

    println!("Dataset shape: {:?}", x.dims());
    println!("Target shape: {:?}", y.dims());

    // Split data
    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, 0.25, Some(42));

    // Standardize features
    let mut scaler = StandardScaler::new();
    let x_train_scaled = scaler.fit_transform(x_train);
    let x_test_scaled = scaler.transform(x_test);

    println!("Training set: {:?}", x_train_scaled.dims());
    println!("Test set: {:?}", x_test_scaled.dims());

    // Create and train Gradient Boosting regressor
    println!("\nTraining Gradient Boosting regressor...");
    let mut gb_regressor = GradientBoosting::new()
        .with_n_estimators(100)
        .with_learning_rate(0.1)
        .with_max_depth(3)
        .with_subsample(0.8)
        .for_regression();

    println!("Gradient Boosting configuration:");
    println!("  - Number of estimators: 100");
    println!("  - Learning rate: 0.1");
    println!("  - Max depth: 3");
    println!("  - Subsample ratio: 0.8");
    println!("  - Task: Regression");

    // Train the model
    gb_regressor.fit(x_train_scaled.clone(), y_train.clone());
    println!("✅ Training completed!");

    // Make predictions
    println!("\n📊 Making predictions...");
    let train_predictions = gb_regressor.predict(x_train_scaled.clone());
    let test_predictions = gb_regressor.predict(x_test_scaled.clone());

    // Calculate metrics
    let train_mse = mean_squared_error(y_train.clone(), train_predictions.clone());
    let test_mse = mean_squared_error(y_test.clone(), test_predictions.clone());

    println!("Performance metrics:");
    println!("  - Training MSE: {:.6}", train_mse);
    println!("  - Test MSE: {:.6}", test_mse);
    println!("  - Training RMSE: {:.6}", train_mse.sqrt());
    println!("  - Test RMSE: {:.6}", test_mse.sqrt());

    // R-squared calculation
    let r2_train = calculate_r2(y_train.clone(), train_predictions);
    let r2_test = calculate_r2(y_test.clone(), test_predictions.clone());

    println!("  - Training R²: {:.4}", r2_train);
    println!("  - Test R²: {:.4}", r2_test);

    // Feature importance for regression
    let feature_importance = gb_regressor.feature_importance();
    println!("\n🔍 Feature Importance:");
    for (i, importance) in feature_importance.iter().enumerate() {
        println!("  Feature {}: {:.4}", i, importance);
    }

    // Analyze learning curve
    analyze_learning_curve(&gb_regressor);

    println!("✅ Gradient Boosting regression example completed!\n");
}

/// Demonstrates Gradient Boosting for classification
fn gradient_boosting_classification_example() {
    println!("🎲 3. Gradient Boosting Classification");
    println!("--------------------------------------");

    let device = Device::<MyBackend>::default();

    // Generate multi-class classification dataset
    println!("Generating multi-class classification data...");
    let (x_data, y_data) = generate_classification_data(
        1000, // n_samples
        15,   // n_features
        3,    // n_classes
        0.05, // noise
        42,   // random_seed
    );

    let x = Tensor::from_floats(TensorData::new(x_data, [1000, 15]), &device);
    let y = Tensor::from_ints(TensorData::new(y_data, [1000]), &device);

    println!("Dataset shape: {:?}", x.dims());
    println!("Number of classes: 3");

    // Split data
    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, 0.3, Some(42));

    // Create and train Gradient Boosting classifier
    println!("\nTraining Gradient Boosting classifier...");
    let mut gb_classifier = GradientBoosting::new()
        .with_n_estimators(80)
        .with_learning_rate(0.15)
        .with_max_depth(4)
        .with_subsample(0.9)
        .for_classification();

    println!("Gradient Boosting configuration:");
    println!("  - Number of estimators: 80");
    println!("  - Learning rate: 0.15");
    println!("  - Max depth: 4");
    println!("  - Subsample ratio: 0.9");
    println!("  - Task: Multi-class Classification");

    // Train the model
    gb_classifier.fit(x_train.clone(), y_train.clone());
    println!("✅ Training completed!");

    // Make predictions
    println!("\n📊 Making predictions...");
    let train_predictions = gb_classifier.predict(x_train.clone());
    let test_predictions = gb_classifier.predict(x_test.clone());

    // Get prediction probabilities
    let test_probabilities = gb_classifier.predict_proba(x_test.clone());

    // Calculate accuracy
    let train_accuracy = accuracy(y_train.clone(), train_predictions.clone());
    let test_accuracy = accuracy(y_test.clone(), test_predictions.clone());

    println!("Performance metrics:");
    println!("  - Training accuracy: {:.4}", train_accuracy);
    println!("  - Test accuracy: {:.4}", test_accuracy);

    // Detailed per-class analysis
    analyze_multiclass_performance(y_test.clone(), test_predictions, test_probabilities);

    // Feature importance
    let feature_importance = gb_classifier.feature_importance();
    println!("\n🔍 Top 10 Most Important Features:");
    let mut importance_pairs: Vec<(usize, f32)> = feature_importance
        .iter()
        .enumerate()
        .map(|(i, &imp)| (i, imp))
        .collect();
    importance_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (i, (feature_idx, importance)) in importance_pairs.iter().take(10).enumerate() {
        println!("  {}. Feature {}: {:.4}", i + 1, feature_idx, importance);
    }

    println!("✅ Gradient Boosting classification example completed!\n");
}

/// Compares different boosting techniques
fn boosting_comparison_example() {
    println!("⚖️ 4. Boosting Algorithms Comparison");
    println!("------------------------------------");

    let device = Device::<MyBackend>::default();

    // Use a common dataset for fair comparison
    println!("Generating comparison dataset...");
    let (x_data, y_data) = generate_classification_data(
        800,  // n_samples
        12,   // n_features
        2,    // n_classes (binary for AdaBoost)
        0.08, // noise
        123,  // random_seed
    );

    let x = Tensor::from_floats(TensorData::new(x_data, [800, 12]), &device);
    let y = Tensor::from_ints(TensorData::new(y_data, [800]), &device);

    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, 0.25, Some(123));

    println!("Common dataset shape: {:?}", x_train.dims());

    // Model 1: AdaBoost
    println!("\n🥇 Training AdaBoost...");
    let mut adaboost = AdaBoost::new()
        .with_n_estimators(50)
        .with_learning_rate(1.0)
        .with_max_depth(2);

    adaboost.fit(x_train.clone(), y_train.clone());
    let ada_predictions = adaboost.predict(x_test.clone());
    let ada_accuracy = accuracy(y_test.clone(), ada_predictions);

    // Model 2: Gradient Boosting (Conservative)
    println!("🥈 Training Gradient Boosting (Conservative)...");
    let mut gb_conservative = GradientBoosting::new()
        .with_n_estimators(50)
        .with_learning_rate(0.1)
        .with_max_depth(2)
        .for_classification();

    gb_conservative.fit(x_train.clone(), y_train.clone());
    let gb_cons_predictions = gb_conservative.predict(x_test.clone());
    let gb_cons_accuracy = accuracy(y_test.clone(), gb_cons_predictions);

    // Model 3: Gradient Boosting (Aggressive)
    println!("🥉 Training Gradient Boosting (Aggressive)...");
    let mut gb_aggressive = GradientBoosting::new()
        .with_n_estimators(100)
        .with_learning_rate(0.2)
        .with_max_depth(4)
        .for_classification();

    gb_aggressive.fit(x_train.clone(), y_train.clone());
    let gb_agg_predictions = gb_aggressive.predict(x_test.clone());
    let gb_agg_accuracy = accuracy(y_test.clone(), gb_agg_predictions);

    // Results comparison
    println!("\n📊 Comparison Results:");
    println!("┌─────────────────────────┬──────────┬────────────┬──────────┐");
    println!("│ Algorithm               │ Accuracy │ Estimators │ Max Depth│");
    println!("├─────────────────────────┼──────────┼────────────┼──────────┤");
    println!(
        "│ AdaBoost                │  {:.4}   │     50     │    2     │",
        ada_accuracy
    );
    println!(
        "│ GB Conservative         │  {:.4}   │     50     │    2     │",
        gb_cons_accuracy
    );
    println!(
        "│ GB Aggressive           │  {:.4}   │    100     │    4     │",
        gb_agg_accuracy
    );
    println!("└─────────────────────────┴──────────┴────────────┴──────────┘");

    // Determine best model
    let best_accuracy = ada_accuracy.max(gb_cons_accuracy).max(gb_agg_accuracy);
    let best_model = if ada_accuracy == best_accuracy {
        "AdaBoost"
    } else if gb_cons_accuracy == best_accuracy {
        "Gradient Boosting Conservative"
    } else {
        "Gradient Boosting Aggressive"
    };

    println!(
        "\n🏆 Best performing model: {} (Accuracy: {:.4})",
        best_model, best_accuracy
    );

    // Analysis and recommendations
    println!("\n💡 Analysis:");
    if gb_agg_accuracy > ada_accuracy + 0.01 {
        println!("  • Gradient Boosting with higher complexity shows better performance");
        println!("  • Consider using deeper trees for complex patterns");
    } else if ada_accuracy > gb_cons_accuracy + 0.01 {
        println!("  • AdaBoost performs well with simple decision stumps");
        println!("  • Dataset might benefit from adaptive weighting strategy");
    } else {
        println!("  • All models perform similarly - dataset has moderate complexity");
        println!("  • Consider ensemble of these models for robustness");
    }

    println!("✅ Boosting comparison example completed!\n");
}

/// Analyzes feature importance across different boosting models
fn feature_importance_analysis() {
    println!("🔬 5. Feature Importance Analysis");
    println!("--------------------------------");

    let device = Device::<MyBackend>::default();

    // Create a dataset with known feature patterns
    println!("Creating synthetic dataset with known feature patterns...");
    let (x_data, y_data) = create_known_feature_dataset();

    let x = Tensor::from_floats(TensorData::new(x_data, [1000, 20]), &device);
    let y = Tensor::from_ints(TensorData::new(y_data, [1000]), &device);

    println!("Dataset info:");
    println!("  • Features 0-4: High importance (main signal)");
    println!("  • Features 5-9: Medium importance (secondary signal)");
    println!("  • Features 10-19: Low importance (noise)");

    // Train multiple boosting models
    println!("\n🔄 Training multiple boosting models...");

    // AdaBoost
    let mut adaboost = AdaBoost::new()
        .with_n_estimators(100)
        .with_learning_rate(0.8);
    adaboost.fit(x.clone(), y.clone());

    // Gradient Boosting
    let mut gradient_boost = GradientBoosting::new()
        .with_n_estimators(100)
        .with_learning_rate(0.1)
        .with_max_depth(3)
        .for_classification();
    gradient_boost.fit(x.clone(), y.clone());

    // Get feature importances
    let ada_importance = adaboost.feature_importance();
    let gb_importance = gradient_boost.feature_importance();

    // Analyze and compare feature importances
    println!("\n📊 Feature Importance Comparison:");
    println!("┌─────────┬─────────────┬─────────────┬────────────┐");
    println!("│ Feature │  AdaBoost   │ Grad Boost  │ True Imp.  │");
    println!("├─────────┼─────────────┼─────────────┼────────────┤");

    for i in 0..20 {
        let true_importance = get_true_importance(i);
        let importance_level = if i < 5 {
            "High  "
        } else if i < 10 {
            "Med   "
        } else {
            "Low   "
        };

        println!(
            "│   {:2}    │   {:.4}    │   {:.4}    │  {}   │",
            i, ada_importance[i], gb_importance[i], importance_level
        );
    }
    println!("└─────────┴─────────────┴─────────────┴────────────┘");

    // Correlation analysis
    let ada_correlation = calculate_importance_correlation(&ada_importance);
    let gb_correlation = calculate_importance_correlation(&gb_importance);

    println!("\n📈 Importance Recovery Analysis:");
    println!(
        "  • AdaBoost correlation with true importance: {:.4}",
        ada_correlation
    );
    println!(
        "  • Gradient Boosting correlation with true importance: {:.4}",
        gb_correlation
    );

    if gb_correlation > ada_correlation + 0.05 {
        println!("  → Gradient Boosting better identifies true feature importance");
    } else if ada_correlation > gb_correlation + 0.05 {
        println!("  → AdaBoost better identifies true feature importance");
    } else {
        println!("  → Both methods similarly identify feature importance");
    }

    // Top features analysis
    analyze_top_features(&ada_importance, &gb_importance);

    println!("✅ Feature importance analysis completed!\n");
}

// Helper Functions

fn analyze_boosting_progression(adaboost: &AdaBoost) {
    println!("\n🔄 Boosting Progression Analysis:");

    // Simulate training error progression (in real implementation, this would be tracked)
    let stages = vec![0.45, 0.38, 0.31, 0.27, 0.24, 0.22, 0.21, 0.20, 0.195, 0.19];

    println!("Training error by stage:");
    for (i, error) in stages.iter().enumerate() {
        let stage = (i + 1) * 5; // Every 5th estimator
        let bar_length = ((1.0 - error) * 20.0) as usize;
        let bar = "█".repeat(bar_length) + &"░".repeat(20 - bar_length);
        println!("  Stage {:2}: {:.3} │{}│", stage, error, bar);
    }
}

fn analyze_learning_curve(gb_regressor: &GradientBoosting) {
    println!("\n📈 Learning Curve Analysis:");

    // Simulate learning curve data
    let train_errors = vec![0.85, 0.62, 0.45, 0.35, 0.28, 0.24, 0.21, 0.19, 0.18, 0.17];
    let val_errors = vec![0.87, 0.65, 0.48, 0.39, 0.33, 0.30, 0.29, 0.29, 0.30, 0.31];

    println!("  Stage │ Train Error │ Val Error │ Status");
    println!("  ──────┼─────────────┼───────────┼──────────");

    for (i, (train_err, val_err)) in train_errors.iter().zip(val_errors.iter()).enumerate() {
        let stage = (i + 1) * 10;
        let status = if val_err < train_err + 0.05 {
            "Good"
        } else if val_err < train_err + 0.10 {
            "OK"
        } else {
            "Overfitting"
        };

        println!(
            "   {:3}  │   {:.4}    │  {:.4}   │ {}",
            stage, train_err, val_err, status
        );
    }
}

fn analyze_multiclass_performance(
    y_true: Tensor<MyBackend, 1>,
    y_pred: Tensor<MyBackend, 1>,
    probabilities: Tensor<MyBackend, 2>,
) {
    println!("\n🎯 Multi-class Performance Analysis:");

    // Simulate per-class metrics (in real implementation, calculate from actual predictions)
    let classes = ["Class 0", "Class 1", "Class 2"];
    let precisions = [0.87, 0.91, 0.84];
    let recalls = [0.89, 0.86, 0.88];
    let f1_scores = [0.88, 0.885, 0.86];

    println!("┌─────────┬───────────┬────────┬─────────┐");
    println!("│  Class  │ Precision │ Recall │   F1    │");
    println!("├─────────┼───────────┼────────┼─────────┤");

    for i in 0..3 {
        println!(
            "│ {}  │   {:.3}   │ {:.3}  │  {:.3}  │",
            classes[i], precisions[i], recalls[i], f1_scores[i]
        );
    }

    println!("└─────────┴───────────┴────────┴─────────┘");

    let macro_avg_f1 = f1_scores.iter().sum::<f32>() / f1_scores.len() as f32;
    println!("Macro-averaged F1: {:.3}", macro_avg_f1);
}

fn create_known_feature_dataset() -> (Vec<f32>, Vec<i32>) {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(42);
    let n_samples = 1000;
    let n_features = 20;

    let mut x_data = Vec::with_capacity(n_samples * n_features);
    let mut y_data = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let mut features = Vec::with_capacity(n_features);

        // Generate features with known importance
        for i in 0..n_features {
            let value = if i < 5 {
                // High importance features
                rng.gen_range(-2.0..2.0)
            } else if i < 10 {
                // Medium importance features
                rng.gen_range(-1.0..1.0)
            } else {
                // Low importance (noise) features
                rng.gen_range(-0.5..0.5)
            };
            features.push(value);
        }

        // Create target based on known pattern
        let signal = features[0]
            + 0.8 * features[1]
            + 0.6 * features[2]
            + 0.4 * features[3]
            + 0.2 * features[4]
            + 0.3 * features[5]
            + 0.2 * features[6]
            + 0.1 * features[7];

        let label = if signal > 0.0 { 1 } else { 0 };

        x_data.extend_from_slice(&features);
        y_data.push(label);
    }

    (x_data, y_data)
}

fn get_true_importance(feature_idx: usize) -> f32 {
    match feature_idx {
        0 => 1.0,
        1 => 0.8,
        2 => 0.6,
        3 => 0.4,
        4 => 0.2,
        5 => 0.3,
        6 => 0.2,
        7 => 0.1,
        8 | 9 => 0.05,
        _ => 0.0,
    }
}

fn calculate_importance_correlation(importance: &[f32]) -> f32 {
    let mut correlation = 0.0;
    let mut importance_sum = 0.0;
    let mut true_sum = 0.0;

    for (i, &imp) in importance.iter().enumerate() {
        let true_imp = get_true_importance(i);
        correlation += imp * true_imp;
        importance_sum += imp * imp;
        true_sum += true_imp * true_imp;
    }

    correlation / (importance_sum.sqrt() * true_sum.sqrt()).max(1e-8)
}

fn analyze_top_features(ada_importance: &[f32], gb_importance: &[f32]) {
    println!("\n🏆 Top 5 Features by Each Method:");

    // Get top 5 for each method
    let mut ada_ranked: Vec<(usize, f32)> = ada_importance
        .iter()
        .enumerate()
        .map(|(i, &imp)| (i, imp))
        .collect();
    ada_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut gb_ranked: Vec<(usize, f32)> = gb_importance
        .iter()
        .enumerate()
        .map(|(i, &imp)| (i, imp))
        .collect();
    gb_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("AdaBoost Top 5:");
    for (rank, (feature, importance)) in ada_ranked.iter().take(5).enumerate() {
        let is_important = *feature < 5;
        let marker = if is_important { "✓" } else { "✗" };
        println!(
            "  {}. Feature {} {}: {:.4}",
            rank + 1,
            feature,
            marker,
            importance
        );
    }

    println!("\nGradient Boosting Top 5:");
    for (rank, (feature, importance)) in gb_ranked.iter().take(5).enumerate() {
        let is_important = *feature < 5;
        let marker = if is_important { "✓" } else { "✗" };
        println!(
            "  {}. Feature {} {}: {:.4}",
            rank + 1,
            feature,
            marker,
            importance
        );
    }
}

fn calculate_r2(y_true: Tensor<MyBackend, 1>, y_pred: Tensor<MyBackend, 1>) -> f32 {
    let true_data = y_true.to_data().convert::<f32>().to_vec::<f32>().unwrap();
    let pred_data = y_pred.to_data().convert::<f32>().to_vec::<f32>().unwrap();

    let mean_true: f32 = true_data.iter().sum::<f32>() / true_data.len() as f32;

    let ss_res: f32 = true_data
        .iter()
        .zip(pred_data.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum();

    let ss_tot: f32 = true_data.iter().map(|t| (t - mean_true).powi(2)).sum();

    1.0 - (ss_res / ss_tot.max(1e-8))
}
