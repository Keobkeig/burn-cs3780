use burn_cs3780::{
    datasets,
    metrics::{ClassificationMetrics, CrossValidation},
    models::KNearestNeighbors,
    utils::Visualization,
    DefaultBackend,
};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    println!("🔥 Burn CS3780: k-Nearest Neighbors Example");
    println!("============================================");

    let device = Default::default();

    // Generate datasets
    println!("\n📊 Generating datasets...");

    // 1. Linearly separable data
    let linear_dataset =
        datasets::make_linearly_separable::<DefaultBackend>(200, &device, Some(42));
    println!(
        "✅ Generated linearly separable dataset: {} samples",
        linear_dataset.num_samples()
    );

    // 2. XOR data (non-linearly separable)
    let xor_dataset = datasets::make_xor_dataset::<DefaultBackend>(200, 0.1, &device, Some(42));
    println!(
        "✅ Generated XOR dataset: {} samples",
        xor_dataset.num_samples()
    );

    // Split datasets
    let (linear_train, linear_test) = linear_dataset.train_test_split(0.8, Some(42));
    let (xor_train, xor_test) = xor_dataset.train_test_split(0.8, Some(42));

    println!("\n🔬 Training and Testing k-NN Classifiers");
    println!("==========================================");

    // Test different k values on linear data
    println!("\n📈 Linear Dataset Results:");
    test_knn_on_dataset(&linear_train, &linear_test, "Linear")?;

    // Test different k values on XOR data
    println!("\n📈 XOR Dataset Results:");
    test_knn_on_dataset(&xor_train, &xor_test, "XOR")?;

    // Find optimal k using cross-validation
    println!("\n🎯 Finding Optimal k using Cross-Validation");
    println!("=============================================");

    let optimal_k_linear = find_optimal_k(&linear_dataset, "Linear")?;
    let optimal_k_xor = find_optimal_k(&xor_dataset, "XOR")?;

    println!("\n🏆 Final Results:");
    println!("Linear data - Optimal k: {}", optimal_k_linear);
    println!("XOR data - Optimal k: {}", optimal_k_xor);

    // Demonstrate different distance metrics
    println!("\n📏 Comparing Distance Metrics");
    println!("==============================");
    test_distance_metrics(&linear_train, &linear_test)?;

    // Demonstrate weight functions
    println!("\n⚖️  Comparing Weight Functions");
    println!("===============================");
    test_weight_functions(&linear_train, &linear_test)?;

    println!("\n✅ k-NN example completed successfully!");

    Ok(())
}

fn test_knn_on_dataset(
    train_data: &datasets::Dataset<DefaultBackend>,
    test_data: &datasets::Dataset<DefaultBackend>,
    dataset_name: &str,
) -> anyhow::Result<()> {
    let k_values = [1, 3, 5, 7, 9, 15];

    println!("k\tAccuracy\tPrecision\tRecall\t\tF1-Score");
    println!("─────────────────────────────────────────────────────");

    for k in k_values {
        let mut knn = KNearestNeighbors::new(k);
        knn.fit(
            train_data.features.clone(),
            train_data.labels.clone().squeeze::<1>(),
        );

        let predictions = knn.predict(&test_data.features);
        let y_true = test_data.labels.clone().squeeze::<1>();

        let accuracy = ClassificationMetrics::accuracy(&y_true, &predictions);
        let precision = ClassificationMetrics::precision(&y_true, &predictions);
        let recall = ClassificationMetrics::recall(&y_true, &predictions);
        let f1 = ClassificationMetrics::f1_score(&y_true, &predictions);

        println!(
            "{}\t{:.4}\t\t{:.4}\t\t{:.4}\t\t{:.4}",
            k, accuracy, precision, recall, f1
        );
    }

    Ok(())
}

fn find_optimal_k(
    dataset: &datasets::Dataset<DefaultBackend>,
    dataset_name: &str,
) -> anyhow::Result<usize> {
    use burn_cs3780::models::KNNUtils;

    println!("Finding optimal k for {} dataset...", dataset_name);

    let (optimal_k, scores) = KNNUtils::find_optimal_k(
        &dataset.features,
        &dataset.labels.clone().squeeze::<1>(),
        1..20,
        5,    // 5-fold CV
        true, // classification
    );

    println!("Cross-validation scores by k:");
    println!("k\tCV Score");
    println!("─────────────");
    for (i, score) in scores.iter().enumerate() {
        let k = i + 1;
        if k == optimal_k {
            println!("{}*\t{:.4} ← Optimal", k, score);
        } else {
            println!("{}\t{:.4}", k, score);
        }
    }

    Ok(optimal_k)
}

fn test_distance_metrics(
    train_data: &datasets::Dataset<DefaultBackend>,
    test_data: &datasets::Dataset<DefaultBackend>,
) -> anyhow::Result<()> {
    use burn_cs3780::models::{DistanceMetric, WeightFunction};

    let metrics = [
        (DistanceMetric::Euclidean, "Euclidean"),
        (DistanceMetric::Manhattan, "Manhattan"),
        (DistanceMetric::Cosine, "Cosine"),
    ];

    println!("Distance Metric\tAccuracy");
    println!("─────────────────────────");

    for (metric, name) in metrics {
        let mut knn = KNearestNeighbors::new(5).with_distance_metric(metric);

        knn.fit(
            train_data.features.clone(),
            train_data.labels.clone().squeeze::<1>(),
        );
        let predictions = knn.predict(&test_data.features);
        let accuracy =
            ClassificationMetrics::accuracy(&test_data.labels.clone().squeeze::<1>(), &predictions);

        println!("{}\t\t{:.4}", name, accuracy);
    }

    Ok(())
}

fn test_weight_functions(
    train_data: &datasets::Dataset<DefaultBackend>,
    test_data: &datasets::Dataset<DefaultBackend>,
) -> anyhow::Result<()> {
    use burn_cs3780::models::WeightFunction;

    let weight_functions = [
        (WeightFunction::Uniform, "Uniform"),
        (WeightFunction::Distance, "Distance-weighted"),
        (WeightFunction::Exponential, "Exponential"),
    ];

    println!("Weight Function\t\tAccuracy");
    println!("─────────────────────────────");

    for (weight_fn, name) in weight_functions {
        let mut knn = KNearestNeighbors::new(5).with_weights(weight_fn);

        knn.fit(
            train_data.features.clone(),
            train_data.labels.clone().squeeze::<1>(),
        );
        let predictions = knn.predict(&test_data.features);
        let accuracy =
            ClassificationMetrics::accuracy(&test_data.labels.clone().squeeze::<1>(), &predictions);

        println!("{}\t{:.4}", name, accuracy);
    }

    Ok(())
}
