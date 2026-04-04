use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "burn-cs3780")]
#[command(about = "A comprehensive machine learning library implementing CS3780 concepts using Burn")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run k-Nearest Neighbors example
    Knn,
    /// Run Decision Trees example
    Trees,
    /// Run Linear Regression example
    LinearRegression,
    /// Run Logistic Regression example
    LogisticRegression,
    /// Run Perceptron example
    Perceptron,
    /// Run SVM example
    Svm,
    /// Run Kernels example
    Kernels,
    /// Run Neural Networks example
    NeuralNets,
    /// Run Transformers example
    Transformers,
    /// Run Autoencoders example
    Autoencoders,
    /// Run Boosting example
    Boosting,
    /// Run Clustering example
    Clustering,
    /// Run PCA example
    Pca,
    /// Run Online Learning example
    OnlineLearning,
    /// Run Optimization example
    Optimization,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    println!("Burn CS3780: Machine Learning Framework");
    println!("==========================================");

    match cli.command {
        Commands::Knn => println!("Running k-Nearest Neighbors example..."),
        Commands::Trees => println!("Running Decision Trees example..."),
        Commands::LinearRegression => println!("Running Linear Regression example..."),
        Commands::LogisticRegression => println!("Running Logistic Regression example..."),
        Commands::Perceptron => println!("Running Perceptron example..."),
        Commands::Svm => println!("Running SVM example..."),
        Commands::Kernels => println!("Running Kernels example..."),
        Commands::NeuralNets => println!("Running Neural Networks example..."),
        Commands::Transformers => println!("Running Transformers example..."),
        Commands::Autoencoders => println!("Running Autoencoders example..."),
        Commands::Boosting => println!("Running Boosting example..."),
        Commands::Clustering => println!("Running Clustering example..."),
        Commands::Pca => println!("Running PCA example..."),
        Commands::OnlineLearning => println!("Running Online Learning example..."),
        Commands::Optimization => println!("Running Optimization example..."),
    }

    println!("Example completed successfully!");
    Ok(())
}
