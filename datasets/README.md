# Datasets

This folder contains CSV datasets used in the CS3780 Burn examples.

| File | Description | Features | Samples | Task |
|------|-------------|----------|---------|------|
| `iris.csv` | Fisher's Iris dataset | sepal_length, sepal_width, petal_length, petal_width | 150 | Multi-class classification (3 species) |
| `xor.csv` | XOR problem dataset | x1, x2 | 20 | Binary classification (non-linearly separable) |
| `linear_regression.csv` | Simple linear regression toy dataset | x | 20 | Regression (y ≈ 2x) |

## Loading datasets

Use the `csv` crate to read these files, or use the synthetic generators in `src/datasets.rs`.

### Iris class labels
- `0` = Iris setosa
- `1` = Iris versicolor
- `2` = Iris virginica
