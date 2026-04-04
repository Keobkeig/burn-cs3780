//! Classification evaluation metrics

use burn::tensor::{backend::Backend, Tensor};

/// Classification metrics
pub struct ClassificationMetrics;

impl ClassificationMetrics {
    /// Calculate accuracy (fraction of correct predictions)
    pub fn accuracy<B: Backend<FloatElem = f32>>(
        y_true: &Tensor<B, 1>,
        y_pred: &Tensor<B, 1>,
    ) -> f32 {
        let correct = y_true.clone().equal(y_pred.clone()).float();
        let total = correct.dims()[0] as f32;
        let sum_correct: f32 = correct.sum().into_scalar();
        sum_correct / total
    }

    /// Calculate precision for binary classification
    pub fn precision<B: Backend<FloatElem = f32>>(
        y_true: &Tensor<B, 1>,
        y_pred: &Tensor<B, 1>,
    ) -> f32 {
        let tp = Self::true_positives(y_true, y_pred);
        let fp = Self::false_positives(y_true, y_pred);

        if tp + fp == 0.0 {
            0.0
        } else {
            tp / (tp + fp)
        }
    }

    /// Calculate recall for binary classification
    pub fn recall<B: Backend<FloatElem = f32>>(
        y_true: &Tensor<B, 1>,
        y_pred: &Tensor<B, 1>,
    ) -> f32 {
        let tp = Self::true_positives(y_true, y_pred);
        let fn_count = Self::false_negatives(y_true, y_pred);

        if tp + fn_count == 0.0 {
            0.0
        } else {
            tp / (tp + fn_count)
        }
    }

    /// Calculate F1 score for binary classification
    pub fn f1_score<B: Backend<FloatElem = f32>>(
        y_true: &Tensor<B, 1>,
        y_pred: &Tensor<B, 1>,
    ) -> f32 {
        let precision = Self::precision(y_true, y_pred);
        let recall = Self::recall(y_true, y_pred);

        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * (precision * recall) / (precision + recall)
        }
    }

    /// Generate confusion matrix for binary classification
    pub fn confusion_matrix<B: Backend<FloatElem = f32>>(
        y_true: &Tensor<B, 1>,
        y_pred: &Tensor<B, 1>,
    ) -> [[f32; 2]; 2] {
        let tp = Self::true_positives(y_true, y_pred);
        let fp = Self::false_positives(y_true, y_pred);
        let fn_count = Self::false_negatives(y_true, y_pred);
        let tn: f32 = y_true
            .clone()
            .equal_elem(0.0)
            .float()
            .mul(y_pred.clone().equal_elem(0.0).float())
            .sum()
            .into_scalar();

        [
            [tn, fp],       // Row 0: True Negative, False Positive
            [fn_count, tp], // Row 1: False Negative, True Positive
        ]
    }

    fn true_positives<B: Backend<FloatElem = f32>>(
        y_true: &Tensor<B, 1>,
        y_pred: &Tensor<B, 1>,
    ) -> f32 {
        let true_pos = y_true
            .clone()
            .equal_elem(1.0)
            .float()
            .mul(y_pred.clone().equal_elem(1.0).float());
        true_pos.sum().into_scalar()
    }

    fn false_positives<B: Backend<FloatElem = f32>>(
        y_true: &Tensor<B, 1>,
        y_pred: &Tensor<B, 1>,
    ) -> f32 {
        let false_pos = y_true
            .clone()
            .equal_elem(0.0)
            .float()
            .mul(y_pred.clone().equal_elem(1.0).float());
        false_pos.sum().into_scalar()
    }

    fn false_negatives<B: Backend<FloatElem = f32>>(
        y_true: &Tensor<B, 1>,
        y_pred: &Tensor<B, 1>,
    ) -> f32 {
        let false_neg = y_true
            .clone()
            .equal_elem(1.0)
            .float()
            .mul(y_pred.clone().equal_elem(0.0).float());
        false_neg.sum().into_scalar()
    }
}
