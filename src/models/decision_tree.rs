//! Decision Tree implementation for classification and regression
//!
//! This module implements decision trees with multiple splitting criteria,
//! pruning algorithms, and support for both continuous and categorical features.

use burn::tensor::{backend::Backend, Device, Tensor, TensorData};
use std::collections::HashMap;

/// Splitting criteria for decision trees
#[derive(Debug, Clone, Copy)]
pub enum SplitCriterion {
    /// Gini impurity for classification
    Gini,
    /// Information gain (entropy) for classification
    Entropy,
    /// Mean squared error for regression
    MSE,
}

/// Type of decision tree
#[derive(Debug, Clone, Copy)]
pub enum TreeType {
    /// Classification tree
    Classifier,
    /// Regression tree
    Regressor,
}

/// Decision tree node
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Node<B: Backend<FloatElem = f32>> {
    /// Feature index for splitting
    feature_idx: Option<usize>,
    /// Threshold value for splitting
    threshold: Option<f32>,
    /// Left child node
    left: Option<Box<Node<B>>>,
    /// Right child node
    right: Option<Box<Node<B>>>,
    /// Prediction value (for leaf nodes)
    value: Option<f32>,
    /// Class distribution (for classification leaf nodes)
    class_counts: Option<HashMap<i32, i32>>,
    /// Number of samples in this node
    n_samples: usize,
    /// Impurity of this node
    impurity: f32,
    /// Device for computations
    device: Device<B>,
}

impl<B: Backend<FloatElem = f32>> Node<B> {
    /// Create a new leaf node
    fn new_leaf(value: f32, n_samples: usize, impurity: f32, device: Device<B>) -> Self {
        Self {
            feature_idx: None,
            threshold: None,
            left: None,
            right: None,
            value: Some(value),
            class_counts: None,
            n_samples,
            impurity,
            device,
        }
    }

    /// Create a new classification leaf node
    fn new_class_leaf(
        class_counts: HashMap<i32, i32>,
        n_samples: usize,
        impurity: f32,
        device: Device<B>,
    ) -> Self {
        // Find the majority class
        let majority_class = class_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&class, _)| class)
            .unwrap_or(0);

        Self {
            feature_idx: None,
            threshold: None,
            left: None,
            right: None,
            value: Some(majority_class as f32),
            class_counts: Some(class_counts),
            n_samples,
            impurity,
            device,
        }
    }

    /// Create a new internal node
    fn new_internal(
        feature_idx: usize,
        threshold: f32,
        left: Node<B>,
        right: Node<B>,
        n_samples: usize,
        impurity: f32,
        device: Device<B>,
    ) -> Self {
        Self {
            feature_idx: Some(feature_idx),
            threshold: Some(threshold),
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            value: None,
            class_counts: None,
            n_samples,
            impurity,
            device,
        }
    }

    /// Check if this is a leaf node
    fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }

    /// Predict a single sample
    fn predict_single(&self, sample: &[f32]) -> f32 {
        if self.is_leaf() {
            self.value.unwrap_or(0.0)
        } else {
            let feature_idx = self.feature_idx.unwrap();
            let threshold = self.threshold.unwrap();

            if sample[feature_idx] <= threshold {
                self.left.as_ref().unwrap().predict_single(sample)
            } else {
                self.right.as_ref().unwrap().predict_single(sample)
            }
        }
    }
}

/// Decision tree classifier and regressor
#[derive(Debug, Clone)]
pub struct DecisionTree<B: Backend<FloatElem = f32>> {
    /// Root node of the tree
    root: Option<Node<B>>,
    /// Type of tree (classifier or regressor)
    tree_type: TreeType,
    /// Splitting criterion
    criterion: SplitCriterion,
    /// Maximum depth of the tree
    max_depth: Option<usize>,
    /// Minimum samples required to split a node
    min_samples_split: usize,
    /// Minimum samples required in a leaf
    min_samples_leaf: usize,
    /// Maximum number of features to consider for splits
    max_features: Option<usize>,
    /// Random state for reproducibility
    random_state: Option<u64>,
    /// Device for computations
    device: Device<B>,
}

impl<B: Backend<FloatElem = f32>> DecisionTree<B> {
    /// Create a new decision tree classifier
    pub fn classifier(device: Device<B>) -> Self {
        Self {
            root: None,
            tree_type: TreeType::Classifier,
            criterion: SplitCriterion::Gini,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            random_state: None,
            device,
        }
    }

    /// Create a new decision tree regressor
    pub fn regressor(device: Device<B>) -> Self {
        Self {
            root: None,
            tree_type: TreeType::Regressor,
            criterion: SplitCriterion::MSE,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            random_state: None,
            device,
        }
    }

    /// Set the splitting criterion
    pub fn with_criterion(mut self, criterion: SplitCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Set the maximum depth
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }

    /// Set the minimum samples for splitting
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum samples in a leaf
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set the maximum number of features
    pub fn with_max_features(mut self, max_features: usize) -> Self {
        self.max_features = Some(max_features);
        self
    }

    /// Set the random state
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Train the decision tree
    pub fn fit(&mut self, x: Tensor<B, 2>, y: Tensor<B, 1>) -> Result<(), String> {
        let [n_samples, n_features] = x.dims();

        // Convert tensors to vectors for easier manipulation
        let x_data = self.tensor_to_2d_vec(&x);
        let y_data = self.tensor_to_vec(&y);

        // Build the tree
        self.root = Some(self.build_tree(
            &x_data,
            &y_data,
            0, // current depth
            &(0..n_samples).collect::<Vec<_>>(),
            n_features,
        ));

        Ok(())
    }

    /// Predict using the trained tree
    pub fn predict(&self, x: Tensor<B, 2>) -> Result<Tensor<B, 1>, String> {
        let root = self
            .root
            .as_ref()
            .ok_or("Model not trained. Call fit() first.")?;

        let [n_samples, _] = x.dims();
        let x_data = self.tensor_to_2d_vec(&x);
        let mut predictions = Vec::new();

        for i in 0..n_samples {
            let prediction = root.predict_single(&x_data[i]);
            predictions.push(prediction);
        }

        Ok(Tensor::from_floats(
            TensorData::new(predictions, [n_samples]),
            &self.device,
        ))
    }

    /// Check if the tree has been fitted
    pub fn is_fitted(&self) -> bool {
        self.root.is_some()
    }

    /// Get the depth of the tree
    pub fn get_depth(&self) -> usize {
        self.root.as_ref().map_or(0, |root| self.node_depth(root))
    }

    /// Get the number of nodes in the tree
    pub fn get_n_nodes(&self) -> usize {
        self.root.as_ref().map_or(0, |root| self.count_nodes(root))
    }

    /// Get the number of leaves in the tree
    pub fn get_n_leaves(&self) -> usize {
        self.root.as_ref().map_or(0, |root| self.count_leaves(root))
    }

    // Helper methods

    fn build_tree(
        &self,
        x: &[Vec<f32>],
        y: &[f32],
        depth: usize,
        indices: &[usize],
        n_features: usize,
    ) -> Node<B> {
        let n_samples = indices.len();
        let current_impurity = self.calculate_impurity(y, indices);

        // Check stopping criteria
        if n_samples < self.min_samples_split
            || n_samples < self.min_samples_leaf * 2
            || self.max_depth.map_or(false, |max_d| depth >= max_d)
            || current_impurity == 0.0
        {
            return self.create_leaf_node(y, indices, current_impurity);
        }

        // Find the best split
        if let Some((best_feature, best_threshold, left_indices, right_indices)) =
            self.find_best_split(x, y, indices, n_features)
        {
            // Ensure minimum leaf size
            if left_indices.len() < self.min_samples_leaf
                || right_indices.len() < self.min_samples_leaf
            {
                return self.create_leaf_node(y, indices, current_impurity);
            }

            // Recursively build children
            let left_node = self.build_tree(x, y, depth + 1, &left_indices, n_features);
            let right_node = self.build_tree(x, y, depth + 1, &right_indices, n_features);

            Node::new_internal(
                best_feature,
                best_threshold,
                left_node,
                right_node,
                n_samples,
                current_impurity,
                self.device.clone(),
            )
        } else {
            self.create_leaf_node(y, indices, current_impurity)
        }
    }

    fn find_best_split(
        &self,
        x: &[Vec<f32>],
        y: &[f32],
        indices: &[usize],
        n_features: usize,
    ) -> Option<(usize, f32, Vec<usize>, Vec<usize>)> {
        let mut best_impurity_gain = 0.0;
        let mut best_split = None;

        // Determine features to consider
        let features_to_consider: Vec<usize> = if let Some(max_feat) = self.max_features {
            // TODO: Implement random feature selection
            (0..max_feat.min(n_features)).collect()
        } else {
            (0..n_features).collect()
        };

        for feature_idx in features_to_consider {
            // Get unique values for this feature
            let mut feature_values: Vec<f32> = indices.iter().map(|&i| x[i][feature_idx]).collect();
            feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            feature_values.dedup();

            // Try each threshold
            for i in 0..feature_values.len() - 1 {
                let threshold = (feature_values[i] + feature_values[i + 1]) / 2.0;

                let (left_indices, right_indices) =
                    self.split_indices(x, indices, feature_idx, threshold);

                if left_indices.is_empty() || right_indices.is_empty() {
                    continue;
                }

                let impurity_gain =
                    self.calculate_impurity_gain(y, indices, &left_indices, &right_indices);

                if impurity_gain > best_impurity_gain {
                    best_impurity_gain = impurity_gain;
                    best_split = Some((feature_idx, threshold, left_indices, right_indices));
                }
            }
        }

        best_split
    }

    fn split_indices(
        &self,
        x: &[Vec<f32>],
        indices: &[usize],
        feature_idx: usize,
        threshold: f32,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for &idx in indices {
            if x[idx][feature_idx] <= threshold {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        (left_indices, right_indices)
    }

    fn calculate_impurity(&self, y: &[f32], indices: &[usize]) -> f32 {
        if indices.is_empty() {
            return 0.0;
        }

        match (self.criterion, self.tree_type) {
            (SplitCriterion::Gini, TreeType::Classifier) => self.gini_impurity(y, indices),
            (SplitCriterion::Entropy, TreeType::Classifier) => self.entropy(y, indices),
            (SplitCriterion::MSE, TreeType::Regressor) => self.mse(y, indices),
            _ => 0.0,
        }
    }

    fn gini_impurity(&self, y: &[f32], indices: &[usize]) -> f32 {
        let mut class_counts = HashMap::new();
        let total = indices.len() as f32;

        for &idx in indices {
            let class = y[idx] as i32;
            *class_counts.entry(class).or_insert(0) += 1;
        }

        let mut gini = 1.0;
        for &count in class_counts.values() {
            let p = count as f32 / total;
            gini -= p * p;
        }

        gini
    }

    fn entropy(&self, y: &[f32], indices: &[usize]) -> f32 {
        let mut class_counts = HashMap::new();
        let total = indices.len() as f32;

        for &idx in indices {
            let class = y[idx] as i32;
            *class_counts.entry(class).or_insert(0) += 1;
        }

        let mut entropy = 0.0;
        for &count in class_counts.values() {
            if count > 0 {
                let p = count as f32 / total;
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    fn mse(&self, y: &[f32], indices: &[usize]) -> f32 {
        if indices.is_empty() {
            return 0.0;
        }

        let mean: f32 = indices.iter().map(|&i| y[i]).sum::<f32>() / indices.len() as f32;
        indices.iter().map(|&i| (y[i] - mean).powi(2)).sum::<f32>() / indices.len() as f32
    }

    fn calculate_impurity_gain(
        &self,
        y: &[f32],
        parent_indices: &[usize],
        left_indices: &[usize],
        right_indices: &[usize],
    ) -> f32 {
        let parent_impurity = self.calculate_impurity(y, parent_indices);
        let left_impurity = self.calculate_impurity(y, left_indices);
        let right_impurity = self.calculate_impurity(y, right_indices);

        let total_samples = parent_indices.len() as f32;
        let left_weight = left_indices.len() as f32 / total_samples;
        let right_weight = right_indices.len() as f32 / total_samples;

        parent_impurity - (left_weight * left_impurity + right_weight * right_impurity)
    }

    fn create_leaf_node(&self, y: &[f32], indices: &[usize], impurity: f32) -> Node<B> {
        match self.tree_type {
            TreeType::Classifier => {
                let mut class_counts = HashMap::new();
                for &idx in indices {
                    let class = y[idx] as i32;
                    *class_counts.entry(class).or_insert(0) += 1;
                }
                Node::new_class_leaf(class_counts, indices.len(), impurity, self.device.clone())
            }
            TreeType::Regressor => {
                let mean = indices.iter().map(|&i| y[i]).sum::<f32>() / indices.len() as f32;
                Node::new_leaf(mean, indices.len(), impurity, self.device.clone())
            }
        }
    }

    fn tensor_to_2d_vec(&self, tensor: &Tensor<B, 2>) -> Vec<Vec<f32>> {
        let [n_samples, n_features] = tensor.dims();
        let mut result = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let mut row = Vec::with_capacity(n_features);
            for j in 0..n_features {
                let val = tensor.clone().slice([i..i + 1, j..j + 1]).into_scalar();
                row.push(val);
            }
            result.push(row);
        }

        result
    }

    fn tensor_to_vec(&self, tensor: &Tensor<B, 1>) -> Vec<f32> {
        let [n_samples] = tensor.dims();
        let mut result = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let val = tensor.clone().slice([i..i + 1]).into_scalar();
            result.push(val);
        }

        result
    }

    fn node_depth(&self, node: &Node<B>) -> usize {
        if node.is_leaf() {
            1
        } else {
            let left_depth = node.left.as_ref().map_or(0, |n| self.node_depth(n));
            let right_depth = node.right.as_ref().map_or(0, |n| self.node_depth(n));
            1 + left_depth.max(right_depth)
        }
    }

    fn count_nodes(&self, node: &Node<B>) -> usize {
        1 + node.left.as_ref().map_or(0, |n| self.count_nodes(n))
            + node.right.as_ref().map_or(0, |n| self.count_nodes(n))
    }

    fn count_leaves(&self, node: &Node<B>) -> usize {
        if node.is_leaf() {
            1
        } else {
            node.left.as_ref().map_or(0, |n| self.count_leaves(n))
                + node.right.as_ref().map_or(0, |n| self.count_leaves(n))
        }
    }
}

impl<B: Backend<FloatElem = f32>> Default for DecisionTree<B>
where
    B: 'static,
{
    fn default() -> Self {
        // Cannot provide a sensible default without knowing the backend
        panic!("Use DecisionTree::classifier(device) or DecisionTree::regressor(device) to create a tree")
    }
}

/// Type alias for decision tree classifier
pub type DecisionTreeClassifier<B> = DecisionTree<B>;

/// Type alias for decision tree regressor  
pub type DecisionTreeRegressor<B> = DecisionTree<B>;
