//! Precomputed kernel for custom kernel matrices

/// Precomputed kernel for custom kernel matrices
#[derive(Debug, Clone)]
pub struct PrecomputedKernel {
    /// Pre-calculated kernel matrix values
    #[allow(dead_code)]
    kernel_matrix: Vec<Vec<f32>>,
}

impl PrecomputedKernel {
    /// Create a new precomputed kernel from a kernel matrix
    pub fn new(kernel_matrix: Vec<Vec<f32>>) -> Self {
        Self { kernel_matrix }
    }
}
