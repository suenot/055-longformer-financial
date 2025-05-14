//! Attention mechanisms for Longformer
//!
//! This module implements the two key attention patterns:
//! - Sliding window attention: O(n) local attention
//! - Global attention: Full attention for special tokens

mod sliding_window;
mod global;

pub use sliding_window::SlidingWindowAttention;
pub use global::GlobalAttention;

use ndarray::{Array2, Array3};

/// Configuration for attention mechanisms
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Size of sliding window (one side)
    pub window_size: usize,
    /// Dropout probability
    pub dropout: f64,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            d_model: 256,
            n_heads: 8,
            window_size: 256,
            dropout: 0.1,
        }
    }
}

/// Trait for attention mechanisms
pub trait Attention {
    /// Compute attention scores and output
    ///
    /// # Arguments
    /// * `query` - Query tensor of shape (batch, seq_len, d_model)
    /// * `key` - Key tensor of shape (batch, seq_len, d_model)
    /// * `value` - Value tensor of shape (batch, seq_len, d_model)
    /// * `mask` - Optional attention mask
    ///
    /// # Returns
    /// Attention output of shape (batch, seq_len, d_model)
    fn forward(
        &self,
        query: &Array3<f64>,
        key: &Array3<f64>,
        value: &Array3<f64>,
        mask: Option<&Array2<bool>>,
    ) -> Array3<f64>;
}

/// Softmax along the last axis
pub fn softmax_last_axis(x: &Array2<f64>) -> Array2<f64> {
    let max_vals = x.map_axis(ndarray::Axis(1), |row| {
        row.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    });

    let exp_x = x.clone() - &max_vals.insert_axis(ndarray::Axis(1));
    let exp_x = exp_x.mapv(f64::exp);

    let sum_exp = exp_x.sum_axis(ndarray::Axis(1));
    exp_x / &sum_exp.insert_axis(ndarray::Axis(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_softmax() {
        let x = array![[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]];
        let result = softmax_last_axis(&x);

        // Check that each row sums to 1
        for row in result.rows() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        // Check that larger values have larger probabilities
        assert!(result[[0, 2]] > result[[0, 1]]);
        assert!(result[[0, 1]] > result[[0, 0]]);
    }
}
