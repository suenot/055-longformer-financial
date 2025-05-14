//! Sliding window attention implementation
//!
//! Efficient O(n) attention where each position attends only to
//! a fixed-size window of neighboring positions.

use ndarray::{Array2, Array3, Axis, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rayon::prelude::*;

use super::{Attention, AttentionConfig, softmax_last_axis};

/// Sliding window attention mechanism
///
/// Each position attends to `window_size` positions on each side,
/// giving a total attention span of `2 * window_size + 1`.
pub struct SlidingWindowAttention {
    /// Configuration
    config: AttentionConfig,

    /// Query projection weights (d_model, d_model)
    w_query: Array2<f64>,

    /// Key projection weights (d_model, d_model)
    w_key: Array2<f64>,

    /// Value projection weights (d_model, d_model)
    w_value: Array2<f64>,

    /// Output projection weights (d_model, d_model)
    w_output: Array2<f64>,

    /// Scaling factor for attention scores
    scale: f64,
}

impl SlidingWindowAttention {
    /// Create a new sliding window attention layer
    pub fn new(config: AttentionConfig) -> Self {
        let d_model = config.d_model;
        let scale = 1.0 / ((d_model / config.n_heads) as f64).sqrt();

        // Initialize weights with Xavier/Glorot initialization
        let limit = (6.0 / (d_model + d_model) as f64).sqrt();
        let dist = Uniform::new(-limit, limit);

        Self {
            config,
            w_query: Array2::random((d_model, d_model), dist),
            w_key: Array2::random((d_model, d_model), dist),
            w_value: Array2::random((d_model, d_model), dist),
            w_output: Array2::random((d_model, d_model), dist),
            scale,
        }
    }

    /// Compute sliding window attention for a single batch element
    fn attention_single(
        &self,
        query: &Array2<f64>,
        key: &Array2<f64>,
        value: &Array2<f64>,
        mask: Option<&Array2<bool>>,
    ) -> Array2<f64> {
        let seq_len = query.nrows();
        let d_model = self.config.d_model;
        let n_heads = self.config.n_heads;
        let head_dim = d_model / n_heads;
        let window = self.config.window_size;

        // Project Q, K, V
        let q = query.dot(&self.w_query);
        let k = key.dot(&self.w_key);
        let v = value.dot(&self.w_value);

        // Compute attention for each position
        let mut output = Array2::<f64>::zeros((seq_len, d_model));

        // Process each position
        for i in 0..seq_len {
            // Determine window bounds
            let start = i.saturating_sub(window);
            let end = (i + window + 1).min(seq_len);
            let window_size = end - start;

            // Extract window slices
            let q_i = q.slice(s![i, ..]);
            let k_window = k.slice(s![start..end, ..]);
            let v_window = v.slice(s![start..end, ..]);

            // Compute attention scores for this position
            // scores[j] = q_i · k_window[j] / sqrt(d_k)
            let mut scores = Array2::<f64>::zeros((1, window_size));
            for (j, k_j) in k_window.rows().into_iter().enumerate() {
                let score: f64 = q_i.iter().zip(k_j.iter()).map(|(a, b)| a * b).sum();
                scores[[0, j]] = score * self.scale;
            }

            // Apply mask if provided
            if let Some(m) = mask {
                for j in 0..window_size {
                    if !m[[i, start + j]] {
                        scores[[0, j]] = f64::NEG_INFINITY;
                    }
                }
            }

            // Softmax
            let attn_weights = softmax_last_axis(&scores);

            // Weighted sum of values
            for (j, v_j) in v_window.rows().into_iter().enumerate() {
                let weight = attn_weights[[0, j]];
                for (k, v_k) in v_j.iter().enumerate() {
                    output[[i, k]] += weight * v_k;
                }
            }
        }

        // Project output
        output.dot(&self.w_output)
    }

    /// Get the effective attention span (total positions attended)
    pub fn attention_span(&self) -> usize {
        2 * self.config.window_size + 1
    }

    /// Get the window size
    pub fn window_size(&self) -> usize {
        self.config.window_size
    }

    /// Update weights (for training)
    pub fn set_weights(
        &mut self,
        w_query: Array2<f64>,
        w_key: Array2<f64>,
        w_value: Array2<f64>,
        w_output: Array2<f64>,
    ) {
        self.w_query = w_query;
        self.w_key = w_key;
        self.w_value = w_value;
        self.w_output = w_output;
    }
}

impl Attention for SlidingWindowAttention {
    fn forward(
        &self,
        query: &Array3<f64>,
        key: &Array3<f64>,
        value: &Array3<f64>,
        mask: Option<&Array2<bool>>,
    ) -> Array3<f64> {
        let batch_size = query.shape()[0];
        let seq_len = query.shape()[1];
        let d_model = query.shape()[2];

        // Process each batch element in parallel
        let outputs: Vec<Array2<f64>> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                let q = query.slice(s![b, .., ..]).to_owned();
                let k = key.slice(s![b, .., ..]).to_owned();
                let v = value.slice(s![b, .., ..]).to_owned();
                self.attention_single(&q, &k, &v, mask)
            })
            .collect();

        // Stack outputs
        let mut result = Array3::<f64>::zeros((batch_size, seq_len, d_model));
        for (b, output) in outputs.into_iter().enumerate() {
            for i in 0..seq_len {
                for j in 0..d_model {
                    result[[b, i, j]] = output[[i, j]];
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_sliding_window_attention() {
        let config = AttentionConfig {
            d_model: 64,
            n_heads: 4,
            window_size: 2,
            dropout: 0.0,
        };

        let attn = SlidingWindowAttention::new(config);

        // Create test input
        let batch = 2;
        let seq_len = 10;
        let d_model = 64;

        let x = Array3::random((batch, seq_len, d_model), Uniform::new(-1.0, 1.0));

        let output = attn.forward(&x, &x, &x, None);

        assert_eq!(output.shape(), &[batch, seq_len, d_model]);
    }

    #[test]
    fn test_attention_span() {
        let config = AttentionConfig {
            d_model: 64,
            n_heads: 4,
            window_size: 128,
            dropout: 0.0,
        };

        let attn = SlidingWindowAttention::new(config);
        assert_eq!(attn.attention_span(), 257); // 2 * 128 + 1
    }
}
