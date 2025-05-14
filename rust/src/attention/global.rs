//! Global attention implementation
//!
//! Global attention allows selected tokens to attend to all positions
//! and be attended by all positions, enabling long-range dependencies.

use ndarray::{Array2, Array3, Axis, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rayon::prelude::*;

use super::{Attention, AttentionConfig, softmax_last_axis};

/// Global attention mechanism
///
/// Tokens marked as "global" attend to the entire sequence and
/// all other tokens attend to the global tokens.
pub struct GlobalAttention {
    /// Configuration
    config: AttentionConfig,

    /// Query projection for global tokens
    w_query_global: Array2<f64>,

    /// Key projection for global tokens
    w_key_global: Array2<f64>,

    /// Value projection for global tokens
    w_value_global: Array2<f64>,

    /// Output projection
    w_output: Array2<f64>,

    /// Scaling factor
    scale: f64,
}

impl GlobalAttention {
    /// Create a new global attention layer
    pub fn new(config: AttentionConfig) -> Self {
        let d_model = config.d_model;
        let scale = 1.0 / ((d_model / config.n_heads) as f64).sqrt();

        let limit = (6.0 / (d_model + d_model) as f64).sqrt();
        let dist = Uniform::new(-limit, limit);

        Self {
            config,
            w_query_global: Array2::random((d_model, d_model), dist),
            w_key_global: Array2::random((d_model, d_model), dist),
            w_value_global: Array2::random((d_model, d_model), dist),
            w_output: Array2::random((d_model, d_model), dist),
            scale,
        }
    }

    /// Compute global attention
    ///
    /// # Arguments
    /// * `x` - Input tensor (seq_len, d_model)
    /// * `global_indices` - Indices of global tokens
    ///
    /// # Returns
    /// Output tensor with global attention applied
    pub fn forward_with_indices(
        &self,
        x: &Array2<f64>,
        global_indices: &[usize],
    ) -> Array2<f64> {
        let seq_len = x.nrows();
        let d_model = self.config.d_model;

        if global_indices.is_empty() {
            return x.clone();
        }

        // Project to Q, K, V for global tokens
        let q_global = x.dot(&self.w_query_global);
        let k_global = x.dot(&self.w_key_global);
        let v_global = x.dot(&self.w_value_global);

        let mut output = x.clone();

        // Global tokens attend to all positions
        for &g_idx in global_indices {
            if g_idx >= seq_len {
                continue;
            }

            let q_g = q_global.slice(s![g_idx, ..]);

            // Compute attention scores: q_g · all keys
            let mut scores = Array2::<f64>::zeros((1, seq_len));
            for (j, k_j) in k_global.rows().into_iter().enumerate() {
                let score: f64 = q_g.iter().zip(k_j.iter()).map(|(a, b)| a * b).sum();
                scores[[0, j]] = score * self.scale;
            }

            // Softmax
            let attn_weights = softmax_last_axis(&scores);

            // Weighted sum of all values
            let mut attended = vec![0.0; d_model];
            for (j, v_j) in v_global.rows().into_iter().enumerate() {
                let weight = attn_weights[[0, j]];
                for (k, v_k) in v_j.iter().enumerate() {
                    attended[k] += weight * v_k;
                }
            }

            // Update global token output
            for (k, &v) in attended.iter().enumerate() {
                output[[g_idx, k]] = v;
            }
        }

        // All positions attend to global tokens
        for i in 0..seq_len {
            // Skip if this is a global token (already processed)
            if global_indices.contains(&i) {
                continue;
            }

            let q_i = q_global.slice(s![i, ..]);

            // Compute attention scores to global tokens only
            let n_global = global_indices.len();
            let mut scores = Array2::<f64>::zeros((1, n_global));

            for (j, &g_idx) in global_indices.iter().enumerate() {
                if g_idx < seq_len {
                    let k_g = k_global.slice(s![g_idx, ..]);
                    let score: f64 = q_i.iter().zip(k_g.iter()).map(|(a, b)| a * b).sum();
                    scores[[0, j]] = score * self.scale;
                }
            }

            // Softmax over global tokens
            let attn_weights = softmax_last_axis(&scores);

            // Compute weighted contribution from global tokens
            let mut global_contribution = vec![0.0; d_model];
            for (j, &g_idx) in global_indices.iter().enumerate() {
                if g_idx < seq_len {
                    let weight = attn_weights[[0, j]];
                    let v_g = v_global.slice(s![g_idx, ..]);
                    for (k, v_k) in v_g.iter().enumerate() {
                        global_contribution[k] += weight * v_k;
                    }
                }
            }

            // Add global contribution (residual connection style)
            for (k, &gc) in global_contribution.iter().enumerate() {
                output[[i, k]] = (output[[i, k]] + gc) / 2.0;
            }
        }

        // Project output
        output.dot(&self.w_output)
    }
}

impl Attention for GlobalAttention {
    fn forward(
        &self,
        query: &Array3<f64>,
        key: &Array3<f64>,
        value: &Array3<f64>,
        mask: Option<&Array2<bool>>,
    ) -> Array3<f64> {
        // For the trait implementation, assume first token is global
        // (CLS token pattern)
        let global_indices = vec![0];

        let batch_size = query.shape()[0];
        let seq_len = query.shape()[1];
        let d_model = query.shape()[2];

        let outputs: Vec<Array2<f64>> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                let x = query.slice(s![b, .., ..]).to_owned();
                self.forward_with_indices(&x, &global_indices)
            })
            .collect();

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
    use ndarray::Array2;

    #[test]
    fn test_global_attention() {
        let config = AttentionConfig {
            d_model: 64,
            n_heads: 4,
            window_size: 8,
            dropout: 0.0,
        };

        let attn = GlobalAttention::new(config);

        let seq_len = 20;
        let d_model = 64;
        let x = Array2::random((seq_len, d_model), Uniform::new(-1.0, 1.0));

        // First and last tokens are global
        let global_indices = vec![0, seq_len - 1];

        let output = attn.forward_with_indices(&x, &global_indices);

        assert_eq!(output.shape(), &[seq_len, d_model]);
    }

    #[test]
    fn test_empty_global_indices() {
        let config = AttentionConfig {
            d_model: 32,
            n_heads: 2,
            window_size: 4,
            dropout: 0.0,
        };

        let attn = GlobalAttention::new(config);

        let x = Array2::random((10, 32), Uniform::new(-1.0, 1.0));
        let output = attn.forward_with_indices(&x, &[]);

        // With no global tokens, output should equal input
        assert_eq!(output.shape(), x.shape());
    }
}
