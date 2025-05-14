//! Longformer encoder layer implementation

use ndarray::{Array2, Array3, Axis, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

use crate::attention::{AttentionConfig, SlidingWindowAttention, GlobalAttention, Attention};

/// Layer normalization
pub struct LayerNorm {
    /// Normalized shape
    normalized_shape: usize,
    /// Learnable scale parameter
    gamma: Array2<f64>,
    /// Learnable shift parameter
    beta: Array2<f64>,
    /// Small constant for numerical stability
    eps: f64,
}

impl LayerNorm {
    /// Create a new layer normalization
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            normalized_shape,
            gamma: Array2::ones((1, normalized_shape)),
            beta: Array2::zeros((1, normalized_shape)),
            eps: 1e-5,
        }
    }

    /// Apply layer normalization
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let mean = x.mean_axis(Axis(1)).unwrap();
        let variance = x.var_axis(Axis(1), 0.0);

        let x_centered = x - &mean.insert_axis(Axis(1));
        let std = (variance + self.eps).mapv(f64::sqrt);
        let x_norm = x_centered / &std.insert_axis(Axis(1));

        &x_norm * &self.gamma + &self.beta
    }
}

/// Feed-forward network
pub struct FeedForward {
    /// First linear layer weights
    w1: Array2<f64>,
    /// First linear layer bias
    b1: Array2<f64>,
    /// Second linear layer weights
    w2: Array2<f64>,
    /// Second linear layer bias
    b2: Array2<f64>,
}

impl FeedForward {
    /// Create a new feed-forward network
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let limit1 = (6.0 / (d_model + d_ff) as f64).sqrt();
        let limit2 = (6.0 / (d_ff + d_model) as f64).sqrt();

        Self {
            w1: Array2::random((d_model, d_ff), Uniform::new(-limit1, limit1)),
            b1: Array2::zeros((1, d_ff)),
            w2: Array2::random((d_ff, d_model), Uniform::new(-limit2, limit2)),
            b2: Array2::zeros((1, d_model)),
        }
    }

    /// Apply feed-forward transformation with GELU activation
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        // First linear + GELU
        let h = x.dot(&self.w1) + &self.b1;
        let h = h.mapv(|v| v * 0.5 * (1.0 + (v / 2.0_f64.sqrt()).tanh()));

        // Second linear
        h.dot(&self.w2) + &self.b2
    }
}

/// Longformer encoder layer
///
/// Combines sliding window attention with optional global attention,
/// layer normalization, and feed-forward networks.
pub struct LongformerEncoder {
    /// Sliding window attention
    sliding_attention: SlidingWindowAttention,
    /// Global attention (optional)
    global_attention: Option<GlobalAttention>,
    /// First layer normalization
    norm1: LayerNorm,
    /// Second layer normalization
    norm2: LayerNorm,
    /// Feed-forward network
    ffn: FeedForward,
    /// Dropout probability
    dropout: f64,
    /// Global token indices
    global_indices: Vec<usize>,
}

impl LongformerEncoder {
    /// Create a new encoder layer
    pub fn new(config: AttentionConfig, use_global: bool, global_indices: Vec<usize>) -> Self {
        let d_model = config.d_model;
        let d_ff = d_model * 4;

        Self {
            sliding_attention: SlidingWindowAttention::new(config.clone()),
            global_attention: if use_global {
                Some(GlobalAttention::new(config.clone()))
            } else {
                None
            },
            norm1: LayerNorm::new(d_model),
            norm2: LayerNorm::new(d_model),
            ffn: FeedForward::new(d_model, d_ff),
            dropout: config.dropout,
            global_indices,
        }
    }

    /// Forward pass through the encoder layer
    ///
    /// Uses pre-norm architecture: LayerNorm -> Attention -> Residual
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let seq_len = x.nrows();
        let d_model = x.ncols();

        // Pre-norm for attention
        let x_norm = self.norm1.forward(x);

        // Convert to 3D for attention (batch=1)
        let x_3d = x_norm.clone().insert_axis(Axis(0));

        // Sliding window attention
        let attn_out_3d = self.sliding_attention.forward(&x_3d, &x_3d, &x_3d, None);

        // Convert back to 2D
        let mut attn_out = attn_out_3d.slice(s![0, .., ..]).to_owned();

        // Add global attention if present
        if let Some(ref global_attn) = self.global_attention {
            if !self.global_indices.is_empty() {
                let global_out = global_attn.forward_with_indices(&x_norm, &self.global_indices);
                // Combine sliding and global attention
                attn_out = (&attn_out + &global_out).mapv(|v| v / 2.0);
            }
        }

        // Residual connection
        let x = x + &attn_out;

        // Pre-norm for FFN
        let x_norm = self.norm2.forward(&x);

        // Feed-forward
        let ffn_out = self.ffn.forward(&x_norm);

        // Residual connection
        &x + &ffn_out
    }

    /// Set global token indices
    pub fn set_global_indices(&mut self, indices: Vec<usize>) {
        self.global_indices = indices;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::new(64);
        let x = Array2::random((10, 64), Uniform::new(-1.0, 1.0));
        let output = ln.forward(&x);

        assert_eq!(output.shape(), &[10, 64]);

        // Check that output is approximately normalized (mean ≈ 0, std ≈ 1)
        for row in output.rows() {
            let mean: f64 = row.iter().sum::<f64>() / row.len() as f64;
            assert!(mean.abs() < 0.1, "Mean should be close to 0: {}", mean);
        }
    }

    #[test]
    fn test_feed_forward() {
        let ffn = FeedForward::new(64, 256);
        let x = Array2::random((10, 64), Uniform::new(-1.0, 1.0));
        let output = ffn.forward(&x);

        assert_eq!(output.shape(), &[10, 64]);
    }

    #[test]
    fn test_encoder_layer() {
        let config = AttentionConfig {
            d_model: 64,
            n_heads: 4,
            window_size: 4,
            dropout: 0.0,
        };

        let encoder = LongformerEncoder::new(config, true, vec![0]);

        let x = Array2::random((16, 64), Uniform::new(-1.0, 1.0));
        let output = encoder.forward(&x);

        assert_eq!(output.shape(), &[16, 64]);
    }
}
