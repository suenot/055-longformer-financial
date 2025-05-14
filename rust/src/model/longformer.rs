//! Main Longformer model implementation

use ndarray::{Array1, Array2, Array3, Axis, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

use crate::attention::AttentionConfig;
use super::encoder::LongformerEncoder;

/// Configuration for Longformer model
#[derive(Debug, Clone)]
pub struct LongformerConfig {
    /// Input feature dimension (for time series)
    pub input_dim: usize,
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of encoder layers
    pub n_layers: usize,
    /// Sliding window size (one side)
    pub window_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Number of global tokens
    pub num_global_tokens: usize,
    /// Dropout probability
    pub dropout: f64,
}

impl Default for LongformerConfig {
    fn default() -> Self {
        Self {
            input_dim: 32,
            d_model: 256,
            n_heads: 8,
            n_layers: 4,
            window_size: 256,
            max_seq_len: 4096,
            num_global_tokens: 1,
            dropout: 0.1,
        }
    }
}

/// Longformer model for financial time series
pub struct Longformer {
    /// Model configuration
    config: LongformerConfig,
    /// Input projection layer
    input_proj: Array2<f64>,
    /// Positional embeddings
    pos_embeddings: Array2<f64>,
    /// Encoder layers
    encoders: Vec<LongformerEncoder>,
    /// Output projection for classification/regression
    output_proj: Array2<f64>,
    /// Output bias
    output_bias: Array1<f64>,
}

impl Longformer {
    /// Create a new Longformer model
    pub fn new(config: LongformerConfig) -> Self {
        let attn_config = AttentionConfig {
            d_model: config.d_model,
            n_heads: config.n_heads,
            window_size: config.window_size,
            dropout: config.dropout,
        };

        // Initialize input projection
        let limit = (6.0 / (config.input_dim + config.d_model) as f64).sqrt();
        let input_proj = Array2::random(
            (config.input_dim, config.d_model),
            Uniform::new(-limit, limit),
        );

        // Initialize positional embeddings (sinusoidal)
        let pos_embeddings = Self::create_sinusoidal_embeddings(
            config.max_seq_len,
            config.d_model,
        );

        // Create encoder layers
        let global_indices: Vec<usize> = (0..config.num_global_tokens).collect();
        let encoders: Vec<LongformerEncoder> = (0..config.n_layers)
            .map(|_| LongformerEncoder::new(
                attn_config.clone(),
                true,
                global_indices.clone(),
            ))
            .collect();

        // Output projection (3 classes: buy, hold, sell)
        let out_limit = (6.0 / (config.d_model + 3) as f64).sqrt();
        let output_proj = Array2::random(
            (config.d_model, 3),
            Uniform::new(-out_limit, out_limit),
        );
        let output_bias = Array1::zeros(3);

        Self {
            config,
            input_proj,
            pos_embeddings,
            encoders,
            output_proj,
            output_bias,
        }
    }

    /// Create sinusoidal positional embeddings
    fn create_sinusoidal_embeddings(max_len: usize, d_model: usize) -> Array2<f64> {
        let mut embeddings = Array2::<f64>::zeros((max_len, d_model));

        for pos in 0..max_len {
            for i in 0..d_model {
                let angle = pos as f64 / (10000.0_f64).powf((2 * (i / 2)) as f64 / d_model as f64);
                if i % 2 == 0 {
                    embeddings[[pos, i]] = angle.sin();
                } else {
                    embeddings[[pos, i]] = angle.cos();
                }
            }
        }

        embeddings
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape (batch_size, seq_len, input_dim)
    ///
    /// # Returns
    /// Output logits of shape (batch_size, 3) for buy/hold/sell classification
    pub fn forward(&self, x: &Array3<f64>) -> Array2<f64> {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        let mut outputs = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            // Extract single batch
            let x_b = x.slice(s![b, .., ..]).to_owned();

            // Input projection: (seq_len, input_dim) -> (seq_len, d_model)
            let h = x_b.dot(&self.input_proj);

            // Add positional embeddings
            let pos_emb = self.pos_embeddings.slice(s![0..seq_len, ..]).to_owned();
            let mut h = h + pos_emb;

            // Pass through encoder layers
            for encoder in &self.encoders {
                h = encoder.forward(&h);
            }

            // Use first token (CLS) for classification
            let cls_output = h.slice(s![0, ..]).to_owned();

            // Project to output
            let logits = cls_output.dot(&self.output_proj) + &self.output_bias;

            outputs.push(logits);
        }

        // Stack outputs
        let mut result = Array2::<f64>::zeros((batch_size, 3));
        for (b, logits) in outputs.into_iter().enumerate() {
            for i in 0..3 {
                result[[b, i]] = logits[i];
            }
        }

        result
    }

    /// Predict class probabilities using softmax
    pub fn predict_proba(&self, x: &Array3<f64>) -> Array2<f64> {
        let logits = self.forward(x);

        // Softmax
        let mut probs = Array2::<f64>::zeros(logits.dim());

        for (i, row) in logits.rows().into_iter().enumerate() {
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_row: Vec<f64> = row.iter().map(|&v| (v - max_val).exp()).collect();
            let sum_exp: f64 = exp_row.iter().sum();

            for (j, &exp_val) in exp_row.iter().enumerate() {
                probs[[i, j]] = exp_val / sum_exp;
            }
        }

        probs
    }

    /// Predict class labels
    ///
    /// Returns: 0 = Sell, 1 = Hold, 2 = Buy
    pub fn predict(&self, x: &Array3<f64>) -> Vec<usize> {
        let logits = self.forward(x);

        logits.rows().into_iter().map(|row| {
            let (max_idx, _) = row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            max_idx
        }).collect()
    }

    /// Get model configuration
    pub fn config(&self) -> &LongformerConfig {
        &self.config
    }

    /// Get the number of parameters in the model
    pub fn num_parameters(&self) -> usize {
        let input_params = self.input_proj.len();
        let pos_params = self.pos_embeddings.len();
        let output_params = self.output_proj.len() + self.output_bias.len();

        // Approximate encoder parameters (simplified)
        let d_model = self.config.d_model;
        let encoder_params = self.config.n_layers * (
            4 * d_model * d_model +  // Q, K, V, O projections
            2 * d_model +            // Layer norm params
            d_model * d_model * 8 +  // FFN
            d_model * 2              // FFN bias
        );

        input_params + pos_params + output_params + encoder_params
    }
}

/// Factory function to create a Longformer for time series trading
pub fn create_longformer_timeseries(
    input_dim: usize,
    seq_len: usize,
    d_model: usize,
    n_layers: usize,
) -> Longformer {
    let config = LongformerConfig {
        input_dim,
        d_model,
        n_heads: d_model / 32,  // Head dim = 32
        n_layers,
        window_size: seq_len / 16,  // ~6% of sequence
        max_seq_len: seq_len,
        num_global_tokens: 1,
        dropout: 0.1,
    };

    Longformer::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_longformer_creation() {
        let config = LongformerConfig {
            input_dim: 16,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            window_size: 8,
            max_seq_len: 128,
            num_global_tokens: 1,
            dropout: 0.1,
        };

        let model = Longformer::new(config);
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_longformer_forward() {
        let config = LongformerConfig {
            input_dim: 16,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            window_size: 8,
            max_seq_len: 128,
            num_global_tokens: 1,
            dropout: 0.0,
        };

        let model = Longformer::new(config);

        // Create test input
        let batch = 2;
        let seq_len = 32;
        let input_dim = 16;

        let x = Array3::random((batch, seq_len, input_dim), Uniform::new(-1.0, 1.0));

        let output = model.forward(&x);

        assert_eq!(output.shape(), &[batch, 3]);
    }

    #[test]
    fn test_predictions() {
        let model = create_longformer_timeseries(8, 64, 32, 2);

        let x = Array3::random((4, 64, 8), Uniform::new(-1.0, 1.0));

        let probs = model.predict_proba(&x);
        assert_eq!(probs.shape(), &[4, 3]);

        // Check probabilities sum to 1
        for row in probs.rows() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        let predictions = model.predict(&x);
        assert_eq!(predictions.len(), 4);

        // Check predictions are valid class indices
        for &pred in &predictions {
            assert!(pred < 3);
        }
    }

    #[test]
    fn test_sinusoidal_embeddings() {
        let embeddings = Longformer::create_sinusoidal_embeddings(100, 64);

        assert_eq!(embeddings.shape(), &[100, 64]);

        // Check that embeddings are bounded
        for &val in embeddings.iter() {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }
}
