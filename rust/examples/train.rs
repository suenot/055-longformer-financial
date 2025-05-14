//! Example: Training a Longformer model on financial data
//!
//! This example demonstrates how to:
//! 1. Create a Longformer model
//! 2. Generate synthetic training data
//! 3. Run forward passes through the model

use longformer_financial::model::{Longformer, LongformerConfig, create_longformer_timeseries};
use ndarray::{Array3, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

fn main() {
    println!("=== Longformer Financial: Training Example ===\n");

    // Create model configuration
    let config = LongformerConfig {
        input_dim: 32,      // Features per timestep
        d_model: 128,       // Model dimension
        n_heads: 4,         // Attention heads
        n_layers: 2,        // Encoder layers
        window_size: 64,    // Sliding window size
        max_seq_len: 512,   // Maximum sequence length
        num_global_tokens: 1, // CLS token for classification
        dropout: 0.1,
    };

    println!("Model Configuration:");
    println!("  Input dim: {}", config.input_dim);
    println!("  Model dim: {}", config.d_model);
    println!("  Attention heads: {}", config.n_heads);
    println!("  Encoder layers: {}", config.n_layers);
    println!("  Window size: {}", config.window_size);
    println!("  Max sequence: {}", config.max_seq_len);

    // Create model
    let model = Longformer::new(config);
    println!("\nModel parameters: ~{}", model.num_parameters());

    // Generate synthetic training data
    println!("\n=== Generating Synthetic Data ===");

    let batch_size = 4;
    let seq_len = 256;
    let input_dim = 32;

    // Create random input (simulating financial features)
    let x = Array3::random((batch_size, seq_len, input_dim), Uniform::new(-1.0, 1.0));

    println!("Input shape: {:?}", x.shape());

    // Forward pass
    println!("\n=== Forward Pass ===");

    let start = std::time::Instant::now();
    let logits = model.forward(&x);
    let elapsed = start.elapsed();

    println!("Output logits shape: {:?}", logits.shape());
    println!("Forward pass time: {:?}", elapsed);

    // Get predictions
    println!("\n=== Predictions ===");

    let probs = model.predict_proba(&x);
    let predictions = model.predict(&x);

    println!("Class probabilities:");
    for (i, row) in probs.rows().into_iter().enumerate() {
        let sell = row[0];
        let hold = row[1];
        let buy = row[2];
        let pred = match predictions[i] {
            0 => "SELL",
            1 => "HOLD",
            _ => "BUY",
        };
        println!(
            "  Sample {}: Sell={:.3}, Hold={:.3}, Buy={:.3} -> {}",
            i, sell, hold, buy, pred
        );
    }

    // Demonstrate factory function
    println!("\n=== Using Factory Function ===");

    let model2 = create_longformer_timeseries(
        16,    // input_dim
        128,   // seq_len
        64,    // d_model
        2,     // n_layers
    );

    let x2 = Array3::random((2, 128, 16), Uniform::new(-1.0, 1.0));
    let preds2 = model2.predict(&x2);

    println!("Factory model predictions: {:?}", preds2);

    // Performance benchmark
    println!("\n=== Performance Benchmark ===");

    let n_iterations = 10;
    let mut total_time = std::time::Duration::ZERO;

    for _ in 0..n_iterations {
        let x_bench = Array3::random((1, 512, 32), Uniform::new(-1.0, 1.0));
        let start = std::time::Instant::now();
        let _ = model.forward(&x_bench);
        total_time += start.elapsed();
    }

    let avg_time = total_time / n_iterations as u32;
    println!(
        "Average forward pass time (batch=1, seq=512): {:?}",
        avg_time
    );

    // Throughput
    let throughput = 1.0 / avg_time.as_secs_f64();
    println!("Throughput: {:.1} samples/second", throughput);

    println!("\n=== Training Complete ===");
    println!("\nNote: This is a demonstration example.");
    println!("Full training would require:");
    println!("  - Real market data from Bybit/Yahoo Finance");
    println!("  - Proper train/validation split");
    println!("  - Gradient computation (not implemented in pure Rust)");
    println!("  - Consider using PyTorch with Rust bindings for production");
}
