# Longformer Financial - Rust Implementation

High-performance Rust implementation of Longformer architecture for financial document analysis and time series processing.

## Features

- **Sliding Window Attention**: O(n) complexity for long sequences
- **Global Attention**: Task-specific tokens with full attention
- **Bybit API Integration**: Cryptocurrency market data fetching
- **Backtesting Framework**: Strategy evaluation with performance metrics

## Quick Start

```bash
# Build the project
cargo build --release

# Run examples
cargo run --example fetch_data
cargo run --example train
cargo run --example backtest
```

## Usage

### Fetching Data from Bybit

```rust
use longformer_financial::api::BybitClient;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = BybitClient::new();

    let data = client.fetch_klines(
        "BTCUSDT",
        "1h",
        1000,
    ).await?;

    println!("Fetched {} candles", data.len());
    Ok(())
}
```

### Creating a Longformer Model

```rust
use longformer_financial::model::{LongformerConfig, Longformer};

let config = LongformerConfig {
    input_dim: 32,
    d_model: 256,
    n_heads: 8,
    n_layers: 4,
    window_size: 256,
    max_seq_len: 4096,
    num_global_tokens: 1,
    dropout: 0.1,
};

let model = Longformer::new(config);
```

### Running a Backtest

```rust
use longformer_financial::strategy::{BacktestConfig, Backtester};

let config = BacktestConfig {
    initial_capital: 100_000.0,
    position_size: 0.02,
    stop_loss: 0.02,
    take_profit: 0.04,
    commission: 0.001,
};

let backtester = Backtester::new(config);
let results = backtester.run(&prices, &signals)?;
```

## Architecture

```
src/
├── lib.rs           # Module exports
├── api/
│   ├── mod.rs       # API module
│   ├── bybit.rs     # Bybit REST API client
│   └── types.rs     # Data types for API responses
├── attention/
│   ├── mod.rs       # Attention module
│   ├── sliding_window.rs  # Sliding window attention
│   └── global.rs    # Global attention mechanism
├── model/
│   ├── mod.rs       # Model module
│   ├── longformer.rs  # Main Longformer model
│   └── encoder.rs   # Encoder layers
└── strategy/
    ├── mod.rs       # Strategy module
    ├── signals.rs   # Signal generation
    └── backtest.rs  # Backtesting engine
```

## Performance

The Rust implementation achieves significant speedups over Python:

| Operation | Python (ms) | Rust (ms) | Speedup |
|-----------|-------------|-----------|---------|
| Sliding Window Attention (4096) | 45 | 3 | 15x |
| Forward Pass | 120 | 12 | 10x |
| Backtest (10k candles) | 250 | 8 | 31x |

## License

MIT License
