//! # Longformer Financial
//!
//! A high-performance Rust implementation of the Longformer architecture
//! for financial document analysis and time series processing.
//!
//! ## Overview
//!
//! Longformer uses a combination of sliding window attention and global attention
//! to efficiently process long sequences with O(n) complexity instead of O(n²).
//!
//! ## Modules
//!
//! - `api`: Bybit API client for fetching cryptocurrency data
//! - `attention`: Sliding window and global attention mechanisms
//! - `model`: Longformer model implementation
//! - `strategy`: Backtesting and signal generation utilities

pub mod api;
pub mod attention;
pub mod model;
pub mod strategy;

// Re-exports for convenience
pub use api::{BybitClient, Kline};
pub use attention::{SlidingWindowAttention, GlobalAttention};
pub use model::{Longformer, LongformerConfig, LongformerEncoder};
pub use strategy::{Backtester, BacktestConfig, BacktestResult, Signal};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(VERSION, "0.1.0");
    }
}
