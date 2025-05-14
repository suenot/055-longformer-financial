//! Strategy module for backtesting and signal generation
//!
//! Provides tools for evaluating trading strategies using Longformer predictions.

mod signals;
mod backtest;

pub use signals::{Signal, SignalGenerator};
pub use backtest::{Backtester, BacktestConfig, BacktestResult};
