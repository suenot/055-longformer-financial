//! API module for fetching market data
//!
//! This module provides clients for interacting with cryptocurrency
//! and stock market APIs.

mod bybit;
mod types;

pub use bybit::BybitClient;
pub use types::{Kline, KlineInterval, ApiError};
