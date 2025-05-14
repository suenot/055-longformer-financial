//! Data types for API responses

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Error types for API operations
#[derive(Error, Debug)]
pub enum ApiError {
    #[error("HTTP request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),

    #[error("Failed to parse response: {0}")]
    ParseError(#[from] serde_json::Error),

    #[error("API returned error: {code} - {message}")]
    ApiResponse { code: i32, message: String },

    #[error("Invalid interval: {0}")]
    InvalidInterval(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,
}

/// Supported kline intervals
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KlineInterval {
    Min1,
    Min3,
    Min5,
    Min15,
    Min30,
    Hour1,
    Hour2,
    Hour4,
    Hour6,
    Hour12,
    Day1,
    Week1,
    Month1,
}

impl KlineInterval {
    /// Convert to Bybit API string format
    pub fn to_bybit_str(&self) -> &'static str {
        match self {
            KlineInterval::Min1 => "1",
            KlineInterval::Min3 => "3",
            KlineInterval::Min5 => "5",
            KlineInterval::Min15 => "15",
            KlineInterval::Min30 => "30",
            KlineInterval::Hour1 => "60",
            KlineInterval::Hour2 => "120",
            KlineInterval::Hour4 => "240",
            KlineInterval::Hour6 => "360",
            KlineInterval::Hour12 => "720",
            KlineInterval::Day1 => "D",
            KlineInterval::Week1 => "W",
            KlineInterval::Month1 => "M",
        }
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Result<Self, ApiError> {
        match s.to_lowercase().as_str() {
            "1m" | "1" => Ok(KlineInterval::Min1),
            "3m" | "3" => Ok(KlineInterval::Min3),
            "5m" | "5" => Ok(KlineInterval::Min5),
            "15m" | "15" => Ok(KlineInterval::Min15),
            "30m" | "30" => Ok(KlineInterval::Min30),
            "1h" | "60" => Ok(KlineInterval::Hour1),
            "2h" | "120" => Ok(KlineInterval::Hour2),
            "4h" | "240" => Ok(KlineInterval::Hour4),
            "6h" | "360" => Ok(KlineInterval::Hour6),
            "12h" | "720" => Ok(KlineInterval::Hour12),
            "1d" | "d" => Ok(KlineInterval::Day1),
            "1w" | "w" => Ok(KlineInterval::Week1),
            "1mo" | "m" => Ok(KlineInterval::Month1),
            _ => Err(ApiError::InvalidInterval(s.to_string())),
        }
    }

    /// Get interval duration in milliseconds
    pub fn duration_ms(&self) -> i64 {
        match self {
            KlineInterval::Min1 => 60_000,
            KlineInterval::Min3 => 180_000,
            KlineInterval::Min5 => 300_000,
            KlineInterval::Min15 => 900_000,
            KlineInterval::Min30 => 1_800_000,
            KlineInterval::Hour1 => 3_600_000,
            KlineInterval::Hour2 => 7_200_000,
            KlineInterval::Hour4 => 14_400_000,
            KlineInterval::Hour6 => 21_600_000,
            KlineInterval::Hour12 => 43_200_000,
            KlineInterval::Day1 => 86_400_000,
            KlineInterval::Week1 => 604_800_000,
            KlineInterval::Month1 => 2_592_000_000, // Approximate
        }
    }
}

/// OHLCV candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Opening timestamp in milliseconds
    pub timestamp: i64,

    /// Opening price
    pub open: f64,

    /// Highest price
    pub high: f64,

    /// Lowest price
    pub low: f64,

    /// Closing price
    pub close: f64,

    /// Trading volume
    pub volume: f64,

    /// Turnover (quote volume)
    pub turnover: f64,
}

impl Kline {
    /// Create a new Kline
    pub fn new(
        timestamp: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        turnover: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            turnover,
        }
    }

    /// Get timestamp as DateTime<Utc>
    pub fn datetime(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(self.timestamp)
            .unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap())
    }

    /// Calculate the typical price (HLC average)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate the weighted close price
    pub fn weighted_close(&self) -> f64 {
        (self.high + self.low + 2.0 * self.close) / 4.0
    }

    /// Calculate the true range
    pub fn true_range(&self, prev_close: Option<f64>) -> f64 {
        let hl = self.high - self.low;
        match prev_close {
            Some(pc) => {
                let hc = (self.high - pc).abs();
                let lc = (self.low - pc).abs();
                hl.max(hc).max(lc)
            }
            None => hl,
        }
    }

    /// Check if this is a bullish candle
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if this is a bearish candle
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Calculate the body size (absolute)
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Calculate upper shadow size
    pub fn upper_shadow(&self) -> f64 {
        self.high - self.close.max(self.open)
    }

    /// Calculate lower shadow size
    pub fn lower_shadow(&self) -> f64 {
        self.close.min(self.open) - self.low
    }
}

/// Bybit API response wrapper
#[derive(Debug, Deserialize)]
pub struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,

    #[serde(rename = "retMsg")]
    pub ret_msg: String,

    pub result: T,
}

/// Bybit kline response result
#[derive(Debug, Deserialize)]
pub struct BybitKlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_calculations() {
        let kline = Kline::new(
            1704067200000, // 2024-01-01 00:00:00
            100.0,
            110.0,
            95.0,
            105.0,
            1000.0,
            100000.0,
        );

        assert!(kline.is_bullish());
        assert!(!kline.is_bearish());
        assert_eq!(kline.body_size(), 5.0);
        assert_eq!(kline.upper_shadow(), 5.0);
        assert_eq!(kline.lower_shadow(), 5.0);
        assert!((kline.typical_price() - 103.333).abs() < 0.01);
    }

    #[test]
    fn test_interval_parsing() {
        assert_eq!(
            KlineInterval::from_str("1h").unwrap(),
            KlineInterval::Hour1
        );
        assert_eq!(
            KlineInterval::from_str("4h").unwrap(),
            KlineInterval::Hour4
        );
        assert_eq!(
            KlineInterval::from_str("1d").unwrap(),
            KlineInterval::Day1
        );
    }

    #[test]
    fn test_true_range() {
        let kline = Kline::new(0, 100.0, 110.0, 90.0, 105.0, 1000.0, 100000.0);

        // Without previous close
        assert_eq!(kline.true_range(None), 20.0);

        // With previous close outside range
        assert_eq!(kline.true_range(Some(85.0)), 25.0); // high - prev_close
        assert_eq!(kline.true_range(Some(115.0)), 25.0); // prev_close - low
    }
}
