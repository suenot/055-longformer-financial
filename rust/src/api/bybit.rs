//! Bybit REST API client for fetching market data

use reqwest::Client;
use std::time::Duration;

use super::types::{ApiError, BybitKlineResult, BybitResponse, Kline, KlineInterval};

/// Base URL for Bybit API
const BYBIT_API_BASE: &str = "https://api.bybit.com";

/// Bybit API client
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl BybitClient {
    /// Create a new Bybit client with default settings
    pub fn new() -> Self {
        Self::with_timeout(Duration::from_secs(30))
    }

    /// Create a new Bybit client with custom timeout
    pub fn with_timeout(timeout: Duration) -> Self {
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: BYBIT_API_BASE.to_string(),
        }
    }

    /// Create a client pointing to testnet
    pub fn testnet() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: "https://api-testnet.bybit.com".to_string(),
        }
    }

    /// Fetch kline/candlestick data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Kline interval
    /// * `limit` - Number of candles to fetch (max 1000)
    /// * `start_time` - Optional start timestamp in milliseconds
    /// * `end_time` - Optional end timestamp in milliseconds
    ///
    /// # Returns
    /// Vector of Kline data, ordered from oldest to newest
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: KlineInterval,
        limit: u32,
        start_time: Option<i64>,
        end_time: Option<i64>,
    ) -> Result<Vec<Kline>, ApiError> {
        let mut url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url,
            symbol,
            interval.to_bybit_str(),
            limit.min(1000)
        );

        if let Some(start) = start_time {
            url.push_str(&format!("&start={}", start));
        }

        if let Some(end) = end_time {
            url.push_str(&format!("&end={}", end));
        }

        log::debug!("Fetching klines from: {}", url);

        let response = self.client.get(&url).send().await?;

        let bybit_response: BybitResponse<BybitKlineResult> = response.json().await?;

        if bybit_response.ret_code != 0 {
            return Err(ApiError::ApiResponse {
                code: bybit_response.ret_code,
                message: bybit_response.ret_msg,
            });
        }

        let klines: Vec<Kline> = bybit_response
            .result
            .list
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 7 {
                    Some(Kline::new(
                        row[0].parse().ok()?,
                        row[1].parse().ok()?,
                        row[2].parse().ok()?,
                        row[3].parse().ok()?,
                        row[4].parse().ok()?,
                        row[5].parse().ok()?,
                        row[6].parse().ok()?,
                    ))
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first, reverse for chronological order
        let mut klines = klines;
        klines.reverse();

        Ok(klines)
    }

    /// Fetch klines with automatic pagination for large datasets
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol
    /// * `interval` - Kline interval
    /// * `total_candles` - Total number of candles to fetch
    ///
    /// # Returns
    /// Vector of Kline data, ordered from oldest to newest
    pub async fn fetch_klines_paginated(
        &self,
        symbol: &str,
        interval: KlineInterval,
        total_candles: u32,
    ) -> Result<Vec<Kline>, ApiError> {
        let mut all_klines = Vec::with_capacity(total_candles as usize);
        let mut end_time: Option<i64> = None;
        let mut remaining = total_candles;

        while remaining > 0 {
            let batch_size = remaining.min(1000);

            let klines = self
                .fetch_klines(symbol, interval, batch_size, None, end_time)
                .await?;

            if klines.is_empty() {
                break;
            }

            // Update end_time for next batch (get data before oldest candle)
            end_time = Some(klines.first().unwrap().timestamp - 1);

            remaining = remaining.saturating_sub(klines.len() as u32);

            // Insert at beginning to maintain chronological order
            for kline in klines.into_iter().rev() {
                all_klines.insert(0, kline);
            }

            // Small delay to avoid rate limiting
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Ok(all_klines)
    }

    /// Fetch latest ticker price
    pub async fn fetch_ticker(&self, symbol: &str) -> Result<f64, ApiError> {
        let url = format!(
            "{}/v5/market/tickers?category=linear&symbol={}",
            self.base_url, symbol
        );

        let response = self.client.get(&url).send().await?;
        let json: serde_json::Value = response.json().await?;

        if json["retCode"].as_i64() != Some(0) {
            return Err(ApiError::ApiResponse {
                code: json["retCode"].as_i64().unwrap_or(-1) as i32,
                message: json["retMsg"].as_str().unwrap_or("Unknown error").to_string(),
            });
        }

        let price_str = json["result"]["list"][0]["lastPrice"]
            .as_str()
            .ok_or_else(|| ApiError::ParseError(serde_json::Error::io(
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing price")
            )))?;

        price_str
            .parse()
            .map_err(|_| ApiError::ParseError(serde_json::Error::io(
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid price format")
            )))
    }

    /// Get available trading pairs
    pub async fn fetch_symbols(&self) -> Result<Vec<String>, ApiError> {
        let url = format!(
            "{}/v5/market/instruments-info?category=linear",
            self.base_url
        );

        let response = self.client.get(&url).send().await?;
        let json: serde_json::Value = response.json().await?;

        if json["retCode"].as_i64() != Some(0) {
            return Err(ApiError::ApiResponse {
                code: json["retCode"].as_i64().unwrap_or(-1) as i32,
                message: json["retMsg"].as_str().unwrap_or("Unknown error").to_string(),
            });
        }

        let symbols: Vec<String> = json["result"]["list"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| item["symbol"].as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        Ok(symbols)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires network access
    async fn test_fetch_klines() {
        let client = BybitClient::new();
        let klines = client
            .fetch_klines("BTCUSDT", KlineInterval::Hour1, 10, None, None)
            .await
            .unwrap();

        assert!(!klines.is_empty());
        assert!(klines.len() <= 10);

        // Check chronological order
        for i in 1..klines.len() {
            assert!(klines[i].timestamp > klines[i - 1].timestamp);
        }
    }

    #[tokio::test]
    #[ignore] // Requires network access
    async fn test_fetch_ticker() {
        let client = BybitClient::new();
        let price = client.fetch_ticker("BTCUSDT").await.unwrap();

        assert!(price > 0.0);
    }
}
