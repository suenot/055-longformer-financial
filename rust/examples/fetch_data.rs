//! Example: Fetching data from Bybit API
//!
//! This example demonstrates how to use the BybitClient to fetch
//! cryptocurrency market data.

use longformer_financial::api::{BybitClient, KlineInterval};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    env_logger::init();

    println!("=== Longformer Financial: Data Fetching Example ===\n");

    // Create Bybit API client
    let client = BybitClient::new();

    // Fetch BTC/USDT hourly data
    println!("Fetching BTC/USDT hourly candles...");
    let klines = client
        .fetch_klines("BTCUSDT", KlineInterval::Hour1, 100, None, None)
        .await?;

    println!("Fetched {} candles\n", klines.len());

    // Display first few candles
    println!("First 5 candles:");
    println!("{:>20} {:>12} {:>12} {:>12} {:>12} {:>15}",
        "Timestamp", "Open", "High", "Low", "Close", "Volume");
    println!("{}", "-".repeat(85));

    for kline in klines.iter().take(5) {
        println!(
            "{:>20} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.2}",
            kline.datetime().format("%Y-%m-%d %H:%M"),
            kline.open,
            kline.high,
            kline.low,
            kline.close,
            kline.volume
        );
    }

    // Calculate some basic statistics
    println!("\n=== Statistics ===");

    let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
    let min_price = closes.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_price = closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let avg_price = closes.iter().sum::<f64>() / closes.len() as f64;

    println!("Price range: ${:.2} - ${:.2}", min_price, max_price);
    println!("Average price: ${:.2}", avg_price);

    // Calculate average true range
    let mut atr_sum = 0.0;
    let mut prev_close = None;

    for kline in &klines {
        atr_sum += kline.true_range(prev_close);
        prev_close = Some(kline.close);
    }

    let atr = atr_sum / klines.len() as f64;
    println!("Average True Range: ${:.2}", atr);

    // Count bullish vs bearish candles
    let bullish = klines.iter().filter(|k| k.is_bullish()).count();
    let bearish = klines.iter().filter(|k| k.is_bearish()).count();

    println!("\nCandle distribution:");
    println!("  Bullish: {} ({:.1}%)", bullish, bullish as f64 / klines.len() as f64 * 100.0);
    println!("  Bearish: {} ({:.1}%)", bearish, bearish as f64 / klines.len() as f64 * 100.0);

    // Fetch ETH data for comparison
    println!("\n=== Fetching ETH/USDT for comparison ===");
    let eth_klines = client
        .fetch_klines("ETHUSDT", KlineInterval::Hour1, 100, None, None)
        .await?;

    let eth_closes: Vec<f64> = eth_klines.iter().map(|k| k.close).collect();
    let eth_avg = eth_closes.iter().sum::<f64>() / eth_closes.len() as f64;

    println!("ETH average price: ${:.2}", eth_avg);

    // Calculate simple correlation (returns)
    let btc_returns: Vec<f64> = closes.windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    let eth_returns: Vec<f64> = eth_closes.windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    let btc_mean = btc_returns.iter().sum::<f64>() / btc_returns.len() as f64;
    let eth_mean = eth_returns.iter().sum::<f64>() / eth_returns.len() as f64;

    let mut covariance = 0.0;
    let mut btc_var = 0.0;
    let mut eth_var = 0.0;

    for i in 0..btc_returns.len().min(eth_returns.len()) {
        let btc_diff = btc_returns[i] - btc_mean;
        let eth_diff = eth_returns[i] - eth_mean;
        covariance += btc_diff * eth_diff;
        btc_var += btc_diff * btc_diff;
        eth_var += eth_diff * eth_diff;
    }

    let correlation = covariance / (btc_var.sqrt() * eth_var.sqrt());
    println!("BTC-ETH return correlation: {:.4}", correlation);

    println!("\n=== Done ===");

    Ok(())
}
