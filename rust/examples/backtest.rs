//! Example: Running a backtest with Longformer signals
//!
//! This example demonstrates how to:
//! 1. Generate trading signals
//! 2. Run a backtest
//! 3. Analyze performance metrics

use longformer_financial::strategy::{
    Backtester, BacktestConfig, Signal, SignalGenerator,
};

fn main() {
    println!("=== Longformer Financial: Backtesting Example ===\n");

    // Generate synthetic price data (uptrend with noise)
    println!("Generating synthetic price data...");
    let n_days = 500;
    let prices: Vec<f64> = generate_price_series(n_days, 100.0, 0.0002, 0.015);

    println!("Generated {} price points", prices.len());
    println!("  Start price: ${:.2}", prices.first().unwrap());
    println!("  End price: ${:.2}", prices.last().unwrap());
    println!(
        "  Buy & Hold return: {:.2}%",
        (prices.last().unwrap() / prices.first().unwrap() - 1.0) * 100.0
    );

    // Generate synthetic model probabilities
    // In practice, these would come from the Longformer model
    println!("\n=== Generating Model Signals ===");

    let probs = generate_synthetic_probs(&prices);

    // Create signal generator
    let signal_gen = SignalGenerator::with_thresholds(0.55, 0.55)
        .with_confirmation(2);

    let signals = signal_gen.generate_series(&probs);
    let stats = signal_gen.signal_stats(&signals);

    println!("{}", stats);

    // Configure backtest
    println!("\n=== Running Backtest ===");

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        position_size: 0.1,     // 10% per trade
        stop_loss: 0.03,        // 3% stop loss
        take_profit: 0.06,      // 6% take profit
        commission: 0.001,      // 0.1% commission
        slippage: 0.0005,       // 0.05% slippage
        allow_short: false,
    };

    println!("Backtest Configuration:");
    println!("  Initial capital: ${:.0}", config.initial_capital);
    println!("  Position size: {:.0}%", config.position_size * 100.0);
    println!("  Stop loss: {:.1}%", config.stop_loss * 100.0);
    println!("  Take profit: {:.1}%", config.take_profit * 100.0);
    println!("  Commission: {:.2}%", config.commission * 100.0);

    // Run backtest
    let backtester = Backtester::new(config);
    let result = backtester.run(&prices, &signals);

    // Display results
    println!("\n{}", result);

    // Analyze trades
    println!("\n=== Trade Analysis ===");

    if !result.trades.is_empty() {
        let winning: Vec<_> = result.trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losing: Vec<_> = result.trades.iter().filter(|t| t.pnl < 0.0).collect();

        let avg_win = if !winning.is_empty() {
            winning.iter().map(|t| t.pnl).sum::<f64>() / winning.len() as f64
        } else {
            0.0
        };

        let avg_loss = if !losing.is_empty() {
            losing.iter().map(|t| t.pnl.abs()).sum::<f64>() / losing.len() as f64
        } else {
            0.0
        };

        println!("Winning trades: {}", winning.len());
        println!("Losing trades: {}", losing.len());
        println!("Average win: ${:.2}", avg_win);
        println!("Average loss: ${:.2}", avg_loss);

        if avg_loss > 0.0 {
            println!("Risk/Reward ratio: {:.2}", avg_win / avg_loss);
        }

        // Exit reason breakdown
        let signal_exits = result.trades.iter()
            .filter(|t| t.exit_reason == longformer_financial::strategy::ExitReason::Signal)
            .count();
        let sl_exits = result.trades.iter()
            .filter(|t| t.exit_reason == longformer_financial::strategy::ExitReason::StopLoss)
            .count();
        let tp_exits = result.trades.iter()
            .filter(|t| t.exit_reason == longformer_financial::strategy::ExitReason::TakeProfit)
            .count();

        println!("\nExit reasons:");
        println!("  Signal: {}", signal_exits);
        println!("  Stop Loss: {}", sl_exits);
        println!("  Take Profit: {}", tp_exits);
    }

    // Drawdown analysis
    println!("\n=== Drawdown Analysis ===");

    let (max_dd, max_dd_start, max_dd_end) = calculate_max_drawdown_details(&result.equity_curve);
    println!("Maximum drawdown: {:.2}%", max_dd * 100.0);
    println!("Drawdown period: days {} to {}", max_dd_start, max_dd_end);
    println!("Recovery needed: {:.2}%", (1.0 / (1.0 - max_dd) - 1.0) * 100.0);

    // Monthly returns (approximate)
    println!("\n=== Monthly Returns (approximate) ===");

    let monthly_returns = calculate_monthly_returns(&result.equity_curve, 21);
    for (i, ret) in monthly_returns.iter().enumerate().take(12) {
        let sign = if *ret >= 0.0 { "+" } else { "" };
        println!("  Month {:2}: {}{:.2}%", i + 1, sign, ret * 100.0);
    }

    // Compare with benchmark
    println!("\n=== Strategy vs Buy & Hold ===");

    let bh_return = prices.last().unwrap() / prices.first().unwrap() - 1.0;
    let strategy_return = result.total_return;

    println!("Buy & Hold return: {:.2}%", bh_return * 100.0);
    println!("Strategy return:   {:.2}%", strategy_return * 100.0);
    println!(
        "Outperformance:    {:.2}%",
        (strategy_return - bh_return) * 100.0
    );

    println!("\n=== Backtest Complete ===");
}

/// Generate a synthetic price series with drift and volatility
fn generate_price_series(n: usize, start: f64, drift: f64, volatility: f64) -> Vec<f64> {
    use rand::prelude::*;
    use rand_distr::Normal;

    let mut rng = rand::thread_rng();
    let normal = Normal::new(drift, volatility).unwrap();

    let mut prices = Vec::with_capacity(n);
    let mut price = start;

    for _ in 0..n {
        prices.push(price);
        let return_: f64 = rng.sample(normal);
        price *= 1.0 + return_;
    }

    prices
}

/// Generate synthetic model probabilities based on price momentum
fn generate_synthetic_probs(prices: &[f64]) -> Vec<[f64; 3]> {
    let lookback = 20;
    let mut probs = Vec::with_capacity(prices.len());

    for i in 0..prices.len() {
        if i < lookback {
            // Not enough data, neutral signal
            probs.push([0.33, 0.34, 0.33]);
            continue;
        }

        // Simple momentum: compare to moving average
        let ma: f64 = prices[i - lookback..i].iter().sum::<f64>() / lookback as f64;
        let momentum = (prices[i] - ma) / ma;

        // Convert momentum to probabilities
        let (sell, hold, buy) = if momentum > 0.02 {
            // Bullish momentum -> higher buy probability
            let conf = (momentum * 10.0).min(0.4);
            (0.2 - conf / 2.0, 0.4 - conf / 2.0, 0.4 + conf)
        } else if momentum < -0.02 {
            // Bearish momentum -> higher sell probability
            let conf = (-momentum * 10.0).min(0.4);
            (0.4 + conf, 0.4 - conf / 2.0, 0.2 - conf / 2.0)
        } else {
            // Neutral
            (0.25, 0.50, 0.25)
        };

        probs.push([sell, hold, buy]);
    }

    probs
}

/// Calculate maximum drawdown with start and end indices
fn calculate_max_drawdown_details(equity: &[f64]) -> (f64, usize, usize) {
    let mut max_dd = 0.0;
    let mut peak = equity[0];
    let mut peak_idx = 0;
    let mut dd_start = 0;
    let mut dd_end = 0;

    for (i, &value) in equity.iter().enumerate() {
        if value > peak {
            peak = value;
            peak_idx = i;
        }

        let dd = (peak - value) / peak;
        if dd > max_dd {
            max_dd = dd;
            dd_start = peak_idx;
            dd_end = i;
        }
    }

    (max_dd, dd_start, dd_end)
}

/// Calculate approximate monthly returns
fn calculate_monthly_returns(equity: &[f64], days_per_month: usize) -> Vec<f64> {
    let mut returns = Vec::new();

    for chunk in equity.chunks(days_per_month) {
        if chunk.len() >= 2 {
            let ret = (chunk.last().unwrap() / chunk.first().unwrap()) - 1.0;
            returns.push(ret);
        }
    }

    returns
}
