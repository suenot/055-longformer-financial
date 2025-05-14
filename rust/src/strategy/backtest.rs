//! Backtesting engine for trading strategies

use std::fmt;
use super::signals::Signal;

/// Configuration for backtesting
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Position size as fraction of capital (0.0 - 1.0)
    pub position_size: f64,
    /// Stop loss threshold (e.g., 0.02 = 2%)
    pub stop_loss: f64,
    /// Take profit threshold (e.g., 0.04 = 4%)
    pub take_profit: f64,
    /// Commission per trade (e.g., 0.001 = 0.1%)
    pub commission: f64,
    /// Slippage per trade (e.g., 0.0005 = 0.05%)
    pub slippage: f64,
    /// Allow short selling
    pub allow_short: bool,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            position_size: 0.02,
            stop_loss: 0.02,
            take_profit: 0.04,
            commission: 0.001,
            slippage: 0.0005,
            allow_short: false,
        }
    }
}

/// Position state during backtest
#[derive(Debug, Clone, Copy)]
enum Position {
    /// No position
    Flat,
    /// Long position with entry price and quantity
    Long { entry_price: f64, quantity: f64 },
    /// Short position with entry price and quantity
    Short { entry_price: f64, quantity: f64 },
}

/// Trade record
#[derive(Debug, Clone)]
pub struct Trade {
    /// Entry timestamp/index
    pub entry_idx: usize,
    /// Exit timestamp/index
    pub exit_idx: usize,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Trade direction (1 = long, -1 = short)
    pub direction: i32,
    /// Profit/loss amount
    pub pnl: f64,
    /// Return percentage
    pub return_pct: f64,
    /// Exit reason
    pub exit_reason: ExitReason,
}

/// Reason for exiting a trade
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitReason {
    Signal,
    StopLoss,
    TakeProfit,
    EndOfData,
}

impl fmt::Display for ExitReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExitReason::Signal => write!(f, "Signal"),
            ExitReason::StopLoss => write!(f, "Stop Loss"),
            ExitReason::TakeProfit => write!(f, "Take Profit"),
            ExitReason::EndOfData => write!(f, "End of Data"),
        }
    }
}

/// Backtest results
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Total return percentage
    pub total_return: f64,
    /// Annualized return (assuming 252 trading days)
    pub annualized_return: f64,
    /// Sharpe ratio (assuming 0% risk-free rate)
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Calmar ratio (return / max drawdown)
    pub calmar_ratio: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Total number of trades
    pub num_trades: usize,
    /// Final portfolio value
    pub final_value: f64,
    /// All trades
    pub trades: Vec<Trade>,
    /// Equity curve (portfolio value over time)
    pub equity_curve: Vec<f64>,
    /// Daily returns
    pub daily_returns: Vec<f64>,
}

impl fmt::Display for BacktestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Backtest Results ===")?;
        writeln!(f, "Total Return:      {:.2}%", self.total_return * 100.0)?;
        writeln!(f, "Annualized Return: {:.2}%", self.annualized_return * 100.0)?;
        writeln!(f, "Sharpe Ratio:      {:.3}", self.sharpe_ratio)?;
        writeln!(f, "Sortino Ratio:     {:.3}", self.sortino_ratio)?;
        writeln!(f, "Max Drawdown:      {:.2}%", self.max_drawdown * 100.0)?;
        writeln!(f, "Calmar Ratio:      {:.3}", self.calmar_ratio)?;
        writeln!(f, "Win Rate:          {:.1}%", self.win_rate * 100.0)?;
        writeln!(f, "Profit Factor:     {:.3}", self.profit_factor)?;
        writeln!(f, "Number of Trades:  {}", self.num_trades)?;
        writeln!(f, "Final Value:       ${:.2}", self.final_value)?;
        Ok(())
    }
}

/// Backtesting engine
pub struct Backtester {
    config: BacktestConfig,
}

impl Backtester {
    /// Create a new backtester with the given configuration
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run backtest with given prices and signals
    ///
    /// # Arguments
    /// * `prices` - Array of closing prices
    /// * `signals` - Array of trading signals
    ///
    /// # Returns
    /// Backtest results
    pub fn run(&self, prices: &[f64], signals: &[Signal]) -> BacktestResult {
        assert_eq!(prices.len(), signals.len(), "Prices and signals must have same length");

        let mut capital = self.config.initial_capital;
        let mut position = Position::Flat;
        let mut trades: Vec<Trade> = Vec::new();
        let mut equity_curve = vec![capital];
        let mut peak = capital;
        let mut max_drawdown = 0.0;

        for i in 0..prices.len() {
            let price = prices[i];
            let signal = signals[i];

            // Calculate current portfolio value
            let portfolio_value = match position {
                Position::Flat => capital,
                Position::Long { entry_price, quantity } => {
                    capital + quantity * (price - entry_price)
                }
                Position::Short { entry_price, quantity } => {
                    capital + quantity * (entry_price - price)
                }
            };

            // Update max drawdown
            if portfolio_value > peak {
                peak = portfolio_value;
            }
            let drawdown = (peak - portfolio_value) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }

            // Check for stop loss / take profit
            let (should_exit, exit_reason) = match position {
                Position::Long { entry_price, .. } => {
                    let ret = (price - entry_price) / entry_price;
                    if ret <= -self.config.stop_loss {
                        (true, ExitReason::StopLoss)
                    } else if ret >= self.config.take_profit {
                        (true, ExitReason::TakeProfit)
                    } else {
                        (false, ExitReason::Signal)
                    }
                }
                Position::Short { entry_price, .. } => {
                    let ret = (entry_price - price) / entry_price;
                    if ret <= -self.config.stop_loss {
                        (true, ExitReason::StopLoss)
                    } else if ret >= self.config.take_profit {
                        (true, ExitReason::TakeProfit)
                    } else {
                        (false, ExitReason::Signal)
                    }
                }
                Position::Flat => (false, ExitReason::Signal),
            };

            // Handle exits
            if should_exit || (signal.is_actionable() && !matches!(position, Position::Flat)) {
                match position {
                    Position::Long { entry_price, quantity } => {
                        let exit_price = price * (1.0 - self.config.slippage);
                        let gross_pnl = quantity * (exit_price - entry_price);
                        let commission = exit_price * quantity * self.config.commission;
                        let net_pnl = gross_pnl - commission;

                        capital += net_pnl;

                        trades.push(Trade {
                            entry_idx: trades.len(), // Simplified
                            exit_idx: i,
                            entry_price,
                            exit_price,
                            direction: 1,
                            pnl: net_pnl,
                            return_pct: net_pnl / (entry_price * quantity),
                            exit_reason: if should_exit { exit_reason } else { ExitReason::Signal },
                        });

                        position = Position::Flat;
                    }
                    Position::Short { entry_price, quantity } => {
                        let exit_price = price * (1.0 + self.config.slippage);
                        let gross_pnl = quantity * (entry_price - exit_price);
                        let commission = exit_price * quantity * self.config.commission;
                        let net_pnl = gross_pnl - commission;

                        capital += net_pnl;

                        trades.push(Trade {
                            entry_idx: trades.len(),
                            exit_idx: i,
                            entry_price,
                            exit_price,
                            direction: -1,
                            pnl: net_pnl,
                            return_pct: net_pnl / (entry_price * quantity),
                            exit_reason: if should_exit { exit_reason } else { ExitReason::Signal },
                        });

                        position = Position::Flat;
                    }
                    Position::Flat => {}
                }
            }

            // Handle new entries
            if matches!(position, Position::Flat) {
                match signal {
                    Signal::Buy => {
                        let position_value = capital * self.config.position_size;
                        let entry_price = price * (1.0 + self.config.slippage);
                        let commission = position_value * self.config.commission;
                        let quantity = (position_value - commission) / entry_price;

                        position = Position::Long { entry_price, quantity };
                    }
                    Signal::Sell if self.config.allow_short => {
                        let position_value = capital * self.config.position_size;
                        let entry_price = price * (1.0 - self.config.slippage);
                        let commission = position_value * self.config.commission;
                        let quantity = (position_value - commission) / entry_price;

                        position = Position::Short { entry_price, quantity };
                    }
                    _ => {}
                }
            }

            equity_curve.push(portfolio_value);
        }

        // Close any remaining position at end
        if let Position::Long { entry_price, quantity } | Position::Short { entry_price, quantity } = position {
            let price = *prices.last().unwrap();
            let is_long = matches!(position, Position::Long { .. });
            let exit_price = if is_long {
                price * (1.0 - self.config.slippage)
            } else {
                price * (1.0 + self.config.slippage)
            };

            let gross_pnl = if is_long {
                quantity * (exit_price - entry_price)
            } else {
                quantity * (entry_price - exit_price)
            };
            let commission = exit_price * quantity * self.config.commission;
            let net_pnl = gross_pnl - commission;

            capital += net_pnl;

            trades.push(Trade {
                entry_idx: trades.len(),
                exit_idx: prices.len() - 1,
                entry_price,
                exit_price,
                direction: if is_long { 1 } else { -1 },
                pnl: net_pnl,
                return_pct: net_pnl / (entry_price * quantity),
                exit_reason: ExitReason::EndOfData,
            });
        }

        // Calculate metrics
        let total_return = (capital - self.config.initial_capital) / self.config.initial_capital;

        // Daily returns from equity curve
        let daily_returns: Vec<f64> = equity_curve.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let mean_return = if daily_returns.is_empty() {
            0.0
        } else {
            daily_returns.iter().sum::<f64>() / daily_returns.len() as f64
        };

        let std_return = if daily_returns.len() > 1 {
            let variance: f64 = daily_returns.iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>() / (daily_returns.len() - 1) as f64;
            variance.sqrt()
        } else {
            1.0
        };

        let downside_returns: Vec<f64> = daily_returns.iter()
            .filter(|&&r| r < 0.0)
            .cloned()
            .collect();

        let downside_std = if downside_returns.len() > 1 {
            let variance: f64 = downside_returns.iter()
                .map(|r| r.powi(2))
                .sum::<f64>() / (downside_returns.len() - 1) as f64;
            variance.sqrt()
        } else {
            1.0
        };

        let annualized_return = mean_return * 252.0;
        let annualized_std = std_return * (252.0_f64).sqrt();
        let sharpe_ratio = if annualized_std > 0.0 {
            annualized_return / annualized_std
        } else {
            0.0
        };

        let sortino_ratio = if downside_std > 0.0 {
            annualized_return / (downside_std * (252.0_f64).sqrt())
        } else {
            0.0
        };

        let calmar_ratio = if max_drawdown > 0.0 {
            annualized_return / max_drawdown
        } else {
            0.0
        };

        let winning_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losing_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl < 0.0).collect();

        let win_rate = if !trades.is_empty() {
            winning_trades.len() as f64 / trades.len() as f64
        } else {
            0.0
        };

        let gross_profits: f64 = winning_trades.iter().map(|t| t.pnl).sum();
        let gross_losses: f64 = losing_trades.iter().map(|t| t.pnl.abs()).sum();

        let profit_factor = if gross_losses > 0.0 {
            gross_profits / gross_losses
        } else if gross_profits > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        BacktestResult {
            total_return,
            annualized_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            win_rate,
            profit_factor,
            num_trades: trades.len(),
            final_value: capital,
            trades,
            equity_curve,
            daily_returns,
        }
    }

    /// Get backtester configuration
    pub fn config(&self) -> &BacktestConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtest_basic() {
        let config = BacktestConfig {
            initial_capital: 10000.0,
            position_size: 0.5,
            stop_loss: 0.1,
            take_profit: 0.1,
            commission: 0.0,
            slippage: 0.0,
            allow_short: false,
        };

        let backtester = Backtester::new(config);

        // Simple uptrend scenario
        let prices: Vec<f64> = (0..100)
            .map(|i| 100.0 + i as f64 * 0.5)
            .collect();

        let signals: Vec<Signal> = prices.iter().enumerate()
            .map(|(i, _)| {
                if i == 0 { Signal::Buy }
                else if i == 99 { Signal::Sell }
                else { Signal::Hold }
            })
            .collect();

        let result = backtester.run(&prices, &signals);

        assert!(result.total_return > 0.0, "Should profit in uptrend");
        assert_eq!(result.num_trades, 1);
    }

    #[test]
    fn test_stop_loss() {
        let config = BacktestConfig {
            initial_capital: 10000.0,
            position_size: 0.5,
            stop_loss: 0.05,
            take_profit: 0.5,
            commission: 0.0,
            slippage: 0.0,
            allow_short: false,
        };

        let backtester = Backtester::new(config);

        // Price drops 10%
        let prices = vec![100.0, 95.0, 90.0, 85.0, 80.0];
        let signals = vec![Signal::Buy, Signal::Hold, Signal::Hold, Signal::Hold, Signal::Hold];

        let result = backtester.run(&prices, &signals);

        assert!(!result.trades.is_empty());
        assert_eq!(result.trades[0].exit_reason, ExitReason::StopLoss);
    }

    #[test]
    fn test_take_profit() {
        let config = BacktestConfig {
            initial_capital: 10000.0,
            position_size: 0.5,
            stop_loss: 0.5,
            take_profit: 0.05,
            commission: 0.0,
            slippage: 0.0,
            allow_short: false,
        };

        let backtester = Backtester::new(config);

        // Price rises 10%
        let prices = vec![100.0, 105.0, 110.0, 115.0, 120.0];
        let signals = vec![Signal::Buy, Signal::Hold, Signal::Hold, Signal::Hold, Signal::Hold];

        let result = backtester.run(&prices, &signals);

        assert!(!result.trades.is_empty());
        assert_eq!(result.trades[0].exit_reason, ExitReason::TakeProfit);
    }

    #[test]
    fn test_metrics_calculation() {
        let config = BacktestConfig::default();
        let backtester = Backtester::new(config);

        // Generate some price data
        let prices: Vec<f64> = (0..200)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0 + i as f64 * 0.2)
            .collect();

        let signals: Vec<Signal> = prices.iter().enumerate()
            .map(|(i, _)| {
                if i % 20 == 0 { Signal::Buy }
                else if i % 20 == 10 { Signal::Sell }
                else { Signal::Hold }
            })
            .collect();

        let result = backtester.run(&prices, &signals);

        // Check that metrics are reasonable
        assert!(result.max_drawdown >= 0.0 && result.max_drawdown <= 1.0);
        assert!(result.win_rate >= 0.0 && result.win_rate <= 1.0);
        assert!(!result.equity_curve.is_empty());
    }
}
