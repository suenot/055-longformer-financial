//! Trading signal generation

use std::fmt;

/// Trading signal type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    /// Sell signal (close long or open short)
    Sell,
    /// Hold signal (maintain current position)
    Hold,
    /// Buy signal (open long or close short)
    Buy,
}

impl Signal {
    /// Convert from class index
    pub fn from_class(class: usize) -> Self {
        match class {
            0 => Signal::Sell,
            1 => Signal::Hold,
            _ => Signal::Buy,
        }
    }

    /// Convert to class index
    pub fn to_class(&self) -> usize {
        match self {
            Signal::Sell => 0,
            Signal::Hold => 1,
            Signal::Buy => 2,
        }
    }

    /// Check if signal is actionable (not hold)
    pub fn is_actionable(&self) -> bool {
        !matches!(self, Signal::Hold)
    }

    /// Get position direction (-1, 0, 1)
    pub fn direction(&self) -> i32 {
        match self {
            Signal::Sell => -1,
            Signal::Hold => 0,
            Signal::Buy => 1,
        }
    }
}

impl fmt::Display for Signal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Signal::Sell => write!(f, "SELL"),
            Signal::Hold => write!(f, "HOLD"),
            Signal::Buy => write!(f, "BUY"),
        }
    }
}

/// Signal generator with confidence thresholds
pub struct SignalGenerator {
    /// Minimum confidence for buy signal
    buy_threshold: f64,
    /// Minimum confidence for sell signal
    sell_threshold: f64,
    /// Lookback period for signal confirmation
    confirmation_periods: usize,
}

impl SignalGenerator {
    /// Create a new signal generator with default thresholds
    pub fn new() -> Self {
        Self {
            buy_threshold: 0.6,
            sell_threshold: 0.6,
            confirmation_periods: 1,
        }
    }

    /// Create with custom thresholds
    pub fn with_thresholds(buy_threshold: f64, sell_threshold: f64) -> Self {
        Self {
            buy_threshold,
            sell_threshold,
            confirmation_periods: 1,
        }
    }

    /// Set confirmation periods
    pub fn with_confirmation(mut self, periods: usize) -> Self {
        self.confirmation_periods = periods;
        self
    }

    /// Generate signal from model probabilities
    ///
    /// # Arguments
    /// * `probs` - Probabilities for [sell, hold, buy]
    ///
    /// # Returns
    /// Trading signal based on thresholds
    pub fn generate(&self, probs: &[f64; 3]) -> Signal {
        let sell_prob = probs[0];
        let buy_prob = probs[2];

        if buy_prob >= self.buy_threshold && buy_prob > sell_prob {
            Signal::Buy
        } else if sell_prob >= self.sell_threshold && sell_prob > buy_prob {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    /// Generate signals from a series of probabilities
    ///
    /// Applies optional signal confirmation (consecutive signals required)
    pub fn generate_series(&self, probs_series: &[[f64; 3]]) -> Vec<Signal> {
        if probs_series.is_empty() {
            return Vec::new();
        }

        let raw_signals: Vec<Signal> = probs_series
            .iter()
            .map(|p| self.generate(p))
            .collect();

        if self.confirmation_periods <= 1 {
            return raw_signals;
        }

        // Apply confirmation filter
        let mut confirmed = vec![Signal::Hold; raw_signals.len()];

        for i in (self.confirmation_periods - 1)..raw_signals.len() {
            let signal = raw_signals[i];
            if signal == Signal::Hold {
                confirmed[i] = Signal::Hold;
                continue;
            }

            // Check if previous periods have the same signal
            let mut all_same = true;
            for j in 0..self.confirmation_periods {
                if raw_signals[i - j] != signal {
                    all_same = false;
                    break;
                }
            }

            confirmed[i] = if all_same { signal } else { Signal::Hold };
        }

        confirmed
    }

    /// Calculate signal statistics
    pub fn signal_stats(&self, signals: &[Signal]) -> SignalStats {
        let total = signals.len();
        let buys = signals.iter().filter(|&&s| s == Signal::Buy).count();
        let sells = signals.iter().filter(|&&s| s == Signal::Sell).count();
        let holds = total - buys - sells;

        SignalStats {
            total,
            buys,
            sells,
            holds,
            buy_ratio: buys as f64 / total as f64,
            sell_ratio: sells as f64 / total as f64,
            hold_ratio: holds as f64 / total as f64,
        }
    }
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about generated signals
#[derive(Debug, Clone)]
pub struct SignalStats {
    /// Total number of signals
    pub total: usize,
    /// Number of buy signals
    pub buys: usize,
    /// Number of sell signals
    pub sells: usize,
    /// Number of hold signals
    pub holds: usize,
    /// Ratio of buy signals
    pub buy_ratio: f64,
    /// Ratio of sell signals
    pub sell_ratio: f64,
    /// Ratio of hold signals
    pub hold_ratio: f64,
}

impl fmt::Display for SignalStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Signals: {} total | Buy: {} ({:.1}%) | Sell: {} ({:.1}%) | Hold: {} ({:.1}%)",
            self.total,
            self.buys,
            self.buy_ratio * 100.0,
            self.sells,
            self.sell_ratio * 100.0,
            self.holds,
            self.hold_ratio * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_conversion() {
        assert_eq!(Signal::from_class(0), Signal::Sell);
        assert_eq!(Signal::from_class(1), Signal::Hold);
        assert_eq!(Signal::from_class(2), Signal::Buy);

        assert_eq!(Signal::Sell.to_class(), 0);
        assert_eq!(Signal::Hold.to_class(), 1);
        assert_eq!(Signal::Buy.to_class(), 2);
    }

    #[test]
    fn test_signal_generator() {
        let gen = SignalGenerator::with_thresholds(0.5, 0.5);

        // Strong buy signal
        assert_eq!(gen.generate(&[0.1, 0.2, 0.7]), Signal::Buy);

        // Strong sell signal
        assert_eq!(gen.generate(&[0.7, 0.2, 0.1]), Signal::Sell);

        // Neutral - hold
        assert_eq!(gen.generate(&[0.35, 0.3, 0.35]), Signal::Hold);

        // Below threshold - hold
        assert_eq!(gen.generate(&[0.4, 0.3, 0.3]), Signal::Hold);
    }

    #[test]
    fn test_signal_confirmation() {
        let gen = SignalGenerator::with_thresholds(0.5, 0.5)
            .with_confirmation(2);

        let probs = [
            [0.1, 0.2, 0.7], // Buy
            [0.1, 0.2, 0.7], // Buy (confirmed)
            [0.7, 0.2, 0.1], // Sell
            [0.1, 0.2, 0.7], // Buy (not confirmed - previous was sell)
        ];

        let signals = gen.generate_series(&probs);

        assert_eq!(signals[0], Signal::Hold); // First can't be confirmed
        assert_eq!(signals[1], Signal::Buy);  // Second buy confirmed
        assert_eq!(signals[2], Signal::Hold); // Sell not confirmed
        assert_eq!(signals[3], Signal::Hold); // Buy not confirmed
    }

    #[test]
    fn test_signal_stats() {
        let gen = SignalGenerator::new();
        let signals = vec![Signal::Buy, Signal::Buy, Signal::Sell, Signal::Hold, Signal::Hold];

        let stats = gen.signal_stats(&signals);

        assert_eq!(stats.total, 5);
        assert_eq!(stats.buys, 2);
        assert_eq!(stats.sells, 1);
        assert_eq!(stats.holds, 2);
    }
}
