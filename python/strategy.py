"""
Backtesting and Trading Strategy for Longformer Models

This module provides utilities for backtesting Longformer-based trading strategies
and evaluating performance metrics.

Features:
- Signal generation from model predictions
- Full backtest simulation with transaction costs
- Performance metrics (Sharpe, Sortino, Calmar, etc.)
- Visualization utilities
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    max_position_size: float = 1.0  # Maximum position as fraction of capital
    min_position_change: float = 0.01  # Minimum position change to execute
    risk_per_trade: float = 0.02  # 2% risk per trade (for risk management)


class LongformerBacktester:
    """
    Backtesting engine for Longformer trading strategies.

    This class handles:
    - Signal generation from model predictions
    - Trade execution simulation
    - Performance metric calculation
    - Result visualization

    Args:
        model: Trained Longformer model
        config: Backtest configuration
    """

    def __init__(
        self,
        model: nn.Module,
        config: BacktestConfig = BacktestConfig()
    ):
        self.model = model
        self.config = config
        self.model.eval()

    @torch.no_grad()
    def generate_signals(
        self,
        data: np.ndarray,
        threshold: float = 0.001
    ) -> np.ndarray:
        """
        Generate trading signals from model predictions.

        Args:
            data: Input features [n_samples, seq_len, n_features]
            threshold: Minimum predicted return to trigger a signal

        Returns:
            signals: Array of signals {-1, 0, 1} for each sample
        """
        self.model.eval()

        # Process in batches if data is large
        batch_size = 32
        all_signals = []

        for i in range(0, len(data), batch_size):
            batch = torch.tensor(data[i:i+batch_size], dtype=torch.float32)

            if torch.cuda.is_available() and next(self.model.parameters()).is_cuda:
                batch = batch.cuda()

            predictions = self.model(batch)

            if predictions.dim() > 1:
                # Use first prediction step
                pred_returns = predictions[:, 0].cpu().numpy()
            else:
                pred_returns = predictions.cpu().numpy()

            # Generate signals based on threshold
            signals = np.zeros_like(pred_returns)
            signals[pred_returns > threshold] = 1  # Long signal
            signals[pred_returns < -threshold] = -1  # Short signal

            all_signals.append(signals)

        return np.concatenate(all_signals)

    def run_backtest(
        self,
        data: np.ndarray,
        prices: np.ndarray,
        timestamps: pd.DatetimeIndex,
        threshold: float = 0.001
    ) -> Dict:
        """
        Run full backtest simulation.

        Args:
            data: Feature data [n_samples, seq_len, n_features]
            prices: Price series aligned with predictions [n_samples]
            timestamps: Timestamps for the backtest period [n_samples]
            threshold: Signal generation threshold

        Returns:
            Dictionary containing:
            - equity_curve: Portfolio value over time
            - positions: Position size over time
            - returns: Period returns
            - trades: List of trade records
            - timestamps: Time index
            - metrics: Performance metrics dictionary
        """
        logger.info("Generating signals...")
        signals = self.generate_signals(data, threshold=threshold)

        logger.info("Running backtest simulation...")

        # Initialize tracking variables
        capital = self.config.initial_capital
        position = 0.0
        entry_price = 0.0

        # Results tracking
        equity_curve = [capital]
        positions = [0.0]
        returns = []
        trades = []

        for i in range(len(signals)):
            current_price = prices[i]
            signal = signals[i]

            # Calculate target position
            target_position = signal * self.config.max_position_size
            position_change = target_position - position

            # Execute trade if position change is significant
            if abs(position_change) > self.config.min_position_change:
                # Calculate transaction costs
                trade_value = abs(position_change) * capital
                total_costs = trade_value * (
                    self.config.transaction_cost + self.config.slippage
                )

                # Record trade
                trades.append({
                    'timestamp': timestamps[i],
                    'price': current_price,
                    'signal': signal,
                    'position_before': position,
                    'position_after': target_position,
                    'position_change': position_change,
                    'costs': total_costs
                })

                # Apply costs
                capital -= total_costs

                # Update position
                position = target_position
                entry_price = current_price

            # Calculate P&L for the period
            if i > 0 and position != 0:
                price_return = (current_price - prices[i-1]) / prices[i-1]
                period_pnl = position * capital * price_return
                capital += period_pnl
                period_return = period_pnl / equity_curve[-1]
            else:
                period_return = 0.0

            returns.append(period_return)
            equity_curve.append(capital)
            positions.append(position)

        # Calculate performance metrics
        returns = np.array(returns)
        equity_curve = np.array(equity_curve)
        metrics = self._calculate_metrics(returns, equity_curve, trades)

        logger.info(f"Backtest complete. Final capital: ${capital:,.2f}")

        return {
            'equity_curve': equity_curve,
            'positions': positions,
            'returns': returns,
            'trades': trades,
            'timestamps': timestamps,
            'signals': signals,
            'metrics': metrics
        }

    def _calculate_metrics(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray,
        trades: List[Dict]
    ) -> Dict:
        """
        Calculate comprehensive performance metrics.

        Args:
            returns: Period returns array
            equity_curve: Portfolio value over time
            trades: List of trade records

        Returns:
            Dictionary of performance metrics
        """
        # Basic return metrics
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0

        # Risk metrics
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0.0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = np.sqrt(252) * returns.mean() / downside_returns.std()
        else:
            sortino_ratio = sharpe_ratio

        # Maximum drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()

        # Calmar ratio
        if max_drawdown < 0:
            calmar_ratio = annualized_return / abs(max_drawdown)
        else:
            calmar_ratio = float('inf')

        # Trade statistics
        n_trades = len(trades)
        if n_trades > 0:
            # Estimate winning trades based on position changes and price movements
            total_costs = sum(t['costs'] for t in trades)
            avg_trade_cost = total_costs / n_trades
        else:
            total_costs = 0
            avg_trade_cost = 0

        # Volatility
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0

        # Profit factor
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        profit_factor = positive_returns / (negative_returns + 1e-10)

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'profit_factor': profit_factor,
            'n_trades': n_trades,
            'total_costs': total_costs,
            'avg_trade_cost': avg_trade_cost,
            'initial_capital': equity_curve[0],
            'final_capital': equity_curve[-1]
        }

    def plot_results(
        self,
        results: Dict,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10)
    ):
        """
        Visualize backtest results.

        Args:
            results: Dictionary from run_backtest()
            save_path: Optional path to save the figure
            figsize: Figure size (width, height)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed. Skipping visualization.")
            return

        fig, axes = plt.subplots(4, 1, figsize=figsize)

        timestamps = results['timestamps']

        # 1. Equity Curve
        axes[0].plot(timestamps, results['equity_curve'][1:], 'b-', linewidth=1)
        axes[0].fill_between(
            timestamps,
            results['equity_curve'][1:],
            results['equity_curve'][0],
            alpha=0.3
        )
        axes[0].set_title('Equity Curve (Longformer Strategy)', fontsize=12)
        axes[0].set_ylabel('Capital ($)')
        axes[0].grid(True, alpha=0.3)
        axes[0].ticklabel_format(style='plain', axis='y')

        # 2. Positions
        positions = np.array(results['positions'][1:])
        axes[1].fill_between(
            timestamps, positions, 0,
            where=positions > 0, color='green', alpha=0.5, label='Long'
        )
        axes[1].fill_between(
            timestamps, positions, 0,
            where=positions < 0, color='red', alpha=0.5, label='Short'
        )
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_title('Position Size', fontsize=12)
        axes[1].set_ylabel('Position')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        # 3. Drawdown
        equity = np.array(results['equity_curve'][1:])
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100
        axes[2].fill_between(timestamps, drawdown, 0, color='red', alpha=0.5)
        axes[2].set_title('Drawdown', fontsize=12)
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].grid(True, alpha=0.3)

        # 4. Cumulative Returns
        cum_returns = (1 + np.array(results['returns'])).cumprod() - 1
        axes[3].plot(timestamps, cum_returns * 100, 'b-', linewidth=1)
        axes[3].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes[3].set_title('Cumulative Returns', fontsize=12)
        axes[3].set_ylabel('Return (%)')
        axes[3].set_xlabel('Date')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        plt.show()

        # Print metrics summary
        self._print_metrics(results['metrics'])

    def _print_metrics(self, metrics: Dict):
        """Print formatted metrics summary."""
        print("\n" + "=" * 60)
        print("LONGFORMER BACKTEST RESULTS")
        print("=" * 60)

        print("\nReturn Metrics:")
        print(f"  Total Return:       {metrics['total_return']*100:>10.2f}%")
        print(f"  Annualized Return:  {metrics['annualized_return']*100:>10.2f}%")
        print(f"  Volatility:         {metrics['volatility']*100:>10.2f}%")

        print("\nRisk-Adjusted Metrics:")
        print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:>10.4f}")
        print(f"  Sortino Ratio:      {metrics['sortino_ratio']:>10.4f}")
        print(f"  Calmar Ratio:       {metrics['calmar_ratio']:>10.4f}")
        print(f"  Max Drawdown:       {metrics['max_drawdown']*100:>10.2f}%")

        print("\nTrade Statistics:")
        print(f"  Number of Trades:   {metrics['n_trades']:>10d}")
        print(f"  Total Costs:        ${metrics['total_costs']:>9.2f}")
        print(f"  Profit Factor:      {metrics['profit_factor']:>10.4f}")

        print("\nCapital:")
        print(f"  Initial Capital:    ${metrics['initial_capital']:>9,.2f}")
        print(f"  Final Capital:      ${metrics['final_capital']:>9,.2f}")

        print("=" * 60)


class LongformerTrainer:
    """
    Training pipeline for Longformer trading models.

    Args:
        model: Longformer model to train
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        warmup_steps: Number of warmup steps
        device: Device to train on ('cuda' or 'cpu')
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.base_lr = learning_rate

        # Select loss function based on output type
        if hasattr(model, 'output_type'):
            if model.output_type == 'regression':
                self.criterion = nn.MSELoss()
            elif model.output_type == 'classification':
                self.criterion = nn.CrossEntropyLoss()
            elif model.output_type == 'allocation':
                self.criterion = self._sharpe_loss
            else:
                self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.MSELoss()

    def _get_lr_multiplier(self) -> float:
        """Calculate learning rate multiplier for warmup."""
        if self.current_step < self.warmup_steps:
            return self.current_step / max(1, self.warmup_steps)
        return 1.0

    def _sharpe_loss(
        self,
        allocations: torch.Tensor,
        returns: torch.Tensor
    ) -> torch.Tensor:
        """Differentiable Sharpe ratio loss for portfolio optimization."""
        portfolio_returns = allocations * returns
        mean_ret = portfolio_returns.mean()
        std_ret = portfolio_returns.std() + 1e-8
        return -mean_ret / std_ret  # Negative for minimization

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader
    ) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Update learning rate for warmup
            lr_mult = self._get_lr_multiplier()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr * lr_mult

            self.optimizer.zero_grad()

            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            self.current_step += 1

        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        all_preds = []
        all_targets = []
        total_loss = 0.0

        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)

            total_loss += loss.item()
            all_preds.append(predictions.cpu())
            all_targets.append(batch_y.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        metrics = {'loss': total_loss / len(val_loader)}

        # Calculate additional metrics based on task type
        if hasattr(self.model, 'output_type'):
            if self.model.output_type == 'regression':
                mse = ((all_preds - all_targets) ** 2).mean().item()
                mae = (all_preds - all_targets).abs().mean().item()
                metrics['mse'] = mse
                metrics['mae'] = mae

                # Directional accuracy
                pred_dir = (all_preds[:, 0] > 0).float()
                true_dir = (all_targets[:, 0] > 0).float()
                metrics['direction_accuracy'] = (pred_dir == true_dir).float().mean().item()

            elif self.model.output_type == 'classification':
                pred_classes = all_preds.argmax(dim=1)
                metrics['accuracy'] = (pred_classes == all_targets).float().mean().item()

        return metrics

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 100,
        patience: int = 10,
        save_path: Optional[str] = None
    ) -> Dict[str, List]:
        """
        Full training loop with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            save_path: Path to save best model

        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_metrics'].append(val_metrics)

            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_metrics['loss']:.6f} | "
                f"Metrics: {val_metrics}"
            )

            # Early stopping check
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0

                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    logger.info(f"Saved best model to {save_path}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        return history


if __name__ == '__main__':
    # Test the backtester with dummy data
    print("Testing Longformer Backtester...")

    # Create a simple model for testing
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(6, 1)

        def forward(self, x):
            return self.fc(x[:, -1, :])

    model = DummyModel()

    # Create dummy data
    n_samples = 100
    seq_len = 10
    n_features = 6

    data = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    prices = 100 + np.random.randn(n_samples).cumsum()
    timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='1min')

    # Run backtest
    config = BacktestConfig(initial_capital=10000)
    backtester = LongformerBacktester(model, config)
    results = backtester.run_backtest(data, prices, timestamps)

    print(f"\nBacktest completed!")
    print(f"Final capital: ${results['metrics']['final_capital']:,.2f}")
    print(f"Total return: {results['metrics']['total_return']*100:.2f}%")
    print(f"Sharpe ratio: {results['metrics']['sharpe_ratio']:.4f}")
    print(f"Max drawdown: {results['metrics']['max_drawdown']*100:.2f}%")

    print("\nAll tests passed!")
