"""
Longformer for Financial Analysis

This module provides implementations of the Longformer architecture
adapted for financial document analysis and time series processing.

Components:
- model: Longformer model implementations
- data: Data loading utilities for Bybit and Yahoo Finance
- strategy: Backtesting and trading strategy utilities
"""

from .model import (
    LongformerSlidingWindowAttention,
    LongformerEncoderLayer,
    LongformerForTrading,
)
from .data import (
    load_bybit_data,
    load_yahoo_data,
    prepare_timeseries_data,
    calculate_rsi,
    calculate_macd,
)
from .strategy import (
    BacktestConfig,
    LongformerBacktester,
)

__version__ = "0.1.0"
__all__ = [
    "LongformerSlidingWindowAttention",
    "LongformerEncoderLayer",
    "LongformerForTrading",
    "load_bybit_data",
    "load_yahoo_data",
    "prepare_timeseries_data",
    "calculate_rsi",
    "calculate_macd",
    "BacktestConfig",
    "LongformerBacktester",
]
