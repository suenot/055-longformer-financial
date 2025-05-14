"""
Data Loading and Preprocessing for Longformer Financial Analysis

This module provides utilities for loading financial data from various sources
(Bybit, Yahoo Finance) and preparing it for Longformer model training.

Supported data sources:
- Bybit API for cryptocurrency data
- Yahoo Finance for stock market data

Features:
- Automatic feature engineering (returns, volatility, RSI, MACD)
- Sequence preparation for long context models
- Train/validation/test split utilities
"""

import numpy as np
import pandas as pd
import requests
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_bybit_data(
    symbol: str,
    interval: str = '1m',
    limit: int = 10000
) -> pd.DataFrame:
    """
    Load historical kline data from Bybit API.

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Kline interval ('1m', '5m', '15m', '1h', '4h', '1d')
        limit: Maximum number of candles to fetch (max 10000)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume, turnover

    Example:
        >>> df = load_bybit_data('BTCUSDT', interval='1m', limit=1000)
        >>> print(df.head())
    """
    url = "https://api.bybit.com/v5/market/kline"

    # Map interval to Bybit format
    interval_map = {
        '1m': '1',
        '3m': '3',
        '5m': '5',
        '15m': '15',
        '30m': '30',
        '1h': '60',
        '2h': '120',
        '4h': '240',
        '6h': '360',
        '12h': '720',
        '1d': 'D',
        '1w': 'W',
        '1M': 'M'
    }

    params = {
        'category': 'spot',
        'symbol': symbol,
        'interval': interval_map.get(interval, interval),
        'limit': min(limit, 10000)
    }

    logger.info(f"Fetching {symbol} data from Bybit (interval={interval}, limit={limit})")

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data['retCode'] != 0:
            raise ValueError(f"Bybit API error: {data['retMsg']}")

        candles = data['result']['list']

        df = pd.DataFrame(candles, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        # Convert types
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"Loaded {len(df)} candles for {symbol}")

        return df

    except requests.RequestException as e:
        logger.error(f"Failed to fetch data from Bybit: {e}")
        raise


def load_yahoo_data(
    symbol: str,
    period: str = '1y',
    interval: str = '1d'
) -> pd.DataFrame:
    """
    Load historical data from Yahoo Finance.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        period: Data period ('1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
        interval: Data interval ('1m', '5m', '15m', '1h', '1d', '1wk', '1mo')

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume

    Example:
        >>> df = load_yahoo_data('AAPL', period='1y', interval='1d')
        >>> print(df.head())
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required. Install with: pip install yfinance")

    logger.info(f"Fetching {symbol} data from Yahoo Finance (period={period}, interval={interval})")

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]

        # Rename 'date' or 'datetime' to 'timestamp'
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'timestamp'})
        elif 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'timestamp'})

        # Select and order columns
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in cols if c in df.columns]]

        logger.info(f"Loaded {len(df)} candles for {symbol}")

        return df

    except Exception as e:
        logger.error(f"Failed to fetch data from Yahoo Finance: {e}")
        raise


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        prices: Price series (typically close prices)
        period: RSI calculation period

    Returns:
        RSI values (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: Price series (typically close prices)
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line EMA period

    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR calculation period

    Returns:
        ATR values
    """
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.

    Args:
        prices: Price series
        period: Moving average period
        num_std: Number of standard deviations for bands

    Returns:
        Tuple of (upper band, middle band, lower band)
    """
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators and features to DataFrame.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with additional feature columns
    """
    df = df.copy()

    # Log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Volatility (rolling standard deviation of returns)
    df['volatility_20'] = df['log_return'].rolling(20).std()
    df['volatility_50'] = df['log_return'].rolling(50).std()

    # Volume features
    df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(50).mean()
    df['volume_std'] = df['volume'].rolling(20).std() / df['volume'].rolling(20).mean()

    # Price features
    df['price_ma_ratio_20'] = df['close'] / df['close'].rolling(20).mean()
    df['price_ma_ratio_50'] = df['close'] / df['close'].rolling(50).mean()
    df['price_ma_ratio_200'] = df['close'] / df['close'].rolling(200).mean()

    # Technical indicators
    df['rsi'] = calculate_rsi(df['close'], period=14)
    macd, signal, hist = calculate_macd(df['close'])
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_hist'] = hist
    df['atr'] = calculate_atr(df, period=14)

    # Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(df['close'])
    df['bb_position'] = (df['close'] - lower) / (upper - lower + 1e-10)

    # High-low range
    df['hl_range'] = (df['high'] - df['low']) / df['close']

    # Normalize features
    df['rsi_norm'] = (df['rsi'] - 50) / 50  # Normalize to [-1, 1]
    df['bb_position_norm'] = df['bb_position'] * 2 - 1  # Normalize to [-1, 1]

    return df


def prepare_timeseries_data(
    symbols: List[str],
    lookback: int = 4096,
    horizon: int = 24,
    source: str = 'bybit',
    features: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare time series data for Longformer model.

    Args:
        symbols: List of trading symbols
        lookback: Number of historical time steps
        horizon: Number of future steps to predict
        source: Data source ('bybit' or 'yahoo')
        features: List of feature columns to use (default: standard set)

    Returns:
        X: Feature array [n_samples, lookback, n_features]
        y: Target array [n_samples, horizon]

    Example:
        >>> X, y = prepare_timeseries_data(
        ...     symbols=['BTCUSDT', 'ETHUSDT'],
        ...     lookback=4096,
        ...     horizon=24,
        ...     source='bybit'
        ... )
        >>> print(f"X shape: {X.shape}, y shape: {y.shape}")
    """
    if features is None:
        features = [
            'log_return', 'volatility_20', 'volume_ma_ratio',
            'price_ma_ratio_50', 'rsi_norm', 'macd'
        ]

    all_features = []

    for symbol in symbols:
        logger.info(f"Loading data for {symbol}...")

        if source == 'bybit':
            df = load_bybit_data(symbol, interval='1m')
        else:
            df = load_yahoo_data(symbol, period='1y', interval='1d')

        # Add features
        df = add_features(df)
        all_features.append(df)

    # Merge all symbols
    if len(all_features) > 1:
        merged = pd.concat(all_features, axis=1, keys=symbols)
        # Select features for each symbol
        feature_cols = [(s, f) for s in symbols for f in features]
    else:
        merged = all_features[0]
        feature_cols = features

    # Drop NaN rows
    merged = merged.dropna()

    logger.info(f"Total data points after feature engineering: {len(merged)}")

    # Create sequences
    X, y = [], []

    for i in range(lookback, len(merged) - horizon):
        if len(all_features) > 1:
            x_seq = merged.iloc[i-lookback:i][feature_cols].values
        else:
            x_seq = merged.iloc[i-lookback:i][features].values
        X.append(x_seq)

        # Target: future returns for primary symbol
        if len(all_features) > 1:
            y_seq = merged.iloc[i:i+horizon][(symbols[0], 'log_return')].values
        else:
            y_seq = merged.iloc[i:i+horizon]['log_return'].values
        y.append(y_seq)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Handle any remaining NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"Prepared {len(X)} sequences with shape X={X.shape}, y={y.shape}")

    return X, y


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split data into train/validation/test sets chronologically.

    Args:
        X: Feature array
        y: Target array
        train_ratio: Proportion for training
        val_ratio: Proportion for validation

    Returns:
        Dictionary with 'train', 'val', 'test' splits
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        'train': (X[:train_end], y[:train_end]),
        'val': (X[train_end:val_end], y[train_end:val_end]),
        'test': (X[val_end:], y[val_end:])
    }

    logger.info(f"Split sizes: train={train_end}, val={val_end-train_end}, test={n-val_end}")

    return splits


if __name__ == '__main__':
    # Test data loading
    print("Testing data loading utilities...")

    # Test Bybit data loading
    try:
        df_bybit = load_bybit_data('BTCUSDT', interval='1m', limit=100)
        print(f"\nBybit data shape: {df_bybit.shape}")
        print(df_bybit.head())
    except Exception as e:
        print(f"Bybit test skipped: {e}")

    # Test feature engineering
    print("\nTesting feature engineering...")
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=500, freq='1min'),
        'open': np.random.randn(500).cumsum() + 100,
        'high': np.random.randn(500).cumsum() + 101,
        'low': np.random.randn(500).cumsum() + 99,
        'close': np.random.randn(500).cumsum() + 100,
        'volume': np.random.exponential(1000, 500)
    })
    sample_data['high'] = sample_data[['open', 'close']].max(axis=1) + np.abs(np.random.randn(500))
    sample_data['low'] = sample_data[['open', 'close']].min(axis=1) - np.abs(np.random.randn(500))

    df_features = add_features(sample_data)
    print(f"Features added: {list(df_features.columns)}")
    print(df_features.tail())

    print("\nAll data loading tests completed!")
