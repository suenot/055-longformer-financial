# Chapter 57: Longformer for Financial Analysis

This chapter explores **Longformer**, a transformer architecture that uses **sliding window attention** combined with **global attention** to efficiently process long financial documents and time series data with O(n) complexity instead of the standard O(n²).

<p align="center">
<img src="https://i.imgur.com/9YKzL8B.png" width="70%">
</p>

## Contents

1. [Introduction to Longformer](#introduction-to-longformer)
    * [The Long Document Challenge](#the-long-document-challenge)
    * [Key Innovation: Hybrid Attention](#key-innovation-hybrid-attention)
    * [Why Longformer for Finance](#why-longformer-for-finance)
2. [Mathematical Foundation](#mathematical-foundation)
    * [Standard Self-Attention](#standard-self-attention)
    * [Sliding Window Attention](#sliding-window-attention)
    * [Dilated Sliding Window](#dilated-sliding-window)
    * [Global Attention](#global-attention)
3. [Longformer Architecture](#longformer-architecture)
    * [Attention Pattern Design](#attention-pattern-design)
    * [Linear Complexity Analysis](#linear-complexity-analysis)
    * [Implementation Details](#implementation-details)
4. [Financial Applications](#financial-applications)
    * [Long Financial Document Analysis](#long-financial-document-analysis)
    * [Extended Time Series Processing](#extended-time-series-processing)
    * [Multi-Asset Correlation Analysis](#multi-asset-correlation-analysis)
5. [Practical Examples](#practical-examples)
    * [01: Data Preparation](#01-data-preparation)
    * [02: Longformer Model](#02-longformer-model)
    * [03: Training Pipeline](#03-training-pipeline)
    * [04: Backtesting Strategy](#04-backtesting-strategy)
6. [Rust Implementation](#rust-implementation)
7. [Python Implementation](#python-implementation)
8. [Best Practices](#best-practices)
9. [Resources](#resources)

## Introduction to Longformer

### The Long Document Challenge

Standard transformers compute attention between all pairs of tokens, resulting in O(n²) complexity. For financial applications, this becomes problematic:

```
Document Processing Challenge:

Financial Document Lengths:
- SEC 10-K Filing:     ~50,000 tokens
- Earnings Call Transcript: ~10,000 tokens
- Research Report:     ~5,000 tokens
- News Article:        ~1,000 tokens

Standard Transformer (BERT):
- Max sequence: 512 tokens
- 10-K filing would need: 100+ chunks!
- Context is lost between chunks

Trading Time Series:
- 1 month of minute data: 43,200 data points
- Standard attention: 43,200² = 1.86 billion operations!
```

### Key Innovation: Hybrid Attention

Longformer introduces a novel combination of two attention patterns:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LONGFORMER ATTENTION PATTERNS                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. SLIDING WINDOW ATTENTION (Local)                                 │
│     Each token attends to w tokens on each side                      │
│                                                                       │
│     Token:    [1] [2] [3] [4] [5] [6] [7] [8] [9] [10]              │
│     Window=2:          ←─[4]─→                                       │
│                      attends to [2,3,4,5,6]                          │
│                                                                       │
│  2. GLOBAL ATTENTION (Task-specific)                                 │
│     Certain tokens attend to ALL positions                           │
│                                                                       │
│     [CLS] token for classification:                                  │
│     [CLS] ←→ [all tokens]                                            │
│                                                                       │
│     Combined Pattern:                                                │
│     ┌────────────────────────────┐                                   │
│     │▓░░░░░░░░░░░░░░░░░░░░░░░░░▓│ ← Global tokens (CLS, SEP)        │
│     │░▓▓▓░░░░░░░░░░░░░░░░░░░░░░░│                                    │
│     │░▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░│ ← Sliding window                   │
│     │░░▓▓▓▓░░░░░░░░░░░░░░░░░░░░░│   (diagonal band)                  │
│     │░░░▓▓▓▓░░░░░░░░░░░░░░░░░░░░│                                    │
│     │░░░░▓▓▓▓░░░░░░░░░░░░░░░░░░░│                                    │
│     │      ...diagonal...       │                                    │
│     │▓░░░░░░░░░░░░░░░░░░░░░░░▓▓▓│                                    │
│     └────────────────────────────┘                                   │
│                                                                       │
│  Complexity: O(n × w) instead of O(n²)                               │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Longformer for Finance

| Use Case | Standard Transformer | Longformer | Benefit |
|----------|---------------------|------------|---------|
| SEC 10-K Filing | Truncate to 512 tokens | Full 50K tokens | Complete document analysis |
| Earnings Calls | Lose context between chunks | Single pass | Better sentiment extraction |
| Minute-level Trading | ~8 hours max | ~30 days | Longer-term pattern recognition |
| Multi-document QA | Separate processing | Joint attention | Cross-document reasoning |
| News + Price Fusion | Limited context | Extended context | Better event correlation |

## Mathematical Foundation

### Standard Self-Attention

The standard self-attention mechanism computes:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

Where:
- **Q, K, V**: Query, Key, Value matrices ∈ ℝ^(n×d)
- **d_k**: Key dimension (typically d_model / n_heads)
- **n**: Sequence length

The attention matrix `QK^T` is n×n, requiring O(n²) time and space.

### Sliding Window Attention

Sliding window attention restricts each token to attend only to a fixed window of neighboring tokens:

```
For token at position i with window size w:
    Attention_i = {j : |i - j| ≤ w/2}

Example with w=4 (window of 4 tokens total):
Position 5 attends to: [3, 4, 5, 6, 7]

Attention Pattern for w=4, n=10:
        1  2  3  4  5  6  7  8  9  10
    1  [■  ■  ■  ░  ░  ░  ░  ░  ░  ░]
    2  [■  ■  ■  ■  ░  ░  ░  ░  ░  ░]
    3  [■  ■  ■  ■  ■  ░  ░  ░  ░  ░]
    4  [░  ■  ■  ■  ■  ■  ░  ░  ░  ░]
    5  [░  ░  ■  ■  ■  ■  ■  ░  ░  ░]
    6  [░  ░  ░  ■  ■  ■  ■  ■  ░  ░]
    7  [░  ░  ░  ░  ■  ■  ■  ■  ■  ░]
    8  [░  ░  ░  ░  ░  ■  ■  ■  ■  ■]
    9  [░  ░  ░  ░  ░  ░  ■  ■  ■  ■]
   10  [░  ░  ░  ░  ░  ░  ░  ■  ■  ■]

■ = attention computed, ░ = no attention

Complexity: O(n × w) where w << n
```

The receptive field grows linearly with the number of layers:
- Layer 1: w tokens
- Layer 2: 2w tokens
- Layer L: L×w tokens

For L=12 layers and w=512, the top layer has a receptive field of 6,144 tokens.

### Dilated Sliding Window

To increase receptive field without adding computation, Longformer supports dilated windows:

```
Dilated Window with dilation d=2, w=4:

Standard (d=1):  [1, 2, 3, 4, 5] → attends to consecutive tokens
Dilated (d=2):   [1, _, 3, _, 5, _, 7, _, 9] → skips every other token

Position 5 with d=2, w=4 attends to: [1, 3, 5, 7, 9]

This allows reaching position 9 with same computation as reaching position 5!

Receptive field with dilation:
- Layer l with dilation d_l: d_l × w tokens
- Multi-scale: lower layers use d=1 (local), higher layers use d=2+ (global)
```

### Global Attention

Some tokens need full sequence attention (task-specific):

```python
# Global attention for specific tokens
global_attention_mask = torch.zeros(batch_size, seq_len)

# For classification: [CLS] token gets global attention
global_attention_mask[:, 0] = 1  # [CLS] at position 0

# For QA: Question tokens get global attention
global_attention_mask[:, question_start:question_end] = 1

# Global attention is asymmetric:
# 1. Global tokens attend to ALL tokens (full row)
# 2. ALL tokens attend to global tokens (full column)
```

Global attention pattern visualization:

```
With global attention on token 1 and 8 (G = global):

        1  2  3  4  5  6  7  8  9  10
    1  [G  G  G  G  G  G  G  G  G  G]  ← Global token: full attention
    2  [G  ■  ■  ■  ░  ░  ░  G  ░  ░]
    3  [G  ■  ■  ■  ■  ░  ░  G  ░  ░]
    4  [G  ░  ■  ■  ■  ■  ░  G  ░  ░]
    5  [G  ░  ░  ■  ■  ■  ■  G  ░  ░]
    6  [G  ░  ░  ░  ■  ■  ■  G  ■  ░]
    7  [G  ░  ░  ░  ░  ■  ■  G  ■  ■]
    8  [G  G  G  G  G  G  G  G  G  G]  ← Global token: full attention
    9  [G  ░  ░  ░  ░  ░  ■  G  ■  ■]
   10  [G  ░  ░  ░  ░  ░  ░  G  ■  ■]
        ↑                    ↑
        │                    └─ Column: all attend to global
        └─ Column: all attend to global
```

## Longformer Architecture

### Attention Pattern Design

```python
class LongformerAttention(nn.Module):
    """
    Longformer attention combining sliding window and global attention.

    Key features:
    - Sliding window for local context (O(n×w))
    - Global attention for task-specific tokens
    - Configurable attention patterns per layer
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 512,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.one_sided_window = window_size // 2
        self.dilation = dilation

        # Separate projections for local and global attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Global attention uses separate projections
        self.q_global = nn.Linear(d_model, d_model)
        self.k_global = nn.Linear(d_model, d_model)
        self.v_global = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def sliding_window_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute sliding window attention.

        Args:
            query: [batch, n_heads, seq_len, head_dim]
            key: [batch, n_heads, seq_len, head_dim]
            value: [batch, n_heads, seq_len, head_dim]

        Returns:
            output: [batch, n_heads, seq_len, head_dim]
        """
        batch, n_heads, seq_len, head_dim = query.shape

        # For efficiency, we chunk the sequence
        # and compute attention within overlapping chunks

        # Pad sequence for windowing
        padding = self.one_sided_window
        key_padded = F.pad(key, (0, 0, padding, padding))
        value_padded = F.pad(value, (0, 0, padding, padding))

        # Extract sliding windows using unfold
        # Shape: [batch, n_heads, seq_len, window_size, head_dim]
        key_windows = key_padded.unfold(2, self.window_size, 1)
        value_windows = value_padded.unfold(2, self.window_size, 1)

        # Compute attention scores within windows
        # query: [batch, n_heads, seq_len, 1, head_dim]
        # key_windows: [batch, n_heads, seq_len, window_size, head_dim]
        query = query.unsqueeze(-2)

        scores = torch.matmul(query, key_windows.transpose(-1, -2))
        scores = scores / math.sqrt(head_dim)
        scores = scores.squeeze(-2)  # [batch, n_heads, seq_len, window_size]

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights.unsqueeze(-2), value_windows)
        output = output.squeeze(-2)

        return output
```

### Linear Complexity Analysis

```
Memory and Compute Comparison:

Standard Attention (seq_len=4096):
┌─────────────────────────────────────────┐
│ Attention Matrix: 4096 × 4096 = 16.7M   │
│ Memory: ~64MB per head (fp32)           │
│ Compute: O(n²) = O(16.7M)               │
└─────────────────────────────────────────┘

Longformer (seq_len=4096, window=512):
┌─────────────────────────────────────────┐
│ Local Attention: 4096 × 512 = 2.1M      │
│ + Global (100 tokens): 100 × 4096 × 2   │
│                       = 0.8M            │
│ Total: ~2.9M operations                 │
│ Memory: ~12MB per head (fp32)           │
│ Compute: O(n × w) = O(2.1M)             │
└─────────────────────────────────────────┘

Speedup: ~5.8× faster, ~5× less memory

For longer sequences (seq_len=16384, window=512):
Standard: 268M operations
Longformer: 8.4M operations
Speedup: ~32× faster!
```

### Implementation Details

```python
class LongformerLayer(nn.Module):
    """Single Longformer encoder layer."""

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        window_size: int = 512,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        layer_id: int = 0
    ):
        super().__init__()

        # Attention with configurable window/dilation per layer
        self.attention = LongformerAttention(
            d_model=d_model,
            n_heads=n_heads,
            window_size=window_size,
            dropout=dropout
        )

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: Padding mask
            global_attention_mask: Which tokens have global attention

        Returns:
            output: [batch, seq_len, d_model]
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        attn_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )
        hidden_states = residual + attn_output

        # FFN with residual
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = residual + self.ffn(hidden_states)

        return hidden_states
```

## Financial Applications

### Long Financial Document Analysis

Process entire SEC filings, research reports, and earnings transcripts:

```python
class LongformerFinancialNLP(nn.Module):
    """
    Longformer for financial document analysis.

    Use cases:
    - SEC 10-K/10-Q sentiment analysis
    - Earnings call transcription analysis
    - Research report summarization
    - Multi-document question answering
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        window_size: int = 512,
        max_position_embeddings: int = 16384,
        num_labels: int = 3,  # negative, neutral, positive
        dropout: float = 0.1
    ):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)

        self.layers = nn.ModuleList([
            LongformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                window_size=window_size,
                dropout=dropout,
                layer_id=i
            )
            for i in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Task-specific heads
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for document classification.

        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] padding mask
            global_attention_mask: [batch, seq_len] global attention mask
            labels: [batch] classification labels

        Returns:
            Dictionary with loss, logits, etc.
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device)
        hidden_states = self.embeddings(input_ids)
        hidden_states = hidden_states + self.position_embeddings(position_ids)

        # Set global attention on [CLS] token
        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1  # [CLS] token

        # Encode
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            )

        hidden_states = self.norm(hidden_states)

        # Use [CLS] token for classification
        cls_output = hidden_states[:, 0]
        logits = self.classifier(cls_output)

        outputs = {'logits': logits}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            outputs['loss'] = loss

        return outputs
```

### Extended Time Series Processing

Apply Longformer architecture to financial time series:

```python
class LongformerTimeSeries(nn.Module):
    """
    Longformer adapted for financial time series.

    Key adaptations:
    - Continuous input embeddings instead of token embeddings
    - Global attention on recent data and periodic anchors
    - Multi-scale temporal patterns via dilated windows
    """

    def __init__(
        self,
        input_dim: int = 6,          # OHLCV + features
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        window_size: int = 256,      # Local attention window
        max_seq_len: int = 8192,
        pred_horizon: int = 24,
        global_token_freq: int = 256,  # Global token every N positions
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.window_size = window_size
        self.global_token_freq = global_token_freq

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable positional encoding for long sequences
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )

        # Longformer layers with increasing dilation
        self.layers = nn.ModuleList([
            LongformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                window_size=window_size,
                dilation=min(2 ** (i // 2), 8),  # Increase dilation in deeper layers
                dropout=dropout,
                layer_id=i
            )
            for i in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_horizon)
        )

    def create_global_attention_mask(
        self,
        seq_len: int,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create global attention mask for time series.

        Strategy:
        - Last position (most recent) gets global attention
        - Every global_token_freq positions gets global attention
        - This creates periodic "anchor" points
        """
        mask = torch.zeros(batch_size, seq_len, device=device)

        # Global attention on last position (current time)
        mask[:, -1] = 1

        # Global attention on periodic anchors
        anchor_positions = torch.arange(0, seq_len, self.global_token_freq)
        mask[:, anchor_positions] = 1

        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch, seq_len, input_dim] time series features

        Returns:
            predictions: [batch, pred_horizon]
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        hidden_states = self.input_proj(x)

        # Add positional encoding
        hidden_states = hidden_states + self.pos_encoding[:, :seq_len]

        # Create global attention mask
        global_attention_mask = self.create_global_attention_mask(
            seq_len, batch_size, x.device
        )

        # Encode with Longformer layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                global_attention_mask=global_attention_mask
            )

        hidden_states = self.norm(hidden_states)

        # Use last position for prediction
        predictions = self.pred_head(hidden_states[:, -1])

        return predictions
```

### Multi-Asset Correlation Analysis

Analyze correlations across multiple assets with long history:

```python
class LongformerMultiAsset(nn.Module):
    """
    Multi-asset analysis with Longformer.

    Process multiple asset time series jointly with:
    - Per-asset sliding window attention (temporal patterns)
    - Cross-asset global attention (correlations)
    """

    def __init__(
        self,
        n_assets: int = 10,
        features_per_asset: int = 6,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        window_size: int = 128,
        seq_len: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_assets = n_assets
        self.seq_len = seq_len
        self.d_model = d_model

        # Per-asset embedding
        self.asset_embeddings = nn.Embedding(n_assets, d_model)
        self.input_proj = nn.Linear(features_per_asset, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, seq_len, d_model) * 0.02
        )

        # Longformer layers
        self.layers = nn.ModuleList([
            LongformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                window_size=window_size,
                dropout=dropout,
                layer_id=i
            )
            for i in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Output: allocation weights for each asset
        self.allocation_head = nn.Sequential(
            nn.Linear(d_model * n_assets, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_assets),
            nn.Softmax(dim=-1)  # Portfolio weights sum to 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch, n_assets, seq_len, features_per_asset]

        Returns:
            allocations: [batch, n_assets] portfolio weights
        """
        batch_size = x.shape[0]

        # Process each asset
        asset_outputs = []

        for asset_idx in range(self.n_assets):
            # Get asset data: [batch, seq_len, features]
            asset_data = x[:, asset_idx]

            # Project and add embeddings
            hidden = self.input_proj(asset_data)
            hidden = hidden + self.asset_embeddings.weight[asset_idx]
            hidden = hidden + self.pos_encoding[:, :self.seq_len]

            # Create global attention for last timestamp
            global_mask = torch.zeros(batch_size, self.seq_len, device=x.device)
            global_mask[:, -1] = 1  # Last position is global

            # Encode
            for layer in self.layers:
                hidden = layer(hidden, global_attention_mask=global_mask)

            hidden = self.norm(hidden)

            # Take last position output
            asset_outputs.append(hidden[:, -1])

        # Concatenate all asset representations
        combined = torch.cat(asset_outputs, dim=-1)

        # Predict allocation weights
        allocations = self.allocation_head(combined)

        return allocations
```

## Practical Examples

### 01: Data Preparation

```python
# python/data.py

import numpy as np
import pandas as pd
import requests
from typing import List, Tuple, Dict, Optional
from transformers import AutoTokenizer

def prepare_financial_documents(
    documents: List[str],
    tokenizer_name: str = 'allenai/longformer-base-4096',
    max_length: int = 4096
) -> Dict[str, np.ndarray]:
    """
    Prepare financial documents for Longformer processing.

    Args:
        documents: List of document texts
        tokenizer_name: HuggingFace tokenizer to use
        max_length: Maximum sequence length

    Returns:
        Dictionary with input_ids, attention_mask, global_attention_mask
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    encodings = tokenizer(
        documents,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )

    # Create global attention mask (1 for [CLS], 0 elsewhere)
    global_attention_mask = np.zeros_like(encodings['input_ids'])
    global_attention_mask[:, 0] = 1

    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'global_attention_mask': global_attention_mask
    }


def prepare_timeseries_data(
    symbols: List[str],
    lookback: int = 4096,
    horizon: int = 24,
    source: str = 'bybit'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare long time series data for Longformer.

    Args:
        symbols: Trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
        lookback: Historical sequence length
        horizon: Prediction horizon
        source: Data source ('bybit', 'yahoo')

    Returns:
        X: Features [n_samples, lookback, n_features]
        y: Targets [n_samples, horizon]
    """
    all_features = []

    for symbol in symbols:
        if source == 'bybit':
            df = load_bybit_data(symbol, interval='1m')
        else:
            df = load_yahoo_data(symbol)

        # Calculate features
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(20).std()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(50).mean()
        df['price_ma_ratio'] = df['close'] / df['close'].rolling(200).mean()
        df['rsi'] = calculate_rsi(df['close'], period=14)
        df['macd'] = calculate_macd(df['close'])

        all_features.append(df)

    # Combine and create sequences
    features = pd.concat(all_features, axis=1, keys=symbols)
    features = features.dropna()

    X, y = [], []
    feature_cols = ['log_return', 'volatility', 'volume_ma_ratio',
                    'price_ma_ratio', 'rsi', 'macd']

    for i in range(lookback, len(features) - horizon):
        x_seq = features.iloc[i-lookback:i][
            [(s, f) for s in symbols for f in feature_cols]
        ].values
        X.append(x_seq)

        y_seq = features.iloc[i:i+horizon][(symbols[0], 'log_return')].values
        y.append(y_seq)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_bybit_data(symbol: str, interval: str = '1m') -> pd.DataFrame:
    """Load historical data from Bybit."""
    url = "https://api.bybit.com/v5/market/kline"

    params = {
        'category': 'spot',
        'symbol': symbol,
        'interval': '1' if interval == '1m' else interval,
        'limit': 10000
    }

    response = requests.get(url, params=params)
    data = response.json()['result']['list']

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])

    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = pd.to_numeric(df[col])

    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """Calculate MACD indicator."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow
```

### 02: Longformer Model

```python
# python/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

class LongformerSlidingWindowAttention(nn.Module):
    """
    Efficient sliding window attention implementation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.one_sided_window = window_size // 2
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Local attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Global attention projections
        self.q_global = nn.Linear(d_model, d_model)
        self.k_global = nn.Linear(d_model, d_model)
        self.v_global = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _compute_local_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute sliding window attention efficiently.

        Uses chunked computation for memory efficiency.
        """
        batch, n_heads, seq_len, head_dim = query.shape

        # Pad for windowing
        pad = self.one_sided_window
        key_padded = F.pad(key, (0, 0, pad, pad), value=0)
        value_padded = F.pad(value, (0, 0, pad, pad), value=0)

        # Create sliding windows using unfold
        # key_windows: [batch, n_heads, seq_len, window_size, head_dim]
        key_windows = key_padded.unfold(dimension=2, size=self.window_size, step=1)
        key_windows = key_windows.transpose(-1, -2)

        value_windows = value_padded.unfold(dimension=2, size=self.window_size, step=1)
        value_windows = value_windows.transpose(-1, -2)

        # Compute attention scores
        # query: [batch, n_heads, seq_len, 1, head_dim]
        query = query.unsqueeze(-2)

        # scores: [batch, n_heads, seq_len, 1, window_size]
        scores = torch.matmul(query, key_windows.transpose(-1, -2)) * self.scale
        scores = scores.squeeze(-2)  # [batch, n_heads, seq_len, window_size]

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        # [batch, n_heads, seq_len, window_size] @ [batch, n_heads, seq_len, window_size, head_dim]
        attn_weights = attn_weights.unsqueeze(-2)
        output = torch.matmul(attn_weights, value_windows)
        output = output.squeeze(-2)  # [batch, n_heads, seq_len, head_dim]

        return output

    def _compute_global_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        global_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute global attention for marked tokens.

        Returns:
            global_output: Output for global tokens
            extra_attention: Attention from all tokens to global tokens
        """
        batch, n_heads, seq_len, head_dim = query.shape

        # Get global token indices
        global_indices = global_mask.nonzero(as_tuple=True)
        n_global = global_mask.sum().item() // batch

        if n_global == 0:
            return torch.zeros_like(query), torch.zeros_like(query)

        # Extract global queries, keys, values
        # For simplicity, we compute full attention for global tokens
        global_query = query[global_mask.bool().unsqueeze(1).unsqueeze(-1).expand_as(query)]
        global_query = global_query.view(batch, n_heads, -1, head_dim)

        # Global tokens attend to all tokens
        global_scores = torch.matmul(global_query, key.transpose(-1, -2)) * self.scale
        global_attn = F.softmax(global_scores, dim=-1)
        global_attn = self.dropout(global_attn)
        global_output = torch.matmul(global_attn, value)

        # All tokens attend to global tokens
        global_key = key[global_mask.bool().unsqueeze(1).unsqueeze(-1).expand_as(key)]
        global_key = global_key.view(batch, n_heads, -1, head_dim)

        global_value = value[global_mask.bool().unsqueeze(1).unsqueeze(-1).expand_as(value)]
        global_value = global_value.view(batch, n_heads, -1, head_dim)

        extra_scores = torch.matmul(query, global_key.transpose(-1, -2)) * self.scale
        extra_attn = F.softmax(extra_scores, dim=-1)
        extra_attn = self.dropout(extra_attn)
        extra_output = torch.matmul(extra_attn, global_value)

        return global_output, extra_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass combining local and global attention.
        """
        batch, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        Q = Q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute local sliding window attention
        local_output = self._compute_local_attention(Q, K, V)

        # Add global attention if specified
        if global_attention_mask is not None and global_attention_mask.sum() > 0:
            Q_global = self.q_global(hidden_states)
            K_global = self.k_global(hidden_states)
            V_global = self.v_global(hidden_states)

            Q_global = Q_global.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            K_global = K_global.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            V_global = V_global.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

            global_output, extra_output = self._compute_global_attention(
                Q_global, K_global, V_global, global_attention_mask
            )

            # Combine local and global attention
            output = local_output + extra_output

            # Replace global token outputs
            if global_output.shape[2] > 0:
                global_indices = global_attention_mask.nonzero(as_tuple=True)
                output[:, :, global_indices[1]] = global_output
        else:
            output = local_output

        # Reshape and project
        output = output.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        output = self.out_proj(output)
        output = self.dropout(output)

        return output


class LongformerEncoderLayer(nn.Module):
    """Single Longformer encoder layer."""

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        window_size: int = 512,
        dim_feedforward: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attention = LongformerSlidingWindowAttention(
            d_model=d_model,
            n_heads=n_heads,
            window_size=window_size,
            dropout=dropout
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm architecture
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LongformerForTrading(nn.Module):
    """
    Longformer model for trading applications.

    Supports both NLP (document) and time series inputs.
    """

    def __init__(
        self,
        input_type: str = 'timeseries',  # 'timeseries' or 'text'
        # Time series parameters
        input_dim: int = 6,
        # Text parameters
        vocab_size: int = 50000,
        # Shared parameters
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        window_size: int = 256,
        max_seq_len: int = 4096,
        # Output parameters
        output_type: str = 'regression',
        pred_horizon: int = 24,
        n_classes: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_type = input_type
        self.output_type = output_type
        self.d_model = d_model

        # Input embedding
        if input_type == 'timeseries':
            self.input_proj = nn.Linear(input_dim, d_model)
        else:
            self.input_proj = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )

        # Longformer layers
        self.layers = nn.ModuleList([
            LongformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                window_size=window_size,
                dim_feedforward=d_model * 4,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Output head
        if output_type == 'regression':
            self.head = nn.Linear(d_model, pred_horizon)
        elif output_type == 'classification':
            self.head = nn.Linear(d_model, n_classes)
        elif output_type == 'allocation':
            self.head = nn.Sequential(
                nn.Linear(d_model, pred_horizon),
                nn.Tanh()
            )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor
               - Time series: [batch, seq_len, input_dim]
               - Text: [batch, seq_len] (token IDs)
            attention_mask: Padding mask
            global_attention_mask: Global attention positions

        Returns:
            Predictions based on output_type
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Input embedding
        if self.input_type == 'timeseries':
            hidden_states = self.input_proj(x)
        else:
            hidden_states = self.input_proj(x)

        # Add positional encoding
        hidden_states = hidden_states + self.pos_encoding[:, :seq_len]

        # Default global attention on last position
        if global_attention_mask is None:
            global_attention_mask = torch.zeros(batch_size, seq_len, device=x.device)
            global_attention_mask[:, -1] = 1  # Last position
            if self.input_type == 'text':
                global_attention_mask[:, 0] = 1  # [CLS] token

        # Encode
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            )

        hidden_states = self.norm(hidden_states)

        # Use last position for prediction
        output = self.head(hidden_states[:, -1])

        return output
```

### 03: Training Pipeline

```python
# python/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Optional, List
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LongformerTrainer:
    """Training pipeline for Longformer trading model."""

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

        # Loss based on output type
        if model.output_type == 'regression':
            self.criterion = nn.MSELoss()
        elif model.output_type == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif model.output_type == 'allocation':
            self.criterion = self._sharpe_loss

    def _get_lr_multiplier(self) -> float:
        """Linear warmup then constant."""
        if self.current_step < self.warmup_steps:
            return self.current_step / self.warmup_steps
        return 1.0

    def _sharpe_loss(
        self,
        allocations: torch.Tensor,
        returns: torch.Tensor
    ) -> torch.Tensor:
        """Differentiable Sharpe ratio loss."""
        portfolio_returns = allocations * returns
        mean_ret = portfolio_returns.mean()
        std_ret = portfolio_returns.std() + 1e-8
        return -mean_ret / std_ret

    def train_epoch(
        self,
        dataloader: DataLoader,
        global_attention_mode: str = 'last'
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Adjust learning rate
            lr_mult = self._get_lr_multiplier()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_mult

            self.optimizer.zero_grad()

            # Create global attention mask
            batch_size, seq_len = batch_x.shape[0], batch_x.shape[1]
            global_mask = self._create_global_mask(
                batch_size, seq_len, global_attention_mode
            )

            predictions = self.model(batch_x, global_attention_mask=global_mask)
            loss = self.criterion(predictions, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            self.current_step += 1

        return total_loss / len(dataloader)

    def _create_global_mask(
        self,
        batch_size: int,
        seq_len: int,
        mode: str = 'last'
    ) -> torch.Tensor:
        """Create global attention mask."""
        mask = torch.zeros(batch_size, seq_len, device=self.device)

        if mode == 'last':
            mask[:, -1] = 1
        elif mode == 'periodic':
            # Every 256 positions
            positions = torch.arange(0, seq_len, 256)
            mask[:, positions] = 1
            mask[:, -1] = 1
        elif mode == 'first_last':
            mask[:, 0] = 1
            mask[:, -1] = 1

        return mask

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance."""
        self.model.eval()

        all_preds = []
        all_targets = []
        total_loss = 0.0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)

            total_loss += loss.item()
            all_preds.append(predictions.cpu())
            all_targets.append(batch_y.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        metrics = {'loss': total_loss / len(dataloader)}

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
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        patience: int = 10,
        save_dir: str = 'checkpoints'
    ) -> Dict[str, List]:
        """Full training loop."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

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
                f"Val Metrics: {val_metrics}"
            )

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                torch.save(
                    self.model.state_dict(),
                    save_path / 'longformer_best.pt'
                )
                logger.info(f"Saved best model")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        return history


def main():
    """Example training script."""
    from data import prepare_timeseries_data
    from model import LongformerForTrading

    # Load data
    logger.info("Loading data from Bybit...")
    X, y = prepare_timeseries_data(
        symbols=['BTCUSDT', 'ETHUSDT'],
        lookback=4096,
        horizon=24,
        source='bybit'
    )

    # Split
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]

    # Dataloaders
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=16, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=16
    )

    # Model
    model = LongformerForTrading(
        input_type='timeseries',
        input_dim=X.shape[-1],
        d_model=256,
        n_heads=8,
        n_layers=6,
        window_size=256,
        max_seq_len=4096,
        output_type='regression',
        pred_horizon=24
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = LongformerTrainer(model, learning_rate=1e-4)
    history = trainer.train(
        train_loader, val_loader,
        num_epochs=100, patience=15
    )

    logger.info("Training completed!")


if __name__ == '__main__':
    main()
```

### 04: Backtesting Strategy

```python
# python/strategy.py

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    max_position: float = 1.0
    confidence_threshold: float = 0.6


class LongformerBacktester:
    """Backtesting for Longformer trading strategy."""

    def __init__(
        self,
        model: torch.nn.Module,
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
        """Generate trading signals."""
        x = torch.tensor(data, dtype=torch.float32)
        predictions = self.model(x).numpy()

        pred_returns = predictions[:, 0] if len(predictions.shape) > 1 else predictions

        signals = np.zeros_like(pred_returns)
        signals[pred_returns > threshold] = 1
        signals[pred_returns < -threshold] = -1

        return signals

    def run_backtest(
        self,
        data: np.ndarray,
        prices: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> Dict:
        """Run backtest simulation."""
        signals = self.generate_signals(data)

        capital = self.config.initial_capital
        position = 0.0

        equity_curve = [capital]
        positions = [0.0]
        returns = []
        trades = []

        for i in range(len(signals)):
            current_price = prices[i]
            signal = signals[i]

            target_position = signal * self.config.max_position
            position_change = target_position - position

            if abs(position_change) > 0.01:
                trade_value = abs(position_change) * capital
                costs = trade_value * (self.config.transaction_cost + self.config.slippage)
                capital -= costs

                trades.append({
                    'timestamp': timestamps[i],
                    'price': current_price,
                    'signal': signal,
                    'position_change': position_change,
                    'costs': costs
                })

                position = target_position

            if i > 0 and position != 0:
                price_return = (current_price - prices[i-1]) / prices[i-1]
                pnl = position * capital * price_return
                capital += pnl
                returns.append(pnl / equity_curve[-1])
            else:
                returns.append(0.0)

            equity_curve.append(capital)
            positions.append(position)

        returns = np.array(returns)
        equity_curve = np.array(equity_curve)

        metrics = self._calculate_metrics(returns, equity_curve, trades)

        return {
            'equity_curve': equity_curve,
            'positions': positions,
            'returns': returns,
            'trades': trades,
            'timestamps': timestamps,
            'metrics': metrics
        }

    def _calculate_metrics(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray,
        trades: List[Dict]
    ) -> Dict:
        """Calculate performance metrics."""
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0.0

        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            sortino_ratio = np.sqrt(252) * returns.mean() / (downside_returns.std() + 1e-8)
        else:
            sortino_ratio = sharpe_ratio

        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()

        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'num_trades': len(trades),
            'final_capital': equity_curve[-1]
        }

    def plot_results(self, results: Dict, save_path: Optional[str] = None):
        """Visualize backtest results."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Equity curve
        axes[0].plot(results['timestamps'], results['equity_curve'][1:], 'b-', linewidth=1)
        axes[0].set_title('Equity Curve (Longformer Strategy)')
        axes[0].set_ylabel('Capital ($)')
        axes[0].grid(True, alpha=0.3)

        # Positions
        axes[1].fill_between(
            results['timestamps'], results['positions'][1:], 0,
            where=np.array(results['positions'][1:]) > 0, color='green', alpha=0.5, label='Long'
        )
        axes[1].fill_between(
            results['timestamps'], results['positions'][1:], 0,
            where=np.array(results['positions'][1:]) < 0, color='red', alpha=0.5, label='Short'
        )
        axes[1].set_title('Position Size')
        axes[1].set_ylabel('Position')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Drawdown
        equity = np.array(results['equity_curve'][1:])
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100
        axes[2].fill_between(results['timestamps'], drawdown, 0, color='red', alpha=0.5)
        axes[2].set_title('Drawdown')
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

        # Print metrics
        print("\n" + "="*50)
        print("LONGFORMER BACKTEST RESULTS")
        print("="*50)
        for metric, value in results['metrics'].items():
            if isinstance(value, float):
                print(f"{metric:20s}: {value:>12.4f}")
            else:
                print(f"{metric:20s}: {value:>12}")
        print("="*50)


def main():
    """Run backtest example."""
    from model import LongformerForTrading
    from data import prepare_timeseries_data, load_bybit_data

    # Load model
    model = LongformerForTrading(
        input_type='timeseries',
        input_dim=12,
        d_model=256,
        n_heads=8,
        n_layers=6,
        window_size=256,
        max_seq_len=4096,
        output_type='regression',
        pred_horizon=24
    )
    model.load_state_dict(torch.load('checkpoints/longformer_best.pt'))

    # Load test data
    X, y = prepare_timeseries_data(
        symbols=['BTCUSDT', 'ETHUSDT'],
        lookback=4096,
        horizon=24,
        source='bybit'
    )

    price_data = load_bybit_data('BTCUSDT')

    test_start = int(0.8 * len(X))
    X_test = X[test_start:]
    prices = price_data['close'].values[4096+test_start:4096+test_start+len(X_test)]
    timestamps = pd.DatetimeIndex(
        price_data['timestamp'].values[4096+test_start:4096+test_start+len(X_test)]
    )

    # Run backtest
    backtester = LongformerBacktester(model)
    results = backtester.run_backtest(X_test, prices, timestamps)
    backtester.plot_results(results, save_path='longformer_backtest.png')


if __name__ == '__main__':
    main()
```

## Rust Implementation

See [rust/](rust/) for complete Rust implementation.

```
rust/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs
│   ├── api/
│   │   ├── mod.rs
│   │   ├── bybit.rs
│   │   └── types.rs
│   ├── attention/
│   │   ├── mod.rs
│   │   ├── sliding_window.rs
│   │   └── global.rs
│   ├── model/
│   │   ├── mod.rs
│   │   ├── longformer.rs
│   │   └── encoder.rs
│   └── strategy/
│       ├── mod.rs
│       ├── signals.rs
│       └── backtest.rs
└── examples/
    ├── fetch_data.rs
    ├── train.rs
    └── backtest.rs
```

### Quick Start (Rust)

```bash
cd rust

# Fetch data from Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT

# Train model
cargo run --example train -- --epochs 50 --batch-size 16

# Run backtest
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Python Implementation

See [python/](python/) for Python implementation.

```
python/
├── __init__.py
├── model.py           # Longformer implementation
├── data.py            # Data loading (Bybit, Yahoo)
├── train.py           # Training pipeline
├── strategy.py        # Backtesting
├── requirements.txt
└── examples/
    └── longformer_trading.ipynb
```

### Quick Start (Python)

```bash
cd python
pip install -r requirements.txt

# Fetch and train
python train.py --symbols BTCUSDT,ETHUSDT --epochs 100

# Backtest
python strategy.py --model checkpoints/longformer_best.pt
```

## Financial NLP Applications

### Document-Level Sentiment Classification

The primary financial NLP task for Longformer is classifying the sentiment of entire documents -- earnings call transcripts, analyst reports, or news articles -- without truncation. Given a document tokenized into $\{x_1, x_2, \ldots, x_n\}$ with $n \leq 4096$, the Longformer encodes the sequence using its mixed attention pattern. The [CLS] token representation $\mathbf{h}_{\text{CLS}} \in \mathbb{R}^d$ is fed through a classification head:

$$\hat{y} = \text{softmax}(\mathbf{W}\mathbf{h}_{\text{CLS}} + \mathbf{b})$$

Classes typically include positive, negative, and neutral sentiment.

### Named Entity Recognition in Financial Documents

Longformer excels at financial NER because entity context often spans long distances. For example, a company name mentioned on page one of a 10-K may be relevant to a risk factor discussed on page ten. The model uses sliding window attention for local entity detection and global attention on section headers to maintain document-level context.

Each token $x_i$ is classified into BIO tags:

$$\hat{y}_i = \text{softmax}(\mathbf{W}_{\text{ner}}\mathbf{h}_i + \mathbf{b}_{\text{ner}})$$

Entity types relevant to finance include: ORG (organizations), MONEY (monetary amounts), PERCENT (percentages), DATE (dates), PRODUCT (financial instruments), and EVENT (market events).

### Risk Factor Detection

Regulatory filings contain risk disclosures that are material to investment decisions. Longformer can classify each paragraph or section as containing a risk factor or not:

$$P(\text{risk} | \text{paragraph}_i) = \sigma(\mathbf{w}^T \mathbf{h}_i + b)$$

where $\mathbf{h}_i$ is the Longformer representation of the paragraph.

### Pre-training and Fine-tuning Pipeline

Longformer is initialized from RoBERTa checkpoints and continues pre-training on long documents:

1. **Pre-training**: Start from a pre-trained Longformer checkpoint (e.g., `allenai/longformer-base-4096`).
2. **Domain adaptation**: Continue pre-training on a financial corpus (SEC filings, financial news, analyst reports) using masked language modeling.
3. **Fine-tuning**: Train on the downstream task (classification, NER, sentiment) with task-specific heads.

Position embeddings are extended from 512 to 4,096 by replicating the existing embeddings.

### Feature Engineering for Financial NLP

**Token-Level Features:**
- Financial numbers: Monetary values, percentages, ratios should be normalized
- Temporal expressions: Fiscal quarters, year-over-year references
- Legal/regulatory terms: "Material adverse effect", "going concern"
- Sentiment indicators: "Exceeded expectations", "headwinds", "robust growth"

**Document Structure Features:**
- Section headers: Map to global attention tokens for cross-section information routing
- Table markers: Financial tables contain critical quantitative data
- Footnote references: Connect footnote content to main text across long distances

**Aggregated Sentiment Features for Trading:**
- Sentiment score: Continuous value in $[-1, 1]$ from the classification head
- Sentiment momentum: Change in sentiment across consecutive documents $\Delta S_t = S_t - S_{t-1}$
- Cross-document consensus: Average sentiment across multiple sources
- Sentiment surprise: Deviation from expected sentiment

### Specific NLP Application Areas

**Earnings Call Analysis:**
Transcripts are 5,000-15,000 words. Longformer can perform full-transcript sentiment analysis, section-level tone comparison (CEO vs CFO remarks vs Q&A), and forward-looking statement detection.

**Regulatory Filing Analysis:**
10-K and 10-Q filings contain critical information: risk factor extraction, Management Discussion & Analysis interpretation, and material event detection.

**Crypto Market News Analysis:**
For cryptocurrency markets: whitepaper technical evaluation, governance proposal impact assessment, and multi-source thread aggregation for long-context analysis.

### Evaluation Metrics for NLP Tasks

- **Classification**: Accuracy, F1-score (macro and weighted), AUC-ROC
- **NER**: Entity-level F1, precision, recall
- **Sentiment**: Cohen's Kappa, directional accuracy for trading signals
- **Trading performance**: Sharpe ratio, Sortino ratio, maximum drawdown

## Best Practices

### When to Use Longformer

**Ideal use cases:**
- Long documents: SEC filings, research reports, earnings transcripts
- Extended time series: 4K-16K timesteps
- Tasks requiring both local and global context
- Document QA with long source texts

**Consider alternatives when:**
- Short sequences (<512): Standard transformer is fine
- Pure local patterns: Sliding window alone may suffice
- Very sparse global dependencies: Use sparse attention

### Hyperparameter Guidelines

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `window_size` | 256-512 | Larger for more local context |
| `max_seq_len` | 4096-16384 | Based on document/series length |
| `n_layers` | 6-12 | More for complex patterns |
| `global_attention` | Task-specific | [CLS] for classification, recent for time series |

### Window Size Selection

```
Rule of thumb for window size:

Document length → Window size
- 4096 tokens  → 256-512
- 8192 tokens  → 512
- 16384 tokens → 512-1024

For time series (minute data):
- 1 day (1440 min)  → 128-256
- 1 week (10080 min) → 256-512
- 1 month (43200 min) → 512-1024
```

## Resources

### Papers

- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — Original paper (2020)
- [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — Related architecture
- [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732) — Comprehensive overview

### Implementations

- [HuggingFace Transformers](https://huggingface.co/docs/transformers/model_doc/longformer) — Official implementation
- [AllenAI Longformer](https://github.com/allenai/longformer) — Original repository

### Additional NLP References

- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. arXiv:1908.10063.
- Huang, A. H., Wang, H., & Yang, Y. (2023). FinBERT: A Large Language Model for Extracting Information from Financial Text. Contemporary Accounting Research, 40(2), 806-841.
- Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. The Journal of Finance, 66(1), 35-65.
- Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692.

### Related Chapters

- [Chapter 51: Linformer Long Sequences](../51_linformer_long_sequences) — Linear attention
- [Chapter 52: Performer Efficient Attention](../52_performer_efficient_attention) — Random features
- [Chapter 53: BigBird Sparse Attention](../53_bigbird_sparse_attention) — Similar sparse approach
- [Chapter 56: Nystromformer Trading](../56_nystromformer_trading) — Nyström approximation

---

## Difficulty Level

**Intermediate**

Prerequisites:
- Understanding of transformer self-attention
- Basic knowledge of NLP and tokenization
- Familiarity with time series forecasting
- PyTorch or Rust ML library experience

Sources:
- [Longformer Paper (arXiv)](https://arxiv.org/abs/2004.05150)
- [AllenAI Longformer GitHub](https://github.com/allenai/longformer)
- [HuggingFace Longformer](https://huggingface.co/papers/2004.05150)
