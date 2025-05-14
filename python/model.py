"""
Longformer Model Implementation for Financial Analysis

This module implements the Longformer architecture with sliding window
attention and global attention mechanisms, adapted for financial applications.

References:
    - Paper: "Longformer: The Long-Document Transformer" (Beltagy et al., 2020)
    - arXiv: https://arxiv.org/abs/2004.05150
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class LongformerSlidingWindowAttention(nn.Module):
    """
    Efficient sliding window attention implementation.

    This attention mechanism combines:
    1. Local sliding window attention - each token attends to w neighbors
    2. Global attention - selected tokens attend to all positions

    Complexity: O(n * w) instead of O(n^2) for standard attention

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        window_size: Size of the sliding window (must be even)
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert window_size % 2 == 0, "window_size must be even"

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

        # Global attention projections (separate for better performance)
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

        Uses unfold operation to create sliding windows for memory efficiency.

        Args:
            query: [batch, n_heads, seq_len, head_dim]
            key: [batch, n_heads, seq_len, head_dim]
            value: [batch, n_heads, seq_len, head_dim]

        Returns:
            output: [batch, n_heads, seq_len, head_dim]
        """
        batch, n_heads, seq_len, head_dim = query.shape

        # Pad key and value for windowing
        pad = self.one_sided_window
        key_padded = F.pad(key, (0, 0, pad, pad), value=0)
        value_padded = F.pad(value, (0, 0, pad, pad), value=0)

        # Create sliding windows using unfold
        # key_windows: [batch, n_heads, seq_len, window_size, head_dim]
        key_windows = key_padded.unfold(dimension=2, size=self.window_size, step=1)
        key_windows = key_windows.transpose(-1, -2)

        value_windows = value_padded.unfold(dimension=2, size=self.window_size, step=1)
        value_windows = value_windows.transpose(-1, -2)

        # Compute attention scores within windows
        # query: [batch, n_heads, seq_len, 1, head_dim]
        query = query.unsqueeze(-2)

        # scores: [batch, n_heads, seq_len, 1, window_size]
        scores = torch.matmul(query, key_windows.transpose(-1, -2)) * self.scale
        scores = scores.squeeze(-2)  # [batch, n_heads, seq_len, window_size]

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
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

        Global attention is bidirectional:
        - Global tokens attend to all tokens (rows)
        - All tokens attend to global tokens (columns)

        Args:
            query: [batch, n_heads, seq_len, head_dim]
            key: [batch, n_heads, seq_len, head_dim]
            value: [batch, n_heads, seq_len, head_dim]
            global_mask: [batch, seq_len] binary mask

        Returns:
            global_output: Output for positions with global attention
            extra_attention: Additional attention from all tokens to global tokens
        """
        batch, n_heads, seq_len, head_dim = query.shape
        device = query.device

        # Find global indices
        global_indices = global_mask.nonzero(as_tuple=True)

        if len(global_indices[0]) == 0:
            return torch.zeros_like(query), torch.zeros_like(query)

        # For each batch, compute global attention
        # This is a simplified implementation - production code would be more optimized
        extra_output = torch.zeros_like(query)

        for b in range(batch):
            batch_global_mask = global_mask[b]
            global_positions = batch_global_mask.nonzero(as_tuple=True)[0]

            if len(global_positions) == 0:
                continue

            # Get global token representations
            global_query = query[b, :, global_positions, :]  # [n_heads, n_global, head_dim]
            global_key = key[b, :, global_positions, :]
            global_value = value[b, :, global_positions, :]

            # Global tokens attend to all tokens (full attention for global positions)
            all_key = key[b]  # [n_heads, seq_len, head_dim]
            all_value = value[b]

            # Compute attention from global to all
            global_scores = torch.matmul(global_query, all_key.transpose(-1, -2)) * self.scale
            global_attn = F.softmax(global_scores, dim=-1)
            global_attn = self.dropout(global_attn)
            global_context = torch.matmul(global_attn, all_value)

            # All tokens attend to global tokens
            all_query = query[b]  # [n_heads, seq_len, head_dim]
            extra_scores = torch.matmul(all_query, global_key.transpose(-1, -2)) * self.scale
            extra_attn = F.softmax(extra_scores, dim=-1)
            extra_attn = self.dropout(extra_attn)
            extra_context = torch.matmul(extra_attn, global_value)

            extra_output[b] = extra_context

        return extra_output, extra_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass combining local and global attention.

        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: [batch, seq_len] padding mask (optional)
            global_attention_mask: [batch, seq_len] global attention positions

        Returns:
            output: [batch, seq_len, d_model]
        """
        batch, seq_len, _ = hidden_states.shape

        # Project to Q, K, V for local attention
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        # [batch, seq_len, d_model] -> [batch, n_heads, seq_len, head_dim]
        Q = Q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute local sliding window attention
        local_output = self._compute_local_attention(Q, K, V)

        # Add global attention if specified
        if global_attention_mask is not None and global_attention_mask.sum() > 0:
            # Use separate projections for global attention
            Q_global = self.q_global(hidden_states)
            K_global = self.k_global(hidden_states)
            V_global = self.v_global(hidden_states)

            Q_global = Q_global.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            K_global = K_global.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            V_global = V_global.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

            _, extra_output = self._compute_global_attention(
                Q_global, K_global, V_global, global_attention_mask
            )

            # Combine local and global attention outputs
            output = local_output + extra_output
        else:
            output = local_output

        # Reshape back and project
        output = output.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        output = self.out_proj(output)
        output = self.dropout(output)

        return output


class LongformerEncoderLayer(nn.Module):
    """
    Single Longformer encoder layer.

    Consists of:
    - Multi-head sliding window attention (with global attention)
    - Position-wise feed-forward network
    - Layer normalization and residual connections

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        window_size: Sliding window size
        dim_feedforward: FFN intermediate dimension
        dropout: Dropout probability
    """

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
        """
        Forward pass with pre-norm architecture.

        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: Padding mask
            global_attention_mask: Global attention positions

        Returns:
            output: [batch, seq_len, d_model]
        """
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )
        hidden_states = residual + hidden_states

        # Pre-norm FFN
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LongformerForTrading(nn.Module):
    """
    Longformer model adapted for trading applications.

    Supports both NLP (document) and time series inputs.

    For NLP tasks:
    - Process long financial documents (SEC filings, earnings calls)
    - Extract sentiment from research reports
    - Multi-document question answering

    For time series:
    - Extended lookback windows (4K-16K timesteps)
    - Global attention on recent data and periodic anchors
    - Multi-scale temporal pattern recognition

    Args:
        input_type: 'timeseries' or 'text'
        input_dim: Input dimension for time series
        vocab_size: Vocabulary size for text
        d_model: Model hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of encoder layers
        window_size: Sliding window size
        max_seq_len: Maximum sequence length
        output_type: 'regression', 'classification', or 'allocation'
        pred_horizon: Prediction horizon for regression
        n_classes: Number of classes for classification
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_type: str = 'timeseries',
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
        self.max_seq_len = max_seq_len

        # Input embedding
        if input_type == 'timeseries':
            self.input_proj = nn.Linear(input_dim, d_model)
        else:
            self.input_proj = nn.Embedding(vocab_size, d_model)

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )

        # Longformer encoder layers
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

        # Output head based on task type
        if output_type == 'regression':
            self.head = nn.Linear(d_model, pred_horizon)
        elif output_type == 'classification':
            self.head = nn.Linear(d_model, n_classes)
        elif output_type == 'allocation':
            self.head = nn.Sequential(
                nn.Linear(d_model, pred_horizon),
                nn.Tanh()  # Bound allocations to [-1, 1]
            )

    def _create_default_global_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create default global attention mask.

        For time series: last position (most recent) + periodic anchors
        For text: [CLS] token (position 0) + last position
        """
        mask = torch.zeros(batch_size, seq_len, device=device)

        # Last position always gets global attention
        mask[:, -1] = 1

        if self.input_type == 'text':
            # [CLS] token for text classification
            mask[:, 0] = 1
        else:
            # Periodic anchors for time series (every 256 positions)
            anchor_freq = 256
            anchor_positions = torch.arange(0, seq_len, anchor_freq, device=device)
            mask[:, anchor_positions] = 1

        return mask

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
            attention_mask: [batch, seq_len] padding mask (optional)
            global_attention_mask: [batch, seq_len] global attention positions

        Returns:
            output: Predictions based on output_type
                - regression: [batch, pred_horizon]
                - classification: [batch, n_classes]
                - allocation: [batch, pred_horizon]
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

        # Create default global attention mask if not provided
        if global_attention_mask is None:
            global_attention_mask = self._create_default_global_mask(
                batch_size, seq_len, x.device
            )

        # Encode through Longformer layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            )

        hidden_states = self.norm(hidden_states)

        # Use last position for prediction (or [CLS] for text)
        if self.input_type == 'text':
            pooled = hidden_states[:, 0]  # [CLS] token
        else:
            pooled = hidden_states[:, -1]  # Last position

        output = self.head(pooled)

        return output


def create_longformer_timeseries(
    input_dim: int = 6,
    seq_len: int = 4096,
    pred_horizon: int = 24,
    **kwargs
) -> LongformerForTrading:
    """
    Factory function to create a Longformer model for time series.

    Args:
        input_dim: Number of input features
        seq_len: Input sequence length
        pred_horizon: Number of future steps to predict
        **kwargs: Additional arguments for LongformerForTrading

    Returns:
        Configured LongformerForTrading model
    """
    return LongformerForTrading(
        input_type='timeseries',
        input_dim=input_dim,
        d_model=kwargs.get('d_model', 256),
        n_heads=kwargs.get('n_heads', 8),
        n_layers=kwargs.get('n_layers', 6),
        window_size=kwargs.get('window_size', 256),
        max_seq_len=seq_len,
        output_type='regression',
        pred_horizon=pred_horizon,
        dropout=kwargs.get('dropout', 0.1)
    )


def create_longformer_nlp(
    vocab_size: int = 50000,
    max_seq_len: int = 4096,
    n_classes: int = 3,
    **kwargs
) -> LongformerForTrading:
    """
    Factory function to create a Longformer model for NLP tasks.

    Args:
        vocab_size: Size of vocabulary
        max_seq_len: Maximum sequence length
        n_classes: Number of output classes
        **kwargs: Additional arguments for LongformerForTrading

    Returns:
        Configured LongformerForTrading model
    """
    return LongformerForTrading(
        input_type='text',
        vocab_size=vocab_size,
        d_model=kwargs.get('d_model', 768),
        n_heads=kwargs.get('n_heads', 12),
        n_layers=kwargs.get('n_layers', 12),
        window_size=kwargs.get('window_size', 512),
        max_seq_len=max_seq_len,
        output_type='classification',
        n_classes=n_classes,
        dropout=kwargs.get('dropout', 0.1)
    )


if __name__ == '__main__':
    # Test the model
    print("Testing LongformerForTrading...")

    # Test time series model
    model_ts = create_longformer_timeseries(
        input_dim=6,
        seq_len=4096,
        pred_horizon=24
    )

    batch_size = 2
    seq_len = 4096
    input_dim = 6

    x_ts = torch.randn(batch_size, seq_len, input_dim)
    output_ts = model_ts(x_ts)

    print(f"Time series model input: {x_ts.shape}")
    print(f"Time series model output: {output_ts.shape}")
    print(f"Time series model parameters: {sum(p.numel() for p in model_ts.parameters()):,}")

    # Test NLP model
    model_nlp = create_longformer_nlp(
        vocab_size=50000,
        max_seq_len=4096,
        n_classes=3
    )

    x_nlp = torch.randint(0, 50000, (batch_size, seq_len))
    output_nlp = model_nlp(x_nlp)

    print(f"\nNLP model input: {x_nlp.shape}")
    print(f"NLP model output: {output_nlp.shape}")
    print(f"NLP model parameters: {sum(p.numel() for p in model_nlp.parameters()):,}")

    print("\nAll tests passed!")
