# Глава 57: Longformer для Финансового Анализа

Эта глава исследует **Longformer** — архитектуру трансформера, использующую **скользящее окно внимания** в сочетании с **глобальным вниманием** для эффективной обработки длинных финансовых документов и временных рядов со сложностью O(n) вместо стандартной O(n²).

<p align="center">
<img src="https://i.imgur.com/9YKzL8B.png" width="70%">
</p>

## Содержание

1. [Введение в Longformer](#введение-в-longformer)
    * [Проблема длинных документов](#проблема-длинных-документов)
    * [Ключевая инновация: гибридное внимание](#ключевая-инновация-гибридное-внимание)
    * [Почему Longformer для финансов](#почему-longformer-для-финансов)
2. [Математические основы](#математические-основы)
    * [Стандартное self-attention](#стандартное-self-attention)
    * [Скользящее окно внимания](#скользящее-окно-внимания)
    * [Расширенное скользящее окно](#расширенное-скользящее-окно)
    * [Глобальное внимание](#глобальное-внимание)
3. [Архитектура Longformer](#архитектура-longformer)
    * [Дизайн паттернов внимания](#дизайн-паттернов-внимания)
    * [Анализ линейной сложности](#анализ-линейной-сложности)
    * [Детали реализации](#детали-реализации)
4. [Финансовые применения](#финансовые-применения)
    * [Анализ длинных финансовых документов](#анализ-длинных-финансовых-документов)
    * [Обработка расширенных временных рядов](#обработка-расширенных-временных-рядов)
    * [Анализ корреляций между активами](#анализ-корреляций-между-активами)
5. [Практические примеры](#практические-примеры)
    * [01: Подготовка данных](#01-подготовка-данных)
    * [02: Модель Longformer](#02-модель-longformer)
    * [03: Пайплайн обучения](#03-пайплайн-обучения)
    * [04: Бэктестинг стратегии](#04-бэктестинг-стратегии)
6. [Реализация на Rust](#реализация-на-rust)
7. [Реализация на Python](#реализация-на-python)
8. [Лучшие практики](#лучшие-практики)
9. [Ресурсы](#ресурсы)

## Введение в Longformer

### Проблема длинных документов

Стандартные трансформеры вычисляют внимание между всеми парами токенов, что приводит к сложности O(n²). Для финансовых приложений это становится проблемой:

```
Проблема обработки документов:

Длина финансовых документов:
- SEC 10-K отчёт:         ~50,000 токенов
- Транскрипт звонка:      ~10,000 токенов
- Аналитический отчёт:    ~5,000 токенов
- Новостная статья:       ~1,000 токенов

Стандартный трансформер (BERT):
- Максимум: 512 токенов
- 10-K отчёт потребует: 100+ частей!
- Контекст теряется между частями

Торговые временные ряды:
- 1 месяц минутных данных: 43,200 точек
- Стандартное внимание: 43,200² = 1.86 миллиарда операций!
```

### Ключевая инновация: гибридное внимание

Longformer вводит новую комбинацию двух паттернов внимания:

```
┌─────────────────────────────────────────────────────────────────────┐
│                 ПАТТЕРНЫ ВНИМАНИЯ LONGFORMER                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. СКОЛЬЗЯЩЕЕ ОКНО ВНИМАНИЯ (Локальное)                            │
│     Каждый токен обращает внимание на w токенов с каждой стороны    │
│                                                                       │
│     Токен:    [1] [2] [3] [4] [5] [6] [7] [8] [9] [10]              │
│     Окно=2:            ←─[4]─→                                       │
│                      видит [2,3,4,5,6]                               │
│                                                                       │
│  2. ГЛОБАЛЬНОЕ ВНИМАНИЕ (Специфичное для задачи)                    │
│     Определённые токены видят ВСЕ позиции                           │
│                                                                       │
│     [CLS] токен для классификации:                                  │
│     [CLS] ←→ [все токены]                                            │
│                                                                       │
│     Комбинированный паттерн:                                        │
│     ┌────────────────────────────┐                                   │
│     │▓░░░░░░░░░░░░░░░░░░░░░░░░░▓│ ← Глобальные (CLS, SEP)           │
│     │░▓▓▓░░░░░░░░░░░░░░░░░░░░░░░│                                    │
│     │░▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░│ ← Скользящее окно                  │
│     │░░▓▓▓▓░░░░░░░░░░░░░░░░░░░░░│   (диагональная полоса)            │
│     │░░░▓▓▓▓░░░░░░░░░░░░░░░░░░░░│                                    │
│     │░░░░▓▓▓▓░░░░░░░░░░░░░░░░░░░│                                    │
│     │      ...диагональ...      │                                    │
│     │▓░░░░░░░░░░░░░░░░░░░░░░░▓▓▓│                                    │
│     └────────────────────────────┘                                   │
│                                                                       │
│  Сложность: O(n × w) вместо O(n²)                                   │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Почему Longformer для финансов

| Применение | Стандартный трансформер | Longformer | Преимущество |
|------------|------------------------|------------|--------------|
| SEC 10-K отчёт | Обрезка до 512 токенов | Полные 50K токенов | Полный анализ документа |
| Звонки с инвесторами | Потеря контекста между частями | Один проход | Лучшее извлечение настроения |
| Минутный трейдинг | ~8 часов максимум | ~30 дней | Распознавание долгосрочных паттернов |
| Мульти-документный QA | Раздельная обработка | Совместное внимание | Межд-документное рассуждение |
| Новости + Цены | Ограниченный контекст | Расширенный контекст | Лучшая корреляция событий |

## Математические основы

### Стандартное self-attention

Механизм стандартного self-attention вычисляет:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

Где:
- **Q, K, V**: Матрицы Query, Key, Value ∈ ℝ^(n×d)
- **d_k**: Размерность ключа (обычно d_model / n_heads)
- **n**: Длина последовательности

Матрица внимания `QK^T` имеет размер n×n, требуя O(n²) времени и памяти.

### Скользящее окно внимания

Скользящее окно ограничивает каждый токен вниманием только к фиксированному окну соседних токенов:

```
Для токена на позиции i с размером окна w:
    Attention_i = {j : |i - j| ≤ w/2}

Пример с w=4 (окно из 4 токенов):
Позиция 5 видит: [3, 4, 5, 6, 7]

Паттерн внимания для w=4, n=10:
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

■ = внимание вычисляется, ░ = нет внимания

Сложность: O(n × w), где w << n
```

Рецептивное поле растёт линейно с количеством слоёв:
- Слой 1: w токенов
- Слой 2: 2w токенов
- Слой L: L×w токенов

Для L=12 слоёв и w=512, верхний слой имеет рецептивное поле 6,144 токена.

### Расширенное скользящее окно

Для увеличения рецептивного поля без добавления вычислений, Longformer поддерживает расширенные окна:

```
Расширенное окно с dilation d=2, w=4:

Стандартное (d=1): [1, 2, 3, 4, 5] → последовательные токены
Расширенное (d=2): [1, _, 3, _, 5, _, 7, _, 9] → пропуск каждого второго

Позиция 5 с d=2, w=4 видит: [1, 3, 5, 7, 9]

Это позволяет достичь позиции 9 с тем же количеством вычислений!

Рецептивное поле с расширением:
- Слой l с dilation d_l: d_l × w токенов
- Мульти-масштаб: нижние слои d=1 (локально), верхние d=2+ (глобально)
```

### Глобальное внимание

Некоторые токены требуют полного внимания к последовательности:

```python
# Глобальное внимание для определённых токенов
global_attention_mask = torch.zeros(batch_size, seq_len)

# Для классификации: [CLS] токен получает глобальное внимание
global_attention_mask[:, 0] = 1  # [CLS] на позиции 0

# Для QA: Токены вопроса получают глобальное внимание
global_attention_mask[:, question_start:question_end] = 1

# Глобальное внимание асимметрично:
# 1. Глобальные токены видят ВСЕ токены (полная строка)
# 2. ВСЕ токены видят глобальные токены (полный столбец)
```

Визуализация паттерна глобального внимания:

```
С глобальным вниманием на токенах 1 и 8 (G = глобальный):

        1  2  3  4  5  6  7  8  9  10
    1  [G  G  G  G  G  G  G  G  G  G]  ← Глобальный: полное внимание
    2  [G  ■  ■  ■  ░  ░  ░  G  ░  ░]
    3  [G  ■  ■  ■  ■  ░  ░  G  ░  ░]
    4  [G  ░  ■  ■  ■  ■  ░  G  ░  ░]
    5  [G  ░  ░  ■  ■  ■  ■  G  ░  ░]
    6  [G  ░  ░  ░  ■  ■  ■  G  ■  ░]
    7  [G  ░  ░  ░  ░  ■  ■  G  ■  ■]
    8  [G  G  G  G  G  G  G  G  G  G]  ← Глобальный: полное внимание
    9  [G  ░  ░  ░  ░  ░  ■  G  ■  ■]
   10  [G  ░  ░  ░  ░  ░  ░  G  ■  ■]
        ↑                    ↑
        │                    └─ Столбец: все видят глобальный
        └─ Столбец: все видят глобальный
```

## Архитектура Longformer

### Дизайн паттернов внимания

```python
class LongformerAttention(nn.Module):
    """
    Внимание Longformer, комбинирующее скользящее окно и глобальное внимание.

    Ключевые особенности:
    - Скользящее окно для локального контекста (O(n×w))
    - Глобальное внимание для специфичных токенов задачи
    - Настраиваемые паттерны внимания для каждого слоя
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

        # Раздельные проекции для локального и глобального внимания
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Глобальное внимание использует отдельные проекции
        self.q_global = nn.Linear(d_model, d_model)
        self.k_global = nn.Linear(d_model, d_model)
        self.v_global = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
```

### Анализ линейной сложности

```
Сравнение памяти и вычислений:

Стандартное внимание (seq_len=4096):
┌─────────────────────────────────────────┐
│ Матрица внимания: 4096 × 4096 = 16.7M   │
│ Память: ~64MB на голову (fp32)          │
│ Вычисления: O(n²) = O(16.7M)            │
└─────────────────────────────────────────┘

Longformer (seq_len=4096, window=512):
┌─────────────────────────────────────────┐
│ Локальное внимание: 4096 × 512 = 2.1M   │
│ + Глобальное (100 токенов): 100×4096×2  │
│                       = 0.8M            │
│ Всего: ~2.9M операций                   │
│ Память: ~12MB на голову (fp32)          │
│ Вычисления: O(n × w) = O(2.1M)          │
└─────────────────────────────────────────┘

Ускорение: ~5.8× быстрее, ~5× меньше памяти

Для длинных последовательностей (seq_len=16384, window=512):
Стандартное: 268M операций
Longformer: 8.4M операций
Ускорение: ~32× быстрее!
```

### Детали реализации

```python
class LongformerLayer(nn.Module):
    """Один слой энкодера Longformer."""

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

        # Внимание с настраиваемым окном/расширением для слоя
        self.attention = LongformerAttention(
            d_model=d_model,
            n_heads=n_heads,
            window_size=window_size,
            dropout=dropout
        )

        # Сеть прямого распространения
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
        Прямой проход.

        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: Маска паддинга
            global_attention_mask: Токены с глобальным вниманием
        """
        # Self-attention с резидуальным соединением
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        attn_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )
        hidden_states = residual + attn_output

        # FFN с резидуальным соединением
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = residual + self.ffn(hidden_states)

        return hidden_states
```

## Финансовые применения

### Анализ длинных финансовых документов

Обработка полных SEC отчётов, аналитических материалов и транскриптов:

```python
class LongformerFinancialNLP(nn.Module):
    """
    Longformer для анализа финансовых документов.

    Применения:
    - Анализ настроения SEC 10-K/10-Q
    - Анализ звонков с инвесторами
    - Суммаризация аналитических отчётов
    - Мульти-документные вопросы и ответы
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        window_size: int = 512,
        max_position_embeddings: int = 16384,
        num_labels: int = 3,  # негативный, нейтральный, позитивный
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
        self.classifier = nn.Linear(d_model, num_labels)
```

### Обработка расширенных временных рядов

Адаптация архитектуры Longformer для финансовых временных рядов:

```python
class LongformerTimeSeries(nn.Module):
    """
    Longformer адаптированный для финансовых временных рядов.

    Ключевые адаптации:
    - Непрерывные входные эмбеддинги вместо токенов
    - Глобальное внимание на недавних данных и периодических якорях
    - Мульти-масштабные временные паттерны через расширенные окна
    """

    def __init__(
        self,
        input_dim: int = 6,          # OHLCV + признаки
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        window_size: int = 256,      # Окно локального внимания
        max_seq_len: int = 8192,
        pred_horizon: int = 24,
        global_token_freq: int = 256,  # Глобальный токен каждые N позиций
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.window_size = window_size
        self.global_token_freq = global_token_freq

        # Проекция входа
        self.input_proj = nn.Linear(input_dim, d_model)

        # Обучаемое позиционное кодирование для длинных последовательностей
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )

        # Слои Longformer с увеличивающимся расширением
        self.layers = nn.ModuleList([
            LongformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                window_size=window_size,
                dilation=min(2 ** (i // 2), 8),
                dropout=dropout,
                layer_id=i
            )
            for i in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_horizon)
        )
```

### Анализ корреляций между активами

Анализ корреляций между несколькими активами с длинной историей:

```python
class LongformerMultiAsset(nn.Module):
    """
    Мульти-активный анализ с Longformer.

    Обработка нескольких временных рядов активов совместно с:
    - Скользящее окно на актив (временные паттерны)
    - Кросс-активное глобальное внимание (корреляции)
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

        # Эмбеддинги активов
        self.asset_embeddings = nn.Embedding(n_assets, d_model)
        self.input_proj = nn.Linear(features_per_asset, d_model)

        # Выход: веса аллокации для каждого актива
        self.allocation_head = nn.Sequential(
            nn.Linear(d_model * n_assets, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_assets),
            nn.Softmax(dim=-1)  # Веса портфеля суммируются в 1
        )
```

## Практические примеры

### 01: Подготовка данных

```python
# python/data.py

import numpy as np
import pandas as pd
import requests
from typing import List, Tuple, Dict
from transformers import AutoTokenizer

def prepare_financial_documents(
    documents: List[str],
    tokenizer_name: str = 'allenai/longformer-base-4096',
    max_length: int = 4096
) -> Dict[str, np.ndarray]:
    """
    Подготовка финансовых документов для обработки Longformer.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    encodings = tokenizer(
        documents,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )

    # Маска глобального внимания (1 для [CLS], 0 для остальных)
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
    Подготовка длинных временных рядов для Longformer.
    """
    all_features = []

    for symbol in symbols:
        if source == 'bybit':
            df = load_bybit_data(symbol, interval='1m')
        else:
            df = load_yahoo_data(symbol)

        # Расчёт признаков
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(20).std()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(50).mean()
        df['price_ma_ratio'] = df['close'] / df['close'].rolling(200).mean()
        df['rsi'] = calculate_rsi(df['close'], period=14)
        df['macd'] = calculate_macd(df['close'])

        all_features.append(df)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_bybit_data(symbol: str, interval: str = '1m') -> pd.DataFrame:
    """Загрузка исторических данных с Bybit."""
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
```

### 02: Модель Longformer

```python
# python/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

class LongformerForTrading(nn.Module):
    """
    Модель Longformer для торговых приложений.

    Поддерживает как NLP (документы), так и временные ряды.
    """

    def __init__(
        self,
        input_type: str = 'timeseries',  # 'timeseries' или 'text'
        input_dim: int = 6,
        vocab_size: int = 50000,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        window_size: int = 256,
        max_seq_len: int = 4096,
        output_type: str = 'regression',
        pred_horizon: int = 24,
        n_classes: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_type = input_type
        self.output_type = output_type
        self.d_model = d_model

        # Эмбеддинг входа
        if input_type == 'timeseries':
            self.input_proj = nn.Linear(input_dim, d_model)
        else:
            self.input_proj = nn.Embedding(vocab_size, d_model)

        # Позиционное кодирование
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )

        # Слои Longformer
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

        # Выходная голова
        if output_type == 'regression':
            self.head = nn.Linear(d_model, pred_horizon)
        elif output_type == 'classification':
            self.head = nn.Linear(d_model, n_classes)
        elif output_type == 'allocation':
            self.head = nn.Sequential(
                nn.Linear(d_model, pred_horizon),
                nn.Tanh()
            )
```

### 03: Пайплайн обучения

```python
# python/train.py

class LongformerTrainer:
    """Пайплайн обучения для торговой модели Longformer."""

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

        # Функция потерь в зависимости от типа выхода
        if model.output_type == 'regression':
            self.criterion = nn.MSELoss()
        elif model.output_type == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif model.output_type == 'allocation':
            self.criterion = self._sharpe_loss

    def _sharpe_loss(self, allocations, returns):
        """Дифференцируемая функция потерь Шарпа."""
        portfolio_returns = allocations * returns
        mean_ret = portfolio_returns.mean()
        std_ret = portfolio_returns.std() + 1e-8
        return -mean_ret / std_ret

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 100,
        patience: int = 10,
        save_dir: str = 'checkpoints'
    ):
        """Полный цикл обучения."""
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                torch.save(self.model.state_dict(), f'{save_dir}/longformer_best.pt')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        return history
```

### 04: Бэктестинг стратегии

```python
# python/strategy.py

class LongformerBacktester:
    """Бэктестинг для торговой стратегии Longformer."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: BacktestConfig = BacktestConfig()
    ):
        self.model = model
        self.config = config
        self.model.eval()

    @torch.no_grad()
    def generate_signals(self, data, threshold=0.001):
        """Генерация торговых сигналов."""
        x = torch.tensor(data, dtype=torch.float32)
        predictions = self.model(x).numpy()
        pred_returns = predictions[:, 0]

        signals = np.zeros_like(pred_returns)
        signals[pred_returns > threshold] = 1
        signals[pred_returns < -threshold] = -1

        return signals

    def run_backtest(self, data, prices, timestamps):
        """Запуск симуляции бэктеста."""
        signals = self.generate_signals(data)
        # ... логика бэктестинга ...
        return results

    def _calculate_metrics(self, returns, equity_curve, trades):
        """Расчёт метрик производительности."""
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'num_trades': len(trades),
            'final_capital': equity_curve[-1]
        }
```

## Реализация на Rust

Смотрите [rust/](rust/) для полной реализации на Rust.

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

### Быстрый старт (Rust)

```bash
cd rust

# Загрузка данных с Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT

# Обучение модели
cargo run --example train -- --epochs 50 --batch-size 16

# Запуск бэктеста
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Реализация на Python

Смотрите [python/](python/) для реализации на Python.

```
python/
├── __init__.py
├── model.py           # Реализация Longformer
├── data.py            # Загрузка данных (Bybit, Yahoo)
├── train.py           # Пайплайн обучения
├── strategy.py        # Бэктестинг
├── requirements.txt
└── examples/
    └── longformer_trading.ipynb
```

### Быстрый старт (Python)

```bash
cd python
pip install -r requirements.txt

# Загрузка и обучение
python train.py --symbols BTCUSDT,ETHUSDT --epochs 100

# Бэктест
python strategy.py --model checkpoints/longformer_best.pt
```

## Применения Longformer для финансового NLP

### Классификация настроения на уровне документа

Основная задача финансового NLP для Longformer -- классификация настроения целых документов -- транскриптов звонков с инвесторами, аналитических отчётов или новостных статей -- без обрезки. Для документа, токенизированного в $\{x_1, x_2, \ldots, x_n\}$ с $n \leq 4096$, Longformer кодирует последовательность используя смешанный паттерн внимания. Представление [CLS] токена подаётся в классификационную голову:

$$\hat{y} = \text{softmax}(\mathbf{W}\mathbf{h}_{\text{CLS}} + \mathbf{b})$$

Классы обычно включают позитивное, негативное и нейтральное настроение.

### Распознавание именованных сущностей в финансовых документах

Longformer превосходит в финансовом NER, поскольку контекст сущностей часто простирается на большие расстояния. Например, название компании на первой странице 10-K может быть связано с фактором риска на десятой странице.

Каждый токен $x_i$ классифицируется в BIO-теги:

$$\hat{y}_i = \text{softmax}(\mathbf{W}_{\text{ner}}\mathbf{h}_i + \mathbf{b}_{\text{ner}})$$

Типы сущностей для финансов: ORG (организации), MONEY (денежные суммы), PERCENT (проценты), DATE (даты), PRODUCT (финансовые инструменты), EVENT (рыночные события).

### Обнаружение факторов риска

Регуляторные отчёты содержат раскрытия рисков, существенные для инвестиционных решений. Longformer может классифицировать каждый параграф как содержащий фактор риска или нет:

$$P(\text{risk} | \text{paragraph}_i) = \sigma(\mathbf{w}^T \mathbf{h}_i + b)$$

### Пайплайн предобучения и дообучения

1. **Предобучение**: Начать с предобученного чекпоинта Longformer (напр., `allenai/longformer-base-4096`).
2. **Адаптация к домену**: Продолжить предобучение на финансовом корпусе (SEC отчёты, финансовые новости) с маскированным языковым моделированием.
3. **Дообучение**: Обучить на целевой задаче (классификация, NER, настроение) со специфичными головами.

### Конструирование признаков для финансового NLP

**Признаки уровня токенов:**
- Финансовые числа: денежные значения, проценты, коэффициенты
- Временные выражения: фискальные кварталы, сравнения год к году
- Юридические/регуляторные термины: "существенное неблагоприятное воздействие", "непрерывность деятельности"
- Индикаторы настроения: "превзошли ожидания", "встречные ветры", "устойчивый рост"

**Признаки структуры документа:**
- Заголовки разделов: маппинг на глобальные токены внимания
- Маркеры таблиц: финансовые таблицы содержат критичные количественные данные
- Ссылки на сноски: соединение содержимого сносок с основным текстом

**Агрегированные признаки настроения для трейдинга:**
- Оценка настроения: непрерывное значение в $[-1, 1]$
- Моментум настроения: $\Delta S_t = S_t - S_{t-1}$
- Кросс-документный консенсус: среднее настроение по нескольким источникам
- Сюрприз настроения: отклонение от ожидаемого настроения

### Конкретные области применения NLP

**Анализ звонков с инвесторами:** Транскрипты 5,000-15,000 слов. Longformer может выполнять полный анализ настроения транскрипта, сравнение тона между разделами и обнаружение прогнозных высказываний.

**Анализ регуляторных отчётов:** 10-K и 10-Q содержат критичную информацию: извлечение факторов риска, интерпретация MD&A и обнаружение существенных событий.

**Анализ криптоновостей:** Оценка технической сложности вайтпейперов, анализ воздействия предложений по управлению и агрегация многоисточниковых потоков.

### Метрики оценки NLP задач

- **Классификация**: Accuracy, F1-score (macro и weighted), AUC-ROC
- **NER**: F1 на уровне сущностей, precision, recall
- **Настроение**: Cohen's Kappa, направленная точность для торговых сигналов
- **Торговая производительность**: коэффициент Шарпа, коэффициент Сортино, максимальная просадка

## Лучшие практики

### Когда использовать Longformer

**Идеальные случаи:**
- Длинные документы: SEC отчёты, аналитика, транскрипты
- Расширенные временные ряды: 4K-16K точек
- Задачи требующие локального и глобального контекста
- QA по документам с длинным исходным текстом

**Рассмотрите альтернативы когда:**
- Короткие последовательности (<512): Стандартный трансформер подойдёт
- Чисто локальные паттерны: Может хватить только скользящего окна
- Очень разреженные глобальные зависимости: Используйте sparse attention

### Рекомендации по гиперпараметрам

| Параметр | Рекомендация | Примечания |
|----------|--------------|------------|
| `window_size` | 256-512 | Больше для бо́льшего локального контекста |
| `max_seq_len` | 4096-16384 | Зависит от длины документа/ряда |
| `n_layers` | 6-12 | Больше для сложных паттернов |
| `global_attention` | Зависит от задачи | [CLS] для классификации, недавние для рядов |

### Выбор размера окна

```
Правило выбора размера окна:

Длина документа → Размер окна
- 4096 токенов  → 256-512
- 8192 токенов  → 512
- 16384 токенов → 512-1024

Для временных рядов (минутные данные):
- 1 день (1440 мин)  → 128-256
- 1 неделя (10080 мин) → 256-512
- 1 месяц (43200 мин) → 512-1024
```

## Ресурсы

### Научные статьи

- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — Оригинальная статья (2020)
- [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — Родственная архитектура
- [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732) — Обзор эффективных трансформеров

### Реализации

- [HuggingFace Transformers](https://huggingface.co/docs/transformers/model_doc/longformer) — Официальная реализация
- [AllenAI Longformer](https://github.com/allenai/longformer) — Оригинальный репозиторий

### Дополнительные ссылки по NLP

- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. arXiv:1908.10063.
- Huang, A. H., Wang, H., & Yang, Y. (2023). FinBERT: A Large Language Model for Extracting Information from Financial Text. Contemporary Accounting Research, 40(2), 806-841.
- Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. The Journal of Finance, 66(1), 35-65.
- Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692.

### Связанные главы

- [Глава 51: Linformer Long Sequences](../51_linformer_long_sequences) — Линейное внимание
- [Глава 52: Performer Efficient Attention](../52_performer_efficient_attention) — Случайные признаки
- [Глава 53: BigBird Sparse Attention](../53_bigbird_sparse_attention) — Похожий разреженный подход
- [Глава 56: Nystromformer Trading](../56_nystromformer_trading) — Аппроксимация Нистрома

---

## Уровень сложности

**Средний**

Пререквизиты:
- Понимание self-attention трансформера
- Базовые знания NLP и токенизации
- Знакомство с прогнозированием временных рядов
- Опыт с PyTorch или Rust ML библиотеками
