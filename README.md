# Trading Agent — Deep Reinforcement Learning

> A production-grade stock trading agent built on Double DQN with experience replay, technical indicators, and a full backtesting framework.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project trains a Deep Q-Network (DQN) agent to make buy, hold, and sell decisions on stock market data. It goes beyond the typical notebook prototype — every component is modular, configurable, and runnable from the command line with a single command.

The agent learns from OHLCV price data enriched with 15+ technical indicators, optimised using Double DQN with a target network and experience replay, and evaluated with risk-adjusted metrics including Sharpe ratio, Sortino ratio, max drawdown, and win rate.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Data Pipeline                          │
│  yfinance → OHLCV → TechnicalIndicators → Normalisation     │
│              RSI · MACD · Bollinger · ATR · OBV · Stoch     │
└─────────────────────┬───────────────────────────────────────┘
                      │ train / val / test split
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Trading Environment                        │
│  Window state (price + indicators + position + PnL)         │
│  Actions: HOLD=0  BUY=1  SELL=2                             │
│  Reward: Δ net worth · transaction cost penalty             │
└─────────────────────┬───────────────────────────────────────┘
                      │ (state, reward, done)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                     DQN Agent                               │
│  Policy Network  ──► Huber Loss ──► Adam Optimiser          │
│  Target Network  (synced every N steps)                     │
│  Replay Buffer   (uniform sampling, capacity 50k)           │
│  Exploration     (ε-greedy, exponential decay)              │
└─────────────────────┬───────────────────────────────────────┘
                      │ trained model
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Backtesting & Metrics                      │
│  Sharpe · Sortino · Calmar · Max Drawdown · Win Rate        │
│  Per-episode plots: price, portfolio value, drawdown        │
└─────────────────────────────────────────────────────────────┘
```

---

## Features

- **Double DQN** with target network to reduce Q-value overestimation
- **Experience Replay** with configurable buffer size (default 50k transitions)
- **15+ Technical Indicators** — RSI, MACD, Bollinger Bands, ATR, OBV, Stochastic, Williams %R, SMA, EMA
- **Realistic Environment** — transaction costs, multi-share position tracking, normalised state
- **Risk Metrics** — Sharpe ratio, Sortino ratio, Calmar ratio, max drawdown, win rate
- **CLI scripts** for training and backtesting, driven by a YAML config file
- **Model checkpointing** — saves best validation model automatically
- **Data caching** — downloaded data cached locally to avoid redundant API calls
- **Visualisation** — price chart with trade markers, portfolio value curve, drawdown chart

---

## Sample Results

Training curve and trade visualisation on AAPL:

![Training results](https://github.com/user-attachments/assets/294f0a06-bd49-47eb-bdbf-026b931e2f48)

![Backtest portfolio performance](https://github.com/user-attachments/assets/6384ea41-4f08-4e0c-ab15-fbe6cba91e44)

---

## Project Structure

```
trading-agent-rl/
├── src/
│   ├── agent.py          # Double DQN agent with replay buffer
│   ├── environment.py    # Trading environment (gym-style)
│   ├── indicators.py     # Technical indicators
│   ├── data.py           # Data loading, processing, normalisation
│   └── utils.py          # Metrics, plotting, formatting
├── scripts/
│   ├── train.py          # End-to-end training script
│   └── backtest.py       # Load a model and evaluate on any split
├── configs/
│   └── default.yaml      # All hyperparameters in one file
├── models/               # Saved model checkpoints
├── results/              # Plots and metric JSON files
├── requirements.txt
└── setup.py
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the agent

```bash
python scripts/train.py --config configs/default.yaml
```

Override any config value from the command line:

```bash
python scripts/train.py --ticker TSLA --episodes 100
```

### 3. Backtest a saved model

```bash
python scripts/backtest.py --model models/best_model --split test
```

Evaluate across all splits at once:

```bash
python scripts/backtest.py --model models/best_model --split all
```

---

## Configuration

All parameters live in `configs/default.yaml`. Key options:

| Parameter | Default | Description |
|---|---|---|
| `ticker` | `AAPL` | Stock symbol (any yfinance-supported ticker) |
| `start_date` | `2018-01-01` | Training data start |
| `end_date` | `2024-01-01` | Data end |
| `window_size` | `20` | Lookback window for state construction |
| `initial_balance` | `10000` | Starting portfolio value |
| `transaction_cost` | `0.001` | Cost per trade (0.1%) |
| `episodes` | `50` | Training episodes |
| `gamma` | `0.95` | Discount factor |
| `epsilon_decay` | `0.995` | Exploration decay per step |
| `batch_size` | `64` | Replay buffer sample size |
| `hidden_units` | `[256,256,128]` | Policy network architecture |

---

## Metrics

After each evaluation the agent reports:

| Metric | Description |
|---|---|
| Total Return | Net portfolio growth as a percentage |
| Sharpe Ratio | Risk-adjusted return (annualised) |
| Sortino Ratio | Downside-risk-adjusted return |
| Max Drawdown | Worst peak-to-trough decline |
| Calmar Ratio | Annual return divided by max drawdown |
| Win Rate | Percentage of profitable sell trades |
| N Trades | Total completed round-trip trades |

---

## Tech Stack

| Component | Library |
|---|---|
| Neural network | TensorFlow / Keras |
| Market data | yfinance |
| Data processing | pandas, NumPy |
| Visualisation | Matplotlib |
| Config | PyYAML |
| Progress | tqdm |

---

## Future Improvements

- Portfolio-level multi-asset trading with correlated reward signals
- Dueling DQN and Prioritised Experience Replay (PER)
- Transformer-based temporal feature encoder
- Bayesian hyperparameter optimisation
- Live paper-trading integration via broker API

---

## License

MIT

