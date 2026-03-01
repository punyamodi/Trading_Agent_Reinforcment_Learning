import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns - risk_free / periods_per_year
    std = np.std(excess)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def max_drawdown(portfolio_values: np.ndarray) -> float:
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / np.maximum(peak, 1e-8)
    return float(np.min(drawdown))


def calmar_ratio(returns: np.ndarray, portfolio_values: np.ndarray, periods_per_year: int = 252) -> float:
    annual_return = np.mean(returns) * periods_per_year
    mdd = abs(max_drawdown(portfolio_values))
    if mdd == 0:
        return 0.0
    return float(annual_return / mdd)


def sortino_ratio(returns: np.ndarray, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-8
    if downside_std == 0:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))


def win_rate(trade_log: pd.DataFrame) -> float:
    sells = trade_log[trade_log["action"] == "SELL"] if not trade_log.empty else pd.DataFrame()
    if sells.empty or "profit" not in sells.columns:
        return 0.0
    winners = (sells["profit"] > 0).sum()
    return float(winners / len(sells))


def compute_metrics(portfolio_values: np.ndarray, trade_log: pd.DataFrame) -> dict:
    returns = np.diff(portfolio_values) / np.maximum(portfolio_values[:-1], 1e-8)
    return {
        "total_return_pct": float((portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100),
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "max_drawdown_pct": float(max_drawdown(portfolio_values) * 100),
        "calmar_ratio": calmar_ratio(returns, portfolio_values),
        "win_rate_pct": win_rate(trade_log) * 100,
        "n_trades": len(trade_log[trade_log["action"] == "SELL"]) if not trade_log.empty else 0,
        "final_portfolio": float(portfolio_values[-1]),
    }


def format_currency(value: float) -> str:
    if value >= 0:
        return f"${value:,.2f}"
    return f"-${abs(value):,.2f}"


def plot_trading_results(
    close_prices: np.ndarray,
    portfolio_values: np.ndarray,
    trade_log: pd.DataFrame,
    episode: int,
    save_path: Optional[str] = None,
):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(f"Trading Agent Results  Episode {episode}", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    ax1.plot(close_prices, color="steelblue", linewidth=1, label="Close Price")
    if not trade_log.empty:
        buys = trade_log[trade_log["action"] == "BUY"]
        sells = trade_log[trade_log["action"] == "SELL"]
        if not buys.empty:
            ax1.scatter(buys["step"].values, close_prices[buys["step"].values.clip(0, len(close_prices) - 1)],
                        marker="^", color="green", s=60, zorder=5, label="Buy")
        if not sells.empty:
            ax1.scatter(sells["step"].values, close_prices[sells["step"].values.clip(0, len(close_prices) - 1)],
                        marker="v", color="red", s=60, zorder=5, label="Sell")
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(portfolio_values, color="darkorange", linewidth=1.5, label="Portfolio Value")
    ax2.axhline(portfolio_values[0], color="gray", linestyle="--", linewidth=0.8, label="Initial Balance")
    ax2.set_ylabel("Portfolio Value ($)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3)

    ax3 = axes[2]
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / np.maximum(peak, 1e-8) * 100
    ax3.fill_between(range(len(drawdown)), drawdown, 0, color="crimson", alpha=0.5, label="Drawdown")
    ax3.set_ylabel("Drawdown (%)")
    ax3.set_xlabel("Time Step")
    ax3.legend(loc="lower left", fontsize=8)
    ax3.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def print_metrics(metrics: dict, prefix: str = ""):
    width = 40
    border = "=" * width
    print(border)
    if prefix:
        print(f"  {prefix}")
        print(border)
    print(f"  Total Return        : {metrics['total_return_pct']:+.2f}%")
    print(f"  Final Portfolio     : {format_currency(metrics['final_portfolio'])}")
    print(f"  Sharpe Ratio        : {metrics['sharpe_ratio']:.4f}")
    print(f"  Sortino Ratio       : {metrics['sortino_ratio']:.4f}")
    print(f"  Max Drawdown        : {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Calmar Ratio        : {metrics['calmar_ratio']:.4f}")
    print(f"  Win Rate            : {metrics['win_rate_pct']:.1f}%")
    print(f"  Total Trades        : {metrics['n_trades']}")
    print(border)
