import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import DQNAgent
from src.data import DataLoader
from src.environment import TradingEnvironment
from src.utils import compute_metrics, plot_trading_results, print_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_episode(env: TradingEnvironment, agent: DQNAgent, training: bool = True):
    state = env.reset()
    portfolio_values = [env.initial_balance]
    total_loss = 0.0
    loss_count = 0
    done = False

    while not done:
        action = agent.act(state, training=training)
        next_state, reward, done, info = env.step(action)
        portfolio_values.append(info["net_worth"])

        if training:
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train()
            if loss is not None:
                total_loss += loss
                loss_count += 1

        state = next_state

    avg_loss = total_loss / max(loss_count, 1)
    return np.array(portfolio_values), env.get_trade_log(), avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train DQN Trading Agent")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--ticker", type=str, help="Override ticker symbol")
    parser.add_argument("--episodes", type=int, help="Override number of episodes")
    parser.add_argument("--output", type=str, help="Override model output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.ticker:
        cfg["ticker"] = args.ticker
    if args.episodes:
        cfg["episodes"] = args.episodes
    if args.output:
        cfg["model_dir"] = args.output

    logger.info("Loading data for %s from %s to %s", cfg["ticker"], cfg["start_date"], cfg["end_date"])
    loader = DataLoader(
        ticker=cfg["ticker"],
        start=cfg["start_date"],
        end=cfg["end_date"],
        cache_dir=cfg.get("data_cache_dir"),
    )
    loader.process()
    train_df, val_df, test_df = loader.split(cfg["train_ratio"], cfg["val_ratio"])

    train_norm, norm_stats = DataLoader.normalize(train_df)
    val_norm, _ = DataLoader.normalize(val_df, norm_stats)
    test_norm, _ = DataLoader.normalize(test_df, norm_stats)

    logger.info("Train: %d  Val: %d  Test: %d rows", len(train_df), len(val_df), len(test_df))

    env_kwargs = dict(
        initial_balance=cfg["initial_balance"],
        transaction_cost=cfg["transaction_cost"],
        window_size=cfg["window_size"],
        reward_scaling=cfg["reward_scaling"],
    )
    train_env = TradingEnvironment(train_norm, **env_kwargs)
    val_env = TradingEnvironment(val_norm, **env_kwargs)

    agent = DQNAgent(
        state_size=train_env.state_size,
        action_size=train_env.action_size,
        hidden_units=cfg.get("hidden_units", [256, 256, 128]),
        learning_rate=cfg["learning_rate"],
        gamma=cfg["gamma"],
        epsilon=cfg["epsilon"],
        epsilon_min=cfg["epsilon_min"],
        epsilon_decay=cfg["epsilon_decay"],
        buffer_size=cfg["buffer_size"],
        batch_size=cfg["batch_size"],
        target_update_freq=cfg["target_update_freq"],
    )

    model_dir = Path(cfg["model_dir"])
    results_dir = Path(cfg["results_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    best_val_return = -np.inf
    training_history = []

    logger.info("Starting training for %d episodes", cfg["episodes"])
    for ep in range(1, cfg["episodes"] + 1):
        portfolio_values, trade_log, avg_loss = run_episode(train_env, agent, training=True)
        train_metrics = compute_metrics(portfolio_values, trade_log)
        training_history.append({"episode": ep, **train_metrics, "avg_loss": avg_loss, "epsilon": agent.epsilon})

        if ep % cfg.get("log_every", 5) == 0:
            logger.info(
                "Ep %3d/%d  Return: %+.2f%%  Sharpe: %.3f  Trades: %d  Loss: %.4f  Eps: %.3f",
                ep, cfg["episodes"],
                train_metrics["total_return_pct"],
                train_metrics["sharpe_ratio"],
                train_metrics["n_trades"],
                avg_loss,
                agent.epsilon,
            )

        if ep % cfg.get("save_every", 10) == 0:
            val_portfolio, val_trade_log, _ = run_episode(val_env, agent, training=False)
            val_metrics = compute_metrics(val_portfolio, val_trade_log)
            logger.info("Val Ep %d  Return: %+.2f%%  Sharpe: %.3f", ep, val_metrics["total_return_pct"], val_metrics["sharpe_ratio"])

            if val_metrics["total_return_pct"] > best_val_return:
                best_val_return = val_metrics["total_return_pct"]
                agent.save(str(model_dir / "best_model"))
                logger.info("Best model saved (val return: %+.2f%%)", best_val_return)

            plot_trading_results(
                close_prices=val_env.close_prices,
                portfolio_values=val_portfolio,
                trade_log=val_trade_log,
                episode=ep,
                save_path=str(results_dir / f"val_ep{ep:04d}.png"),
            )

    agent.save(str(model_dir / "final_model"))

    history_path = results_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    logger.info("Training history saved to %s", history_path)

    logger.info("Evaluating on test set...")
    test_env = TradingEnvironment(test_norm, **env_kwargs)
    agent.load(str(model_dir / "best_model"))
    test_portfolio, test_trade_log, _ = run_episode(test_env, agent, training=False)
    test_metrics = compute_metrics(test_portfolio, test_trade_log)
    print_metrics(test_metrics, prefix=f"Test Results  {cfg['ticker']}")

    plot_trading_results(
        close_prices=test_env.close_prices,
        portfolio_values=test_portfolio,
        trade_log=test_trade_log,
        episode=0,
        save_path=str(results_dir / "test_results.png"),
    )

    metrics_path = results_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info("Test metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
