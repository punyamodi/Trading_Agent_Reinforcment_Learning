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


def main():
    parser = argparse.ArgumentParser(description="Backtest DQN Trading Agent")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--model", required=True, help="Path to saved model directory")
    parser.add_argument("--ticker", type=str, help="Override ticker symbol")
    parser.add_argument("--start", type=str, help="Override start date")
    parser.add_argument("--end", type=str, help="Override end date")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test",
                        help="Which data split to evaluate on")
    parser.add_argument("--output", type=str, default="results", help="Output directory for results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.ticker:
        cfg["ticker"] = args.ticker
    if args.start:
        cfg["start_date"] = args.start
    if args.end:
        cfg["end_date"] = args.end

    logger.info("Loading data for %s", cfg["ticker"])
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

    split_map = {"train": train_norm, "val": val_norm, "test": test_norm}
    splits_to_eval = list(split_map.items()) if args.split == "all" else [(args.split, split_map[args.split])]

    env_kwargs = dict(
        initial_balance=cfg["initial_balance"],
        transaction_cost=cfg["transaction_cost"],
        window_size=cfg["window_size"],
        reward_scaling=cfg["reward_scaling"],
    )

    dummy_env = TradingEnvironment(train_norm, **env_kwargs)
    agent = DQNAgent(
        state_size=dummy_env.state_size,
        action_size=dummy_env.action_size,
        hidden_units=cfg.get("hidden_units", [256, 256, 128]),
        epsilon_min=0.0,
        epsilon=0.0,
    )
    agent.load(args.model)
    logger.info("Model loaded from %s", args.model)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for split_name, data in splits_to_eval:
        logger.info("Evaluating on %s split...", split_name)
        env = TradingEnvironment(data, **env_kwargs)

        state = env.reset()
        portfolio_values = [env.initial_balance]
        done = False

        while not done:
            action = agent.act(state, training=False)
            next_state, _, done, info = env.step(action)
            portfolio_values.append(info["net_worth"])
            state = next_state

        trade_log = env.get_trade_log()
        metrics = compute_metrics(np.array(portfolio_values), trade_log)
        all_results[split_name] = metrics

        print_metrics(metrics, prefix=f"{cfg['ticker']}  {split_name.upper()} SET")

        plot_trading_results(
            close_prices=env.close_prices,
            portfolio_values=np.array(portfolio_values),
            trade_log=trade_log,
            episode=0,
            save_path=str(output_dir / f"backtest_{split_name}.png"),
        )

        if not trade_log.empty:
            trade_log.to_csv(output_dir / f"trades_{split_name}.csv", index=False)

    results_path = output_dir / "backtest_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
