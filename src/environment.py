import numpy as np
import pandas as pd
from typing import Tuple, Optional


HOLD = 0
BUY = 1
SELL = 2


class TradingEnvironment:
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,
        window_size: int = 10,
        reward_scaling: float = 1e-4,
    ):
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        self.reward_scaling = reward_scaling

        self.feature_cols = [c for c in data.columns if c not in ["Open", "High", "Low", "Volume"]]
        self.n_features = len(self.feature_cols) + 2

        self.close_prices = data["Close"].values.astype(np.float32)

        self._current_step: int = 0
        self._balance: float = initial_balance
        self._shares_held: int = 0
        self._buy_price: float = 0.0
        self._net_worth: float = initial_balance
        self._prev_net_worth: float = initial_balance
        self._total_profit: float = 0.0
        self._trade_log: list = []

    @property
    def state_size(self) -> int:
        return self.window_size * self.n_features

    @property
    def action_size(self) -> int:
        return 3

    def reset(self) -> np.ndarray:
        self._current_step = self.window_size
        self._balance = self.initial_balance
        self._shares_held = 0
        self._buy_price = 0.0
        self._net_worth = self.initial_balance
        self._prev_net_worth = self.initial_balance
        self._total_profit = 0.0
        self._trade_log = []
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        current_price = self.close_prices[self._current_step]
        cost = 0.0
        reward = 0.0

        if action == BUY and self._shares_held == 0 and self._balance >= current_price:
            shares_to_buy = int(self._balance / (current_price * (1 + self.transaction_cost)))
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * self.transaction_cost
                self._balance -= shares_to_buy * current_price + cost
                self._shares_held = shares_to_buy
                self._buy_price = current_price
                self._trade_log.append({"step": self._current_step, "action": "BUY", "price": current_price, "shares": shares_to_buy})

        elif action == SELL and self._shares_held > 0:
            cost = self._shares_held * current_price * self.transaction_cost
            proceeds = self._shares_held * current_price - cost
            profit = (current_price - self._buy_price) * self._shares_held - cost
            self._balance += proceeds
            self._total_profit += profit
            self._trade_log.append({"step": self._current_step, "action": "SELL", "price": current_price, "profit": profit})
            self._shares_held = 0
            self._buy_price = 0.0

        self._net_worth = self._balance + self._shares_held * current_price
        reward = (self._net_worth - self._prev_net_worth) * self.reward_scaling
        self._prev_net_worth = self._net_worth

        self._current_step += 1
        done = self._current_step >= len(self.data) - 1

        info = {
            "net_worth": self._net_worth,
            "balance": self._balance,
            "shares_held": self._shares_held,
            "total_profit": self._total_profit,
            "current_price": current_price,
        }

        return self._get_state(), reward, done, info

    def _get_state(self) -> np.ndarray:
        start = self._current_step - self.window_size
        end = self._current_step

        window = self.data[self.feature_cols].iloc[start:end].values.astype(np.float32)
        position = np.full((self.window_size, 1), self._shares_held > 0, dtype=np.float32)
        pnl = np.full((self.window_size, 1), (self._net_worth - self.initial_balance) / self.initial_balance, dtype=np.float32)

        state = np.concatenate([window, position, pnl], axis=1)
        return state.flatten()

    def get_trade_log(self) -> pd.DataFrame:
        return pd.DataFrame(self._trade_log)

    def get_portfolio_value(self) -> float:
        return self._net_worth

    def get_total_return(self) -> float:
        return (self._net_worth - self.initial_balance) / self.initial_balance
