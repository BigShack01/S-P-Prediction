import os
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# Trading Environment
# =========================
class TradingEnvIndicators(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, ticker="SPY", start=None, end=None):
        super().__init__()
        self.ticker = ticker

        # -------------------------
        # Download data
        # -------------------------
        if start is None and end is None:
            self.data = yf.download(ticker, period="2y", interval="1d",
                                    auto_adjust=True, progress=False).dropna()
        else:
            self.data = yf.download(ticker, start=start, end=end, interval="1d",
                                    auto_adjust=True, progress=False).dropna()

        if len(self.data) < 60:
            raise ValueError(f"Not enough data for {ticker}. Need at least 60 rows, got {len(self.data)}")

        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.get_level_values(0)

        self.data["Close"] = self.data["Close"].astype(float)
        close = self.data["Close"]

        # -------------------------
        # Compute indicators
        # -------------------------
        self.data["SMA_5"] = ta.trend.sma_indicator(close, window=5)
        self.data["SMA_20"] = ta.trend.sma_indicator(close, window=20)
        self.data["SMA_50"] = ta.trend.sma_indicator(close, window=50)
        self.data["RSI"] = ta.momentum.rsi(close, window=14)
        self.data["MACD"] = ta.trend.macd(close)
        self.data["ATR"] = ta.volatility.average_true_range(self.data["High"], self.data["Low"], self.data["Close"], window=14)
        self.data["Close_slope"] = close.diff()
        self.data["RSI_delta"] = self.data["RSI"].diff()
        self.data["Volatility"] = close.rolling(5).std()
        self.data["Volume_delta"] = self.data["Volume"].pct_change()

        # Drop only critical NaNs
        self.data = self.data.dropna(subset=["SMA_20", "SMA_50"]).reset_index(drop=True)

        # Observation & action spaces
        self.observation_space = spaces.Box(low=-5, high=5, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell

        self.initial_balance = 10_000.0
        self.max_step = len(self.data) - 2
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.hold_duration = 0
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.data.iloc[self.current_step]
        obs = np.array([
            (row["Close"] / row["SMA_20"] - 1.0) / 0.05,
            (row["SMA_5"] / row["SMA_20"] - 1.0) / 0.05,
            (row["SMA_20"] / row["SMA_50"] - 1.0) / 0.05,
            (row["RSI"] - 50) / 50.0,
            row["MACD"] / 5.0,
            (row["ATR"] / row["Close"]) / 0.01,
            (row["Close_slope"] / row["Close"]) / 0.01,
            (row["RSI_delta"] / 100.0) / 0.1,
            (row["Volatility"] / row["Close"]) / 0.01,
            row["Volume_delta"] / 0.05
        ], dtype=np.float32)
        return np.clip(obs, -5.0, 5.0)

    def step(self, action):
        current_price = float(self.data.iloc[self.current_step]["Close"])
        prev_net_worth = self.net_worth
        reward = 0.0

        # Execute action
        if action == 1 and self.balance >= current_price:  # Buy
            shares = int((self.balance * 0.2) // current_price)
            if shares > 0:
                self.balance -= shares * current_price
                self.shares_held += shares
                reward += 0.01
        elif action == 2 and self.shares_held > 0:  # Sell
            self.balance += self.shares_held * current_price
            self.shares_held = 0
            self.hold_duration = 0
        else:  # Hold
            reward -= 0.001

        # Advance step
        self.current_step += 1
        done = self.current_step >= self.max_step
        idx = min(self.current_step, self.max_step)
        next_price = float(self.data.iloc[idx]["Close"])
        self.net_worth = self.balance + self.shares_held * next_price

        # Reward shaping
        reward += (self.net_worth - prev_net_worth) / self.initial_balance
        reward = np.clip(reward, -1.0, 1.0)

        # Lookahead
        lookahead = 5
        future_idx = idx + lookahead
        if future_idx < len(self.data):
            future_price = float(self.data.iloc[future_idx]["Close"])
            future_return = (future_price - next_price) / next_price
            reward += np.clip(future_return, -0.1, 0.1)

        # Holding decay
        if self.shares_held > 0:
            self.hold_duration += 1
            reward -= 0.001 * self.hold_duration
        else:
            self.hold_duration = 0

        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, done, False, {"net_worth": self.net_worth}


# =========================
# Next-Day Signal
# =========================
def next_day_signal(model, ticker="SPY", confidence_threshold=0.60):
    data = yf.download(ticker, period="60d", interval="1d", progress=False).dropna()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data["Close"] = data["Close"].astype(float)
    close = data["Close"]

    # Indicators
    data["SMA_5"] = ta.trend.sma_indicator(close, window=5)
    data["SMA_20"] = ta.trend.sma_indicator(close, window=20)
    data["SMA_50"] = ta.trend.sma_indicator(close, window=50)
    data["RSI"] = ta.momentum.rsi(close, window=14)
    data["MACD"] = ta.trend.macd(close)
    data["ATR"] = ta.volatility.average_true_range(data["High"], data["Low"], data["Close"], window=14)
    data["Close_slope"] = close.diff()
    data["RSI_delta"] = data["RSI"].diff()
    data["Volatility"] = close.rolling(5).std()
    data["Volume_delta"] = data["Volume"].pct_change()

    data = data.dropna(subset=["SMA_20", "SMA_50"]).reset_index(drop=True)
    row = data.iloc[-1]

    obs = np.array([
        (row["Close"] / row["SMA_20"] - 1.0) / 0.05,
        (row["SMA_5"] / row["SMA_20"] - 1.0) / 0.05,
        (row["SMA_20"] / row["SMA_50"] - 1.0) / 0.05,
        (row["RSI"] - 50) / 50.0,
        row["MACD"] / 5.0,
        (row["ATR"] / row["Close"]) / 0.01,
        (row["Close_slope"] / row["Close"]) / 0.01,
        (row["RSI_delta"] / 100.0) / 0.1,
        (row["Volatility"] / row["Close"]) / 0.01,
        row["Volume_delta"] / 0.05
    ], dtype=np.float32)

    obs = np.clip(obs, -5.0, 5.0)
    obs_tensor = model.policy.obs_to_tensor(obs)[0]
    dist = model.policy.get_distribution(obs_tensor)
    probs = dist.distribution.probs.detach().cpu().numpy()[0]

    labels = ["Hold", "Buy", "Sell"]
    if probs.max() >= confidence_threshold:
        action = int(np.argmax(probs))
    else:
        action = 0  # default Hold

    print("\n📊 Next-day trading signal:")
    print(f"➡  {labels[action]}")
    print("\n🔎 Confidence weights:")
    for l, p in zip(labels, probs):
        print(f"{l}: {p:.2%}")

    return labels[action], dict(zip(labels, probs))


# =========================
# Main
# =========================
if __name__ == "__main__":
    # Create environment
    env = DummyVecEnv([lambda: TradingEnvIndicators("SPY")])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=5.0)

    model_path = "ppo_spy_model"

    if os.path.exists(model_path + ".zip"):
        print("✅ Loading existing PPO model...")
        model = PPO.load(model_path, env=env)
    else:
        print("🤖 Training new PPO model...")

        model = PPO(
            "MlpPolicy",
            env,
            ent_coef=0.05,
            gamma=0.98,
            learning_rate=3e-4,
            policy_kwargs=dict(net_arch=[128, 128], activation_fn=nn.ReLU),
            verbose=1,
        )

        model.learn(total_timesteps=1_000_000)
        model.save(model_path)
        print("✅ Model trained and saved.")

    # Run next-day signal for a ticker
    next_day_signal(model, "SLDP", confidence_threshold=0.60)
