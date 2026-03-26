import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, capital=100000, cost=0.0005, slippage=0.0005):
        self.initial_capital = capital
        self.cost = cost
        self.slippage = slippage

    def run(self, df):
        df = df.copy()

        df["position"] = df["signal"].shift(1).fillna(0)

        # strategy returns
        df["strategy_return"] = df["position"] * df["return_1"]

        # costs applied on position change
        df["trade"] = df["position"].diff().abs()
        df["costs"] = df["trade"] * (self.cost + self.slippage)

        df["net_return"] = df["strategy_return"] - df["costs"]

        # equity curve
        df["equity"] = (1 + df["net_return"]).cumprod() * self.initial_capital

        return df