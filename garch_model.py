import numpy as np
import pandas as pd
from arch import arch_model


class GARCHModel:
    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.model = None
        self.fitted = None

    def fit(self, returns):
        """
        Fit GARCH model on returns
        """
        returns = returns.dropna() * 100  # scale improves stability

        self.model = arch_model(
            returns,
            vol="Garch",
            p=self.p,
            q=self.q,
            dist="normal"
        )

        self.fitted = self.model.fit(disp="off")

    def forecast(self, horizon=1):
        """
        Forecast future volatility
        """
        forecast = self.fitted.forecast(horizon=horizon)

        # variance → volatility
        vol = np.sqrt(forecast.variance.values[-1, :])

        return vol / 100  # scale back

    def fit_predict(self, df):
        """
        Fit once and generate rolling forecast
        (simple version for now)
        """
        df = df.copy()

        returns = df["return_1"].dropna()

        if len(returns) < 50:
            df["garch_vol"] = np.nan
            return df

        self.fit(returns)

        vol = self.forecast()[0]

        df["garch_vol"] = vol

        return df