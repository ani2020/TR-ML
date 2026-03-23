import xgboost as xgb
import pandas as pd
import numpy as np


class XGBoostModel:
    def __init__(self, params=None):
        default_params = {
            "max_depth": 5,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42
        }

        if params:
            default_params.update(params)

        self.model = xgb.XGBClassifier(**default_params)

    def prepare_data(self, df):
        df = df.copy()

        # --- Target ---
        df["target"] = (df["returns"].shift(-1) > 0).astype(int)

        df = df.dropna()

        # --- Features ---
        feature_cols = [
            "returns",
            "volatility",
            "momentum",
        ]

        # Add HMM probabilities if available
        prob_cols = [col for col in df.columns if "state_prob" in col]
        feature_cols.extend(prob_cols)

        X = df[feature_cols]
        y = df["target"]

        return X, y, feature_cols

    def fit(self, df):
        X, y, self.feature_cols = self.prepare_data(df)
        self.model.fit(X, y)

    def predict(self, df):
        df = df.copy()

        df = df.dropna()

        X = df[self.feature_cols]

        probs = self.model.predict_proba(X)[:, 1]

        df["xgb_prob"] = probs
        df["xgb_signal"] = (probs > 0.5).astype(int)

        return df