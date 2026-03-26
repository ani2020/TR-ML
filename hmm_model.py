import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


class HMMModel:
    def __init__(self, n_components=3, covariance_type="full", n_iter=100):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter

        self.model = None
        self.scaler = StandardScaler()

    def _prepare_features(self, df):
        """
        Select features for HMM (controlled set)
        """

        feature_cols = [
            "return_1",
            "volatility_10",
            "momentum_10",
            "zscore"
        ]

        # fallback if missing
        feature_cols = [col for col in feature_cols if col in df.columns]

        if len(feature_cols) == 0:
            raise ValueError("No valid HMM features found")

        feature_df = df[feature_cols].copy()

        # drop NaNs
        feature_df = feature_df.dropna()

        if feature_df.empty:
            raise ValueError("No valid data after removing NaNs for HMM")

        features_scaled = self.scaler.fit_transform(feature_df.values)

        return features_scaled

    def fit(self, df):
        X = self._prepare_features(df)

        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=42
        )

        self.model.fit(X)

    def predict(self, df):
        df = df.copy()

        feature_cols = [
            "return_1",
            "volatility_10",
            "momentum_10",
            "zscore"
        ]

        feature_cols = [col for col in feature_cols if col in df.columns]

        feature_df = df[feature_cols].copy()

        valid_idx = feature_df.dropna().index

        if len(valid_idx) == 0:
            df["state"] = 0
            return df

        X = self.scaler.transform(feature_df.loc[valid_idx].values)

        states = self.model.predict(X)
        probs = self.model.predict_proba(X)

        df["state"] = np.nan

        for i in range(self.n_components):
            df[f"state_prob_{i}"] = np.nan

        df.loc[valid_idx, "state"] = states

        for i in range(self.n_components):
            df.loc[valid_idx, f"state_prob_{i}"] = probs[:, i]

        return df

    def label_states(self, df):
        """
        Assign meaning to states based on stats
        """
        state_summary = df.groupby("state").agg({
            "return_1": "mean",
            "volatility_10": "mean"
        })

        mapping = {}

        for state, row in state_summary.iterrows():
            if row["return_1"] > 0 and row["volatility_10"] < state_summary["volatility_10"].mean():
                mapping[state] = "bull"
            elif row["return_1"] < 0 and row["volatility_10"] > state_summary["volatility_10"].mean():
                mapping[state] = "bear"
            else:
                mapping[state] = "sideways"

        df["regime"] = df["state"].map(mapping)

        return df, mapping
    
    def derive_state_mapping(self, df):
        """
        Derive regime labels using TRAIN data only
        """
        state_summary = df.groupby("state").agg({
            "return_1": "mean",
            "volatility_10": "mean"
        })

        mapping = {}

        vol_mean = state_summary["volatility_10"].mean()

        for state, row in state_summary.iterrows():
            if row["return_1"] > 0 and row["volatility_10"] < vol_mean:
                mapping[state] = "bull"
            elif row["return_1"] < 0 and row["volatility_10"] > vol_mean:
                mapping[state] = "bear"
            else:
                mapping[state] = "sideways"

        return mapping
    
    def apply_state_mapping(self, df, mapping):
        df = df.copy()
        df["regime"] = df["state"].map(mapping)

        # fallback if unseen state appears
        df["regime"] = df["regime"].fillna("sideways")

        return df