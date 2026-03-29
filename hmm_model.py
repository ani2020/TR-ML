import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

class HMMModel:
    def __init__(self, n_components=2, covariance_type="diag"):
        self.n_components = n_components
        self.model = GaussianHMM(n_components=n_components, covariance_type=covariance_type)
        self.scaler = StandardScaler()

    def _prepare_features(self, df):
        cols = ["return_1", "volatility_10", "momentum_5", "zscore", "garch_vol"]
        cols = [c for c in cols if c in df.columns]

        X = df[cols].dropna()
        X_scaled = self.scaler.fit_transform(X.values)

        return X_scaled, X.index

    def fit(self, df):
        X, _ = self._prepare_features(df)
        self.model.fit(X)

    def predict(self, df):
        df = df.copy()
        X, idx = self._prepare_features(df)

        states = self.model.predict(X)
        probs = self.model.predict_proba(X)

        df["state"] = np.nan
        df.loc[idx, "state"] = states

        for i in range(self.n_components):
            df[f"state_prob_{i}"] = np.nan
            df.loc[idx, f"state_prob_{i}"] = probs[:, i]

        return df

    def derive_state_mapping(self, df):
        mapping = {}

        for state in range(self.n_components):
            avg_return = df[df["state"] == state]["return_1"].mean()

            if avg_return > 0:
                mapping[state] = "bull"
            else:
                mapping[state] = "bear"

        return mapping

    def apply_state_mapping(self, df, mapping):
        df["regime"] = df["state"].map(mapping)
        return df