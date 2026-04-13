import xgboost as xgb
import numpy as np


class XGBoostModel:
    def __init__(self, params=None):
        default = {
            "max_depth": 5,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42
        }

        if params:
            default.update(params)

        self.model = xgb.XGBClassifier(**default)

    def prepare_data(self, df):
        df = df.copy()

        df["target"] = (df["f_return_1"].shift(-1) > 0).astype(int)
        df = df.dropna()

        exclude = ["index", "date", "fut_expiry", "target", "signal", "regime", "state", "position", "trade_signal", "mtype", "runno"]

        features = [c for c in df.columns if c not in exclude and df[c].dtype != "object"]
        #print(f"Features in XGB: {features}")

        X = df[features]
        y = df["target"]

        return X, y, features

    def fit(self, df):  
        X, y, self.features = self.prepare_data(df)
        self.model.fit(X, y)

    def predict(self, df):
        df = df.copy()
        df = df.dropna()

        X = df[self.features]
        probs = self.model.predict_proba(X)[:, 1]

        df["xgb_prob"] = probs
        return df

    def feature_importance(self):
        return dict(zip(self.features, self.model.feature_importances_))