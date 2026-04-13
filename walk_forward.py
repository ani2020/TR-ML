import pandas as pd
from trades import extract_trades
from metrics import compute_metrics
from scoring import compute_score

class WalkForward:
    def __init__(self, train_size, test_size, step_size):
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size

    def run(self, df, pipeline_fn, params):

        results = []
        full_data = df.iloc[:0, :].copy()
        full_data["runno"] = ""
        full_data["mtype"] = ""
        full_data["position"] = 0

        n = len(df)
        if n < self.train_size:
            raise ValueError("Not enough data for training")
        
        i = 0
        if 'date' in df.columns and df['date'].dtype == 'str':
            df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

        while i < n - self.train_size:

            train_start = i
            train_end = i + self.train_size

            test_start = train_end
            test_end = min(train_end + self.test_size, n)  # <-- FIX

            train = df.iloc[train_start:train_end].copy()
            test = df.iloc[test_start:test_end].copy()

            if len(test) == 0:
                break

            train["runno"] = i
            train["mtype"] = "train"
            test["runno"] = i
            test["mtype"] = "test"

            test, feature_importance = pipeline_fn(train, test, params)

            test["position"] = test["signal"].shift(1).fillna(0)
            train["position"] = 0

            trades = extract_trades(test)
            metrics = compute_metrics(test, trades)
            score = compute_score(metrics)
            full_data = pd.concat([full_data, train, test], ignore_index=True)

            i += self.step_size        

            results.append({
                "metrics": metrics,
                "feature_importance": feature_importance,
                "score": score,
                "full_data": full_data
            })

        return results
    
    def get_latest_signal(self, df, pipeline_fn, params):

        train = df.iloc[-self.train_size:]
        test = df.iloc[-1:]

        test, _ = pipeline_fn(train, test, params)

        return test.iloc[-1]