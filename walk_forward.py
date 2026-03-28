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

        for i in range(0, len(df) - self.train_size - self.test_size, self.step_size):

            train = df.iloc[i:i+self.train_size]
            test = df.iloc[i+self.train_size:i+self.train_size+self.test_size]

            test, feature_importance = pipeline_fn(train, test, params)

            test["position"] = test["signal"].shift(1).fillna(0)

            trades = extract_trades(test)
            metrics = compute_metrics(test, trades)
            score = compute_score(metrics)

            results.append({
                "metrics": metrics,
                "feature_importance": feature_importance,
                "score": score
            })

        return results