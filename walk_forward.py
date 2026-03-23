import pandas as pd
from backtester import Backtester
from trades import extract_trades
from metrics import compute_metrics


class WalkForward:
    def __init__(self, train_size, test_size, step_size):
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size

    def run(self, df, pipeline_fn, params):
        results = []
        equity_curves = []

        start = 0

        while True:
            train_end = start + self.train_size
            test_end = train_end + self.test_size

            if test_end > len(df):
                break

            train_df = df.iloc[start:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()

            # Skip bad windows
            if train_df.isnull().values.any() or test_df.isnull().values.any():
                train_df = train_df.dropna()
                test_df = test_df.dropna()

            if len(train_df) == 0 or len(test_df) == 0:
                start += self.step_size
                continue

            print("Train NaNs:", train_df.isna().sum().sum())
            print("Test NaNs:", test_df.isna().sum().sum())

            # --- Run pipeline ---
            test_df = pipeline_fn(train_df, test_df, params)

            # --- Backtest ---
            bt = Backtester()
            test_result = bt.run(test_df)

            trades = extract_trades(test_result)
            metrics = compute_metrics(test_result, trades)

            results.append(metrics)
            
            #equity_curves.append(test_result[["equity"]])
            
            equity_curves.append(test_result)

            start += self.step_size

            print(f"Train: {train_df['date'].min()} → {train_df['date'].max()}")
            print(f"Test: {test_df['date'].min()} → {test_df['date'].max()}")

        return self._aggregate_results(results, equity_curves)

    def _aggregate_results(self, results, equity_curves):
        df_metrics = pd.DataFrame(results)

        avg_metrics = df_metrics.mean().to_dict()

        # Combine equity curves
        equity = pd.concat(equity_curves).reset_index(drop=True)

        return {
            "metrics": avg_metrics,
            "equity_curve": equity,
            "all_runs": df_metrics,
            "full_data": pd.concat(equity_curves).reset_index(drop=True)
        }