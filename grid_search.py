import itertools
import pandas as pd
import uuid
from scoring import compute_score
from logger import log_results
from datetime import datetime


class GridSearchOptimizer:
    def __init__(self, param_grid):
        self.param_grid = param_grid

    def generate_param_combinations(self):
        keys = self.param_grid.keys()
        values = self.param_grid.values()

        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))

    def run(self, df, walkforward, pipeline_fn, results_file="results/summary.csv"):

        all_results = []

        for params in self.generate_param_combinations():

            print(f"Running params: {params}")

            try:
                result = walkforward.run(
                    df=df,
                    pipeline_fn=pipeline_fn,
                    params=params
                )

                metrics = result["metrics"]
                score = compute_score(metrics)


                run_id = str(uuid.uuid4())[:8]

                time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                #df_fdata = result["full_data"].copy()
                results_df_file = f"results/full_data_df_{run_id}_{time_stamp}.csv"
                #df_fdata.to_csv(results_df_file)


                log_results(
                    run_id=run_id,
                    params=params,
                    metrics=metrics,
                    score=score,
                    file_path=f"{results_file}"
                )

                row = {
                    "run_id": run_id,
                    "score": score,
                    **metrics,
                    **params
                }

                all_results.append(row)

            except Exception as e:
                print(f"Error with params {params}: {e}")
                continue

        return pd.DataFrame(all_results)