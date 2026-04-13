import pandas as pd
from datetime import datetime
import itertools
import uuid
from logger import log_results


class GridSearchOptimizer:

    def __init__(self, param_grid):
        self.param_grid = param_grid

    def run(self, df, wf, pipeline_fn, file_path):

        keys = self.param_grid.keys()
        values = self.param_grid.values()
        df_fd = pd.DataFrame()

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))

            result = wf.run(df, pipeline_fn, params)

            for r in result:
                run_id = str(uuid.uuid4())[:8]
                row = {
                    "run_id": run_id,
                    **params,
                    **r["metrics"],
                    **r["feature_importance"],
                    "score": r["score"]
                }
                full_data = r["full_data"]
                full_data["run_id"] = run_id
                df_fd = pd.concat([df_fd, full_data], ignore_index=True)

                log_results(row, file_path)
        
        time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        full_data_file = f"results/full_data_{time_stamp}.csv"
        #df_fd.to_csv(full_data_file, index=False) - # if raw data is needed

            