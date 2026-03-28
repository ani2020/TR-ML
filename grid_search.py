import itertools
import uuid
from logger import log_results


class GridSearchOptimizer:

    def __init__(self, param_grid):
        self.param_grid = param_grid

    def run(self, df, wf, pipeline_fn, file_path):

        keys = self.param_grid.keys()
        values = self.param_grid.values()

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))

            result = wf.run(df, pipeline_fn, params)

            for r in result:
                row = {
                    "run_id": str(uuid.uuid4())[:8],
                    **params,
                    **r["metrics"],
                    "feature_importance": r["feature_importance"],
                    "score": r["score"]
                }

                log_results(row, file_path)