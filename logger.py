import pandas as pd
import os


def log_results(row, file_path):

    df = pd.DataFrame([row])

    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode="a", header=False, index=False)