import pandas as pd
from loader_fut import load_data
from preprocessing_fut import add_returns
from feature_engineering_fut import fut_add_features
from feature_engineering_spot import add_features
from feature_engineering_fut import add_basic_features

# Step 1: Load data
df = load_data(symbol="^NSEI", start="2015-01-01")

df['date'] = pd.to_datetime(df['date'])
df['fut_expiry'] = pd.to_datetime(df['fut_expiry'])
df = df.set_index('date')
df = df.sort_index()
#print(f"Index is : {df.index} number of records: {len(df)}")

# Step 2: Add returns
#df = add_returns(df)
#print(f"Index is : {df.index} number of records: {len(df)}")


# Step 3: Add features
df = fut_add_features(df)
#print(f"Index is : {df.index} number of records: {len(df)}")
df = add_features(df)
#print(f"Index is : {df.index} number of records: {len(df)}")
#df = add_basic_features(df)

# Step 4: Save
df.to_csv("data/processed/NIFTY_full_data_prec_f.csv")

print("Data saved to data/processed/NIFTY_full_data_prec.csv")
#print(df.head())