from loader_spot import load_data
from preprocessing_spot import add_returns
from feature_engineering_spot import add_features
from feature_engineering_spot import add_basic_features

# Step 1: Load data
df = load_data(symbol="^NSEI", start="2015-01-01")

# Step 2: Add returns
df = add_returns(df)

# Step 3: Add features
df = add_features(df)
#df = add_basic_features(df)

# Step 4: Save
df.to_csv("data/processed/sample_data_spot.csv", index=False)

print("Data saved to data/processed/sample_data_spot.csv")
print(df.head())