def validate_dataframe(df, name="df"):
    print(f"\nValidating {name}")
    print("Shape:", df.shape)
    print("NaNs:", df.isna().sum().sum())
    print("Duplicates:", df.duplicated().sum())

    if "date" in df.columns:
        print("Sorted:", df["date"].is_monotonic_increasing)

    print(df.dtypes)
    print("------")