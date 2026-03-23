def validate_dataframe(df, name="df"):
    print(f"\nValidating {name}...")

    print("Shape:", df.shape)

    # Check NaNs
    nan_count = df.isna().sum().sum()
    print("NaNs:", nan_count)

    # Check duplicates
    dup_count = df.duplicated().sum()
    print("Duplicates:", dup_count)

    # Check types
    print(df.dtypes)

    # Check sorted dates
    if "date" in df.columns:
        is_sorted = df["date"].is_monotonic_increasing
        print("Date sorted:", is_sorted)

    print("----")