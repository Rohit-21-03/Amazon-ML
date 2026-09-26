import pandas as pd


def analyze_country(df, source_name):

    result = (
        df["country"]
        .value_counts(dropna=False)
        .reset_index()
    )

    result.columns = ["country", "count"]

    result["source"] = source_name

    return result