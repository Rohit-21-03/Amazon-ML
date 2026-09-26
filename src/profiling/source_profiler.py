import pandas as pd


def profile_source(df, source_name):
    profile = {
        "source": source_name,
        "rows": len(df),
        "unique_entity_ids": df["entity_id"].nunique(),
        "duplicate_entity_ids": len(df) - df["entity_id"].nunique(),
        "missing_business_name": df["business_name"].isna().sum(),
        "missing_business_address": df["business_address"].isna().sum(),
        "missing_country": df["country"].isna().sum(),
        "duplicate_business_names": df["business_name"].duplicated().sum(),
        "duplicate_addresses": df["business_address"].duplicated().sum(),
        "countries": df["country"].nunique()
    }

    return pd.DataFrame([profile])