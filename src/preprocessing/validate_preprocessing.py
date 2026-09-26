from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


EXPECTED_FILES = [
    Path("train")
    / "train_source1_normalized.parquet",

    Path("train")
    / "train_source2_normalized.parquet",

    Path("train")
    / "train_source3_normalized.parquet",

    Path("test")
    / "test_source1_normalized.parquet",

    Path("test")
    / "test_source2_normalized.parquet",

    Path("test")
    / "test_source3_normalized.parquet",
]


EXPECTED_COLUMNS = {
    "entity_id",
    "business_name",
    "business_address",
    "country",
    "name_normalized",
    "name_core",
    "address_normalized",
    "country_normalized",
    "name_missing",
    "address_missing",
    "name_length",
    "address_length",
}


def validate_schema(path):
    parquet_file = pq.ParquetFile(path)
    available_columns = set(parquet_file.schema.names)

    missing_columns = (
        EXPECTED_COLUMNS - available_columns
    )

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return parquet_file.metadata.num_rows


def validate_file(path, chunk_size=250_000):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing processed file: {path}"
        )

    expected_rows = validate_schema(path)

    columns_to_read = [
        "entity_id",
        "name_normalized",
        "address_normalized",
        "country_normalized",
    ]

    parquet_file = pq.ParquetFile(path)

    total_rows = 0
    empty_entity_ids = 0
    missing_entity_ids = 0
    empty_normalized_names = 0
    empty_normalized_addresses = 0
    country_counts = {}

    seen_entity_ids = set()
    duplicate_entity_ids = 0

    for batch in parquet_file.iter_batches(
        batch_size=chunk_size,
        columns=columns_to_read,
    ):
        chunk = batch.to_pandas()

        total_rows += len(chunk)

        entity_ids = (
            chunk["entity_id"]
            .fillna("")
            .astype(str)
        )

        missing_entity_ids += int(
            chunk["entity_id"].isna().sum()
        )

        empty_entity_ids += int(
            entity_ids.eq("").sum()
        )

        empty_normalized_names += int(
            chunk["name_normalized"]
            .fillna("")
            .eq("")
            .sum()
        )

        empty_normalized_addresses += int(
            chunk["address_normalized"]
            .fillna("")
            .eq("")
            .sum()
        )

        chunk_country_counts = (
            chunk["country_normalized"]
            .fillna("")
            .value_counts()
            .to_dict()
        )

        for country, count in chunk_country_counts.items():
            country_counts[country] = (
                country_counts.get(country, 0)
                + int(count)
            )

        for entity_id in entity_ids:
            if not entity_id:
                continue

            if entity_id in seen_entity_ids:
                duplicate_entity_ids += 1
            else:
                seen_entity_ids.add(entity_id)

    if total_rows != expected_rows:
        raise ValueError(
            f"Row-count mismatch in {path.name}. "
            f"Metadata: {expected_rows:,}, "
            f"read: {total_rows:,}"
        )

    return {
        "file": path.name,
        "rows": total_rows,
        "duplicate_entity_ids": duplicate_entity_ids,
        "empty_entity_ids": empty_entity_ids,
        "missing_entity_ids": missing_entity_ids,
        "empty_normalized_names": (
            empty_normalized_names
        ),
        "empty_normalized_addresses": (
            empty_normalized_addresses
        ),
        "countries": str(country_counts),
    }


def main():
    results = []

    for relative_path in EXPECTED_FILES:
        path = (
            PROCESSED_DATA_DIR
            / relative_path
        )

        print()
        print("=" * 70)
        print(f"Validating: {relative_path}")
        print("=" * 70)

        result = validate_file(path)
        result["split"] = relative_path.parent.name

        results.append(result)

        for key, value in result.items():
            print(f"{key}: {value}")

        if result["duplicate_entity_ids"] > 0:
            raise ValueError(
                f"{relative_path} contains duplicate "
                "entity IDs."
            )

        if result["empty_entity_ids"] > 0:
            raise ValueError(
                f"{relative_path} contains empty "
                "entity IDs."
            )

        if result["missing_entity_ids"] > 0:
            raise ValueError(
                f"{relative_path} contains missing "
                "entity IDs."
            )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        REPORTS_DIR
        / "phase1_validation.csv"
    )

    report_df = pd.DataFrame(results)

    report_columns = [
        "split",
        "file",
        "rows",
        "duplicate_entity_ids",
        "empty_entity_ids",
        "missing_entity_ids",
        "empty_normalized_names",
        "empty_normalized_addresses",
        "countries",
    ]

    report_df = report_df[report_columns]
    report_df.to_csv(report_path, index=False)

    print()
    print("=" * 70)
    print("Phase 1 validation completed successfully.")
    print(f"Report saved to: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()