import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from src.preprocessing.text_normalizer import (
    normalize_business_address,
    normalize_business_name,
    normalize_business_name_core,
    normalize_country,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


EXPECTED_COLUMNS = {
    "entity_id",
    "business_name",
    "business_address",
    "country",
}


def validate_columns(columns, input_path):
    missing_columns = EXPECTED_COLUMNS.difference(columns)

    if missing_columns:
        raise ValueError(
            f"{input_path} is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def normalize_chunk(chunk):
    chunk = chunk.copy()

    chunk["entity_id"] = (
        chunk["entity_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    chunk["business_name"] = (
        chunk["business_name"]
        .fillna("")
        .astype(str)
    )

    chunk["business_address"] = (
        chunk["business_address"]
        .fillna("")
        .astype(str)
    )

    chunk["country"] = (
        chunk["country"]
        .fillna("")
        .astype(str)
    )

    chunk["name_normalized"] = (
        chunk["business_name"]
        .map(normalize_business_name)
    )

    chunk["name_core"] = (
        chunk["business_name"]
        .map(normalize_business_name_core)
    )

    chunk["address_normalized"] = (
        chunk["business_address"]
        .map(normalize_business_address)
    )

    chunk["country_normalized"] = (
        chunk["country"]
        .map(normalize_country)
    )

    chunk["name_missing"] = (
        chunk["name_normalized"]
        .eq("")
        .astype("int8")
    )

    chunk["address_missing"] = (
        chunk["address_normalized"]
        .eq("")
        .astype("int8")
    )

    chunk["name_length"] = (
        chunk["name_normalized"]
        .str.len()
        .astype("int32")
    )

    chunk["address_length"] = (
        chunk["address_normalized"]
        .str.len()
        .astype("int32")
    )

    return chunk


def preprocess_file(
    input_path,
    output_path,
    chunk_size=250_000,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        output_path.unlink()

    reader = pd.read_csv(
        input_path,
        sep="\t",
        dtype={
            "entity_id": "string",
            "business_name": "string",
            "business_address": "string",
            "country": "string",
        },
        chunksize=chunk_size,
        keep_default_na=True,
        low_memory=False,
    )

    parquet_writer = None
    total_rows = 0

    try:
        for chunk in tqdm(
            reader,
            desc=f"Processing {input_path.name}",
            unit="chunk",
        ):
            validate_columns(
                chunk.columns,
                input_path,
            )

            normalized_chunk = normalize_chunk(chunk)

            total_rows += len(normalized_chunk)

            table = pa.Table.from_pandas(
                normalized_chunk,
                preserve_index=False,
            )

            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(
                    where=str(output_path),
                    schema=table.schema,
                    compression="snappy",
                )

            parquet_writer.write_table(table)

    except Exception:
        if parquet_writer is not None:
            parquet_writer.close()

        if output_path.exists():
            output_path.unlink()

        raise

    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    if total_rows == 0:
        raise ValueError(
            f"No rows were read from {input_path}"
        )

    print(
        f"Completed: {input_path.name} -> "
        f"{output_path.name} "
        f"({total_rows:,} rows)"
    )

    return total_rows


def get_source_jobs():
    return [
        (
            RAW_DATA_DIR
            / "train"
            / "train_source1.tsv",
            PROCESSED_DATA_DIR
            / "train"
            / "train_source1_normalized.parquet",
            "train",
        ),
        (
            RAW_DATA_DIR
            / "train"
            / "train_source2.tsv",
            PROCESSED_DATA_DIR
            / "train"
            / "train_source2_normalized.parquet",
            "train",
        ),
        (
            RAW_DATA_DIR
            / "train"
            / "train_source3.tsv",
            PROCESSED_DATA_DIR
            / "train"
            / "train_source3_normalized.parquet",
            "train",
        ),
        (
            RAW_DATA_DIR
            / "test"
            / "test_source1.tsv",
            PROCESSED_DATA_DIR
            / "test"
            / "test_source1_normalized.parquet",
            "test",
        ),
        (
            RAW_DATA_DIR
            / "test"
            / "test_source2.tsv",
            PROCESSED_DATA_DIR
            / "test"
            / "test_source2_normalized.parquet",
            "test",
        ),
        (
            RAW_DATA_DIR
            / "test"
            / "test_source3.tsv",
            PROCESSED_DATA_DIR
            / "test"
            / "test_source3_normalized.parquet",
            "test",
        ),
    ]


def verify_input_files(jobs):
    missing_files = []

    for input_path, _, _ in jobs:
        if not input_path.exists():
            missing_files.append(str(input_path))

    if missing_files:
        missing_text = "\n".join(missing_files)

        raise FileNotFoundError(
            "The following input files were not found:\n"
            f"{missing_text}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Normalize Amazon ML Challenge "
            "train and test source files."
        )
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250_000,
        help="Number of source rows processed per chunk.",
    )

    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="all",
        help="Dataset split to process.",
    )

    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise ValueError(
            "--chunk-size must be greater than zero."
        )

    jobs = get_source_jobs()

    if args.split != "all":
        jobs = [
            job
            for job in jobs
            if job[2] == args.split
        ]

    verify_input_files(jobs)

    row_counts = {}

    for input_path, output_path, split in jobs:
        print()
        print("=" * 70)
        print(f"Split: {split}")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print("=" * 70)

        total_rows = preprocess_file(
            input_path=input_path,
            output_path=output_path,
            chunk_size=args.chunk_size,
        )

        row_counts[str(output_path)] = total_rows

    print()
    print("Phase 1 preprocessing completed successfully.")

    for output_path, row_count in row_counts.items():
        print(f"{output_path}: {row_count:,} rows")


if __name__ == "__main__":
    main()