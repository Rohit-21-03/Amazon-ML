import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "train"
    / "train_ground_truth.tsv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train"
    / "train_ground_truth_links.parquet"
)

CHUNK_SIZE = 250_000
MATCH_SEPARATOR = re.compile(r"[,;|]")


def detect_columns(columns):
    possible_source_columns = [
        "source1_entity_id",
        "entity_id",
    ]

    possible_match_columns = [
        "matched_entity_ids",
        "match_entity_ids",
        "matched_ids",
    ]

    source_column = next(
        (
            column
            for column in possible_source_columns
            if column in columns
        ),
        None,
    )

    match_column = next(
        (
            column
            for column in possible_match_columns
            if column in columns
        ),
        None,
    )

    if source_column is None:
        raise ValueError(
            "Could not find the Source 1 entity-ID column. "
            f"Available columns: {list(columns)}"
        )

    if match_column is None:
        raise ValueError(
            "Could not find the matched-ID column. "
            f"Available columns: {list(columns)}"
        )

    return source_column, match_column


def split_matches(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return [
        item.strip()
        for item in MATCH_SEPARATOR.split(value)
        if item.strip()
    ]


def convert_ground_truth():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found: {INPUT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    reader = pd.read_csv(
        INPUT_PATH,
        sep="\t",
        dtype="string",
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    writer = None
    total_links = 0

    try:
        for chunk in tqdm(
            reader,
            desc="Preparing ground-truth links",
            unit="chunk",
        ):
            source_column, match_column = detect_columns(
                chunk.columns
            )

            working = chunk[
                [source_column, match_column]
            ].copy()

            working["candidate_entity_id"] = (
                working[match_column]
                .map(split_matches)
            )

            working = working.explode(
                "candidate_entity_id"
            )

            working = working[
                working["candidate_entity_id"]
                .notna()
            ].copy()

            working["candidate_entity_id"] = (
                working["candidate_entity_id"]
                .astype(str)
                .str.strip()
            )

            working = working[
                working["candidate_entity_id"] != ""
            ]

            links = working.rename(
                columns={
                    source_column:
                        "source1_entity_id"
                }
            )[
                [
                    "source1_entity_id",
                    "candidate_entity_id",
                ]
            ]

            links["source1_entity_id"] = (
                links["source1_entity_id"]
                .astype(str)
                .str.strip()
            )

            links["candidate_source"] = (
                links["candidate_entity_id"]
                .str.extract(
                    r"^(S2|S3)",
                    expand=False,
                )
                .fillna("unknown")
            )

            links = links.drop_duplicates()

            if links.empty:
                continue

            total_links += len(links)

            table = pa.Table.from_pandas(
                links,
                preserve_index=False,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    OUTPUT_PATH,
                    table.schema,
                    compression="zstd",
                )

            writer.write_table(table)

    finally:
        if writer is not None:
            writer.close()

    if total_links == 0:
        raise ValueError(
            "No ground-truth links were produced."
        )

    print()
    print(
        f"Ground-truth links created: "
        f"{OUTPUT_PATH}"
    )
    print(f"Total links: {total_links:,}")


if __name__ == "__main__":
    convert_ground_truth()