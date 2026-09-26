import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"
MODEL_DIR = PROJECT_ROOT / "models"

FEATURE_PATH = TEST_DIR / "test_pair_features.parquet"
MODEL_PATH = MODEL_DIR / "lightgbm_baseline.txt"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
THRESHOLD_PATH = MODEL_DIR / "best_threshold.json"
OUTPUT_PATH = TEST_DIR / "test_scored_pairs.parquet"

ID_COLUMNS = ["source1_entity_id", "candidate_entity_id", "candidate_source"]


def validate_inputs():
    required = (FEATURE_PATH, MODEL_PATH, FEATURE_COLUMNS_PATH, THRESHOLD_PATH)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def load_threshold():
    payload = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    if "threshold" not in payload:
        raise ValueError("best_threshold.json does not contain 'threshold'.")
    return float(payload["threshold"])


def table_to_matrix(table, feature_columns):
    arrays = []
    for column in feature_columns:
        values = table[column].to_numpy(zero_copy_only=False)
        values = np.asarray(values, dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        arrays.append(values)
    return np.column_stack(arrays)


def score(batch_size, threads):
    feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))
    schema_names = set(pq.read_schema(FEATURE_PATH).names)
    missing_features = sorted(set(feature_columns + ID_COLUMNS) - schema_names)
    if missing_features:
        raise ValueError(f"Test feature file is missing columns: {missing_features}")

    model = lgb.Booster(model_file=str(MODEL_PATH))
    if model.feature_name() != feature_columns:
        raise ValueError(
            "Model feature order does not match feature_columns.json.\n"
            f"Model: {model.feature_name()}\nJSON: {feature_columns}"
        )

    threshold = load_threshold()
    parquet_file = pq.ParquetFile(FEATURE_PATH)
    input_rows = parquet_file.metadata.num_rows

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    writer = None
    processed_rows = 0
    predicted_links = 0
    probability_sum = 0.0
    probability_min = 1.0
    probability_max = 0.0

    columns = ID_COLUMNS + feature_columns
    print(f"Loaded model with {model.num_feature()} features.")
    print(f"Best threshold: {threshold:.6f}")
    print(f"Input rows: {input_rows:,}")
    print("Scoring test pairs in batches...")

    try:
        for batch_number, batch in enumerate(
            parquet_file.iter_batches(batch_size=batch_size, columns=columns), start=1
        ):
            table = pa.Table.from_batches([batch])
            matrix = table_to_matrix(table, feature_columns)
            probabilities = model.predict(
                matrix,
                num_iteration=model.best_iteration,
                num_threads=threads,
            ).astype(np.float32)
            accepted = (probabilities >= threshold).astype(np.int8)

            output_table = pa.table({
                "source1_entity_id": table["source1_entity_id"],
                "candidate_entity_id": table["candidate_entity_id"],
                "candidate_source": table["candidate_source"],
                "match_probability": pa.array(probabilities),
                "predicted_match": pa.array(accepted),
            })

            if writer is None:
                writer = pq.ParquetWriter(
                    OUTPUT_PATH,
                    output_table.schema,
                    compression="zstd",
                )
            writer.write_table(output_table, row_group_size=100_000)

            batch_rows = len(probabilities)
            processed_rows += batch_rows
            predicted_links += int(accepted.sum())
            probability_sum += float(probabilities.sum(dtype=np.float64))
            probability_min = min(probability_min, float(probabilities.min()))
            probability_max = max(probability_max, float(probabilities.max()))

            if batch_number % 10 == 0 or processed_rows == input_rows:
                print(
                    f"Batches: {batch_number:,} | "
                    f"Rows: {processed_rows:,}/{input_rows:,} | "
                    f"Accepted: {predicted_links:,}"
                )
    finally:
        if writer is not None:
            writer.close()

    if processed_rows != input_rows:
        raise ValueError(
            f"Scored row count mismatch: input={input_rows:,}, output={processed_rows:,}"
        )
    if processed_rows == 0:
        raise ValueError("No test feature rows were scored.")

    output_rows = pq.ParquetFile(OUTPUT_PATH).metadata.num_rows
    if output_rows != input_rows:
        raise ValueError(
            f"Output row count mismatch: input={input_rows:,}, output={output_rows:,}"
        )

    print("\nPhase 5F complete.")
    print(f"Scored rows: {processed_rows:,}")
    print(f"Predicted links at threshold: {predicted_links:,}")
    print(f"Predicted-link rate: {predicted_links / processed_rows:.6f}")
    print(f"Mean probability: {probability_sum / processed_rows:.6f}")
    print(f"Minimum probability: {probability_min:.6f}")
    print(f"Maximum probability: {probability_max:.6f}")
    print(f"Created: {OUTPUT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Score Phase 5E test pair features.")
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")

    validate_inputs()
    score(args.batch_size, args.threads)


if __name__ == "__main__":
    main()
