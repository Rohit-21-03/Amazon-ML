import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "train" / "train_pair_features.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
SCORED_PATH = PROJECT_ROOT / "data" / "processed" / "train" / "validation_scored_pairs.parquet"
MODEL_PATH = MODEL_DIR / "lightgbm_baseline.txt"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"
METRICS_PATH = REPORT_DIR / "phase4A_pair_metrics.json"
IMPORTANCE_PATH = REPORT_DIR / "phase4A_feature_importance.csv"

FEATURE_COLUMNS = [
    "candidate_rank", "candidate_is_s2", "exact_name", "exact_core",
    "exact_address", "shared_token_count", "shared_address_tokens",
    "token_blocking_score", "address_blocking_score", "exact_blocking_score",
    "combined_blocking_score", "name_fuzzy_similarity", "country_equal",
    "s1_name_missing", "target_name_missing", "s1_address_missing",
    "target_address_missing", "name_jaro_winkler", "core_jaro_winkler",
    "address_jaro_winkler", "name_levenshtein_distance",
    "core_levenshtein_distance", "address_levenshtein_distance",
    "name_levenshtein_similarity", "core_levenshtein_similarity",
    "address_levenshtein_similarity", "name_length_difference",
    "core_length_difference", "address_length_difference",
    "core_contains", "address_contains",
]
ID_COLUMNS = ["source1_entity_id", "candidate_entity_id", "candidate_source"]


def validate_input():
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Missing feature file: {FEATURE_PATH}")
    schema_names = set(pq.read_schema(FEATURE_PATH).names)
    required = set(FEATURE_COLUMNS + ID_COLUMNS + ["label", "dataset_split"])
    missing = sorted(required - schema_names)
    if missing:
        raise ValueError(f"Feature file is missing columns: {missing}")


def read_split(split):
    dataset = ds.dataset(FEATURE_PATH, format="parquet")
    columns = ID_COLUMNS + FEATURE_COLUMNS + ["label"]
    table = dataset.to_table(
        columns=columns,
        filter=ds.field("dataset_split") == split,
    )
    return table


def feature_matrix(table):
    arrays = []
    for column in FEATURE_COLUMNS:
        values = table[column].to_numpy(zero_copy_only=False)
        arrays.append(np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
    return np.column_stack(arrays)


def binary_metrics(labels, probabilities, threshold=0.5):
    predictions = probabilities >= threshold
    labels = labels.astype(bool)
    tp = int(np.sum(predictions & labels))
    fp = int(np.sum(predictions & ~labels))
    fn = int(np.sum(~predictions & labels))
    tn = int(np.sum(~predictions & ~labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    beta2 = 0.25
    f05 = ((1 + beta2) * precision * recall / (beta2 * precision + recall)
           if precision + recall else 0.0)
    return {
        "threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f0_5_pairwise": f05,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Phase 4A LightGBM baseline.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--num-boost-round", type=int, default=1200)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    args = parser.parse_args()

    validate_input()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SCORED_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading training split...")
    train_table = read_split("train")
    print("Loading validation split...")
    valid_table = read_split("validation")

    print("Creating numeric matrices...")
    x_train = feature_matrix(train_table)
    y_train = train_table["label"].to_numpy(zero_copy_only=False).astype(np.int8)
    x_valid = feature_matrix(valid_table)
    y_valid = valid_table["label"].to_numpy(zero_copy_only=False).astype(np.int8)

    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)
    scale_pos_weight = negative / positive if positive else 1.0

    train_data = lgb.Dataset(x_train, label=y_train, feature_name=FEATURE_COLUMNS, free_raw_data=True)
    valid_data = lgb.Dataset(x_valid, label=y_valid, reference=train_data,
                             feature_name=FEATURE_COLUMNS, free_raw_data=True)

    params = {
        "objective": "binary", "metric": ["binary_logloss", "auc"],
        "learning_rate": 0.04, "num_leaves": 63, "max_depth": -1,
        "min_data_in_leaf": 100, "feature_fraction": 0.9,
        "bagging_fraction": 0.9, "bagging_freq": 1,
        "lambda_l1": 0.1, "lambda_l2": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "verbosity": -1, "num_threads": args.threads,
        "seed": 42, "feature_fraction_seed": 42, "bagging_seed": 42,
    }

    print(f"Training rows: {len(y_train):,}")
    print(f"Validation rows: {len(y_valid):,}")
    print(f"Training positives: {positive:,}")
    print(f"Training negatives: {negative:,}")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    evaluation = {}
    model = lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=args.num_boost_round,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "validation"],
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=25),
            lgb.record_evaluation(evaluation),
        ],
    )

    model.save_model(str(MODEL_PATH), num_iteration=model.best_iteration)
    FEATURES_PATH.write_text(json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8")

    probabilities = model.predict(x_valid, num_iteration=model.best_iteration)
    metrics = binary_metrics(y_valid, probabilities, threshold=0.5)
    metrics.update({
        "best_iteration": int(model.best_iteration),
        "training_rows": int(len(y_train)),
        "validation_rows": int(len(y_valid)),
        "scale_pos_weight": float(scale_pos_weight),
        "note": "Pair-level threshold 0.5 only; entity-level F0.5 tuning occurs in Phase 4B.",
    })
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    importance = {
        "feature": FEATURE_COLUMNS,
        "gain_importance": model.feature_importance(importance_type="gain").tolist(),
        "split_importance": model.feature_importance(importance_type="split").tolist(),
    }
    import pandas as pd
    importance_df = pd.DataFrame(importance).sort_values("gain_importance", ascending=False)
    importance_df.to_csv(IMPORTANCE_PATH, index=False)

    scored = {
        "source1_entity_id": valid_table["source1_entity_id"],
        "candidate_entity_id": valid_table["candidate_entity_id"],
        "candidate_source": valid_table["candidate_source"],
        "label": valid_table["label"],
        "match_probability": probabilities.astype(np.float32),
    }
    import pyarrow as pa
    pq.write_table(pa.table(scored), SCORED_PATH, compression="zstd")

    print("\nPhase 4A complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Model: {MODEL_PATH}")
    print(f"Features: {FEATURES_PATH}")
    print(f"Validation scores: {SCORED_PATH}")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Importance: {IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
