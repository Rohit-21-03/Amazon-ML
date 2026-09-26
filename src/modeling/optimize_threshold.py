import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "train"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

SCORED_PAIRS = TRAIN_DIR / "validation_scored_pairs.parquet"
GROUND_TRUTH = TRAIN_DIR / "train_ground_truth_links.parquet"
SOURCE1 = TRAIN_DIR / "train_source1_normalized.parquet"
SEARCH_REPORT = REPORTS_DIR / "phase4B_threshold_search.csv"
METRICS_REPORT = REPORTS_DIR / "phase4B_validation_metrics.json"
THRESHOLD_OUTPUT = MODELS_DIR / "best_threshold.json"


def sql_path(path):
    return Path(path).resolve().as_posix().replace("'", "''")


def validate_inputs():
    required = (SCORED_PAIRS, GROUND_TRUTH, SOURCE1)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def create_connection(memory_limit, threads, temp_limit):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = OUTPUTS_DIR / "duckdb_temp_phase4b"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(OUTPUTS_DIR / "phase4b_thresholds.duckdb"))
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET threads = {threads}")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET temp_directory = '{sql_path(temp_dir)}'")
    con.execute(f"SET max_temp_directory_size = '{temp_limit}'")
    return con


def prepare_tables(con):
    scored = sql_path(SCORED_PAIRS)
    gt = sql_path(GROUND_TRUTH)
    source1 = sql_path(SOURCE1)

    print("Preparing validation entities and ground truth...")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE validation_entities AS
        SELECT CAST(entity_id AS VARCHAR) AS source1_entity_id
        FROM read_parquet('{source1}')
        WHERE hash(CAST(entity_id AS VARCHAR)) % 10 >= 8
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE validation_gt AS
        SELECT DISTINCT
            CAST(g.source1_entity_id AS VARCHAR) AS source1_entity_id,
            CAST(g.candidate_entity_id AS VARCHAR) AS candidate_entity_id
        FROM read_parquet('{gt}') g
        INNER JOIN validation_entities v
          ON CAST(g.source1_entity_id AS VARCHAR) = v.source1_entity_id
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE validation_scores AS
        SELECT
            CAST(source1_entity_id AS VARCHAR) AS source1_entity_id,
            CAST(candidate_entity_id AS VARCHAR) AS candidate_entity_id,
            CAST(match_probability AS DOUBLE) AS match_probability
        FROM read_parquet('{scored}')
    """)

    counts = con.execute("""
        SELECT
            (SELECT COUNT(*) FROM validation_entities) AS validation_entities,
            (SELECT COUNT(*) FROM validation_gt) AS validation_true_links,
            (SELECT COUNT(*) FROM validation_scores) AS scored_pairs
    """).fetchdf()
    print(counts.to_string(index=False))


def evaluate_threshold(con, threshold):
    row = con.execute(f"""
        WITH predictions AS (
            SELECT source1_entity_id, candidate_entity_id
            FROM validation_scores
            WHERE match_probability >= {float(threshold)}
        ),
        true_counts AS (
            SELECT source1_entity_id, COUNT(*) AS true_count
            FROM validation_gt
            GROUP BY source1_entity_id
        ),
        pred_counts AS (
            SELECT source1_entity_id, COUNT(*) AS pred_count
            FROM predictions
            GROUP BY source1_entity_id
        ),
        tp_counts AS (
            SELECT p.source1_entity_id, COUNT(*) AS tp
            FROM predictions p
            INNER JOIN validation_gt g
              ON p.source1_entity_id = g.source1_entity_id
             AND p.candidate_entity_id = g.candidate_entity_id
            GROUP BY p.source1_entity_id
        ),
        entity_metrics AS (
            SELECT
                v.source1_entity_id,
                COALESCE(t.true_count, 0) AS true_count,
                COALESCE(p.pred_count, 0) AS pred_count,
                COALESCE(x.tp, 0) AS tp,
                CASE
                    WHEN COALESCE(t.true_count, 0) = 0
                         AND COALESCE(p.pred_count, 0) = 0 THEN 1.0
                    WHEN COALESCE(x.tp, 0) = 0 THEN 0.0
                    ELSE
                        1.25 * x.tp
                        / (1.25 * x.tp
                           + (p.pred_count - x.tp)
                           + 0.25 * (t.true_count - x.tp))
                END AS entity_f0_5
            FROM validation_entities v
            LEFT JOIN true_counts t USING (source1_entity_id)
            LEFT JOIN pred_counts p USING (source1_entity_id)
            LEFT JOIN tp_counts x USING (source1_entity_id)
        )
        SELECT
            {float(threshold)} AS threshold,
            AVG(entity_f0_5) AS macro_entity_f0_5,
            SUM(tp) AS true_positives,
            SUM(pred_count - tp) AS false_positives,
            SUM(true_count - tp) AS false_negatives,
            SUM(pred_count) AS predicted_links,
            SUM(true_count) AS true_links,
            SUM(CASE WHEN pred_count = 0 THEN 1 ELSE 0 END) AS entities_predicted_empty,
            SUM(CASE WHEN true_count = 0 THEN 1 ELSE 0 END) AS true_singletons,
            SUM(CASE WHEN true_count = 0 AND pred_count = 0 THEN 1 ELSE 0 END)
                AS correctly_empty_entities
        FROM entity_metrics
    """).fetchdf().iloc[0].to_dict()

    tp = int(row["true_positives"])
    fp = int(row["false_positives"])
    fn = int(row["false_negatives"])
    row["micro_precision"] = tp / (tp + fp) if tp + fp else 0.0
    row["micro_recall"] = tp / (tp + fn) if tp + fn else 0.0
    return row


def threshold_grid(start, stop, step):
    values = []
    current = start
    while current <= stop + 1e-12:
        values.append(round(current, 6))
        current += step
    return values


def main():
    parser = argparse.ArgumentParser(description="Optimize entity-level macro F0.5 threshold.")
    parser.add_argument("--memory-limit", default="6GB")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--temp-limit", default="30GB")
    parser.add_argument("--start", type=float, default=0.30)
    parser.add_argument("--stop", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.025)
    args = parser.parse_args()

    validate_inputs()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    con = create_connection(args.memory_limit, args.threads, args.temp_limit)

    try:
        prepare_tables(con)
        results = []
        for threshold in threshold_grid(args.start, args.stop, args.step):
            result = evaluate_threshold(con, threshold)
            results.append(result)
            print(
                f"threshold={threshold:.3f} "
                f"macro_f0.5={result['macro_entity_f0_5']:.6f} "
                f"precision={result['micro_precision']:.6f} "
                f"recall={result['micro_recall']:.6f}"
            )

        report = pd.DataFrame(results).sort_values("threshold")
        report.to_csv(SEARCH_REPORT, index=False)
        best = report.sort_values(
            ["macro_entity_f0_5", "micro_precision"], ascending=[False, False]
        ).iloc[0].to_dict()

        best_clean = {}
        for key, value in best.items():
            if isinstance(value, (int, float)):
                best_clean[key] = float(value)
            else:
                best_clean[key] = value
        best_clean["metric"] = "macro entity-level F0.5"
        best_clean["beta"] = 0.5
        best_clean["search_start"] = args.start
        best_clean["search_stop"] = args.stop
        best_clean["search_step"] = args.step

        THRESHOLD_OUTPUT.write_text(json.dumps(best_clean, indent=2), encoding="utf-8")
        METRICS_REPORT.write_text(json.dumps(best_clean, indent=2), encoding="utf-8")

        print("\nPhase 4B complete.")
        print(json.dumps(best_clean, indent=2))
        print(f"Threshold search: {SEARCH_REPORT}")
        print(f"Best threshold: {THRESHOLD_OUTPUT}")
        print(f"Validation metrics: {METRICS_REPORT}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
