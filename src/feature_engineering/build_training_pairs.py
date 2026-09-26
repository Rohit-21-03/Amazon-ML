import argparse
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "train"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CANDIDATE_PATH = TRAIN_DIR / "train_candidate_pairs_address.parquet"
GROUND_TRUTH_PATH = TRAIN_DIR / "train_ground_truth_links.parquet"
OUTPUT_PATH = TRAIN_DIR / "train_labeled_pairs.parquet"


def sql_path(path):
    return Path(path).resolve().as_posix().replace("'", "''")


def validate_inputs():
    missing = [
        str(path)
        for path in (CANDIDATE_PATH, GROUND_TRUTH_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def create_connection(memory_limit, threads, temp_limit):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = OUTPUTS_DIR / "duckdb_temp_phase3"
    temp_dir.mkdir(parents=True, exist_ok=True)

    database_path = OUTPUTS_DIR / "phase3_training_pairs.duckdb"
    connection = duckdb.connect(str(database_path))
    connection.execute(f"SET memory_limit = '{memory_limit}'")
    connection.execute(f"SET threads = {threads}")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute(f"SET temp_directory = '{sql_path(temp_dir)}'")
    connection.execute(f"SET max_temp_directory_size = '{temp_limit}'")
    connection.execute("SET enable_progress_bar = true")
    return connection


def build_training_pairs(connection, negatives_per_matched, negatives_per_singleton):
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    candidates = sql_path(CANDIDATE_PATH)
    ground_truth = sql_path(GROUND_TRUTH_PATH)
    output = sql_path(OUTPUT_PATH)

    print("Labelling candidate pairs and selecting hard negatives...")

    connection.execute(
        f"""
        COPY (
            WITH gt AS (
                SELECT DISTINCT
                    CAST(source1_entity_id AS VARCHAR) AS source1_entity_id,
                    CAST(candidate_entity_id AS VARCHAR) AS candidate_entity_id
                FROM read_parquet('{ground_truth}')
            ),
            gt_entities AS (
                SELECT DISTINCT source1_entity_id
                FROM gt
            ),
            labelled AS (
                SELECT
                    c.*,
                    CASE
                        WHEN gt.candidate_entity_id IS NOT NULL THEN 1
                        ELSE 0
                    END AS label,
                    CASE
                        WHEN ge.source1_entity_id IS NOT NULL THEN 1
                        ELSE 0
                    END AS entity_has_true_match
                FROM read_parquet('{candidates}') AS c
                LEFT JOIN gt
                    ON CAST(c.source1_entity_id AS VARCHAR) = gt.source1_entity_id
                    AND CAST(c.candidate_entity_id AS VARCHAR) = gt.candidate_entity_id
                LEFT JOIN gt_entities AS ge
                    ON CAST(c.source1_entity_id AS VARCHAR) = ge.source1_entity_id
            ),
            ranked AS (
                SELECT
                    *,
                    CASE
                        WHEN label = 0 THEN
                            ROW_NUMBER() OVER (
                                PARTITION BY source1_entity_id, label
                                ORDER BY
                                    exact_name DESC,
                                    exact_address DESC,
                                    exact_core DESC,
                                    name_fuzzy_similarity DESC,
                                    shared_address_tokens DESC,
                                    address_blocking_score DESC,
                                    combined_blocking_score DESC,
                                    candidate_rank ASC,
                                    candidate_entity_id
                            )
                        ELSE 0
                    END AS negative_rank
                FROM labelled
            ),
            selected AS (
                SELECT *
                FROM ranked
                WHERE
                    label = 1
                    OR (
                        label = 0
                        AND entity_has_true_match = 1
                        AND negative_rank <= {negatives_per_matched}
                    )
                    OR (
                        label = 0
                        AND entity_has_true_match = 0
                        AND negative_rank <= {negatives_per_singleton}
                    )
            )
            SELECT
                source1_entity_id,
                candidate_entity_id,
                candidate_source,
                exact_name,
                exact_core,
                exact_address,
                shared_token_count,
                token_blocking_score,
                exact_blocking_score,
                combined_blocking_score,
                name_fuzzy_similarity,
                shared_address_tokens,
                address_blocking_score,
                candidate_rank,
                label,
                entity_has_true_match,
                negative_rank,
                CASE
                    WHEN hash(source1_entity_id) % 10 < 8 THEN 'train'
                    ELSE 'validation'
                END AS dataset_split
            FROM selected
        ) TO '{output}' (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 100000
        )
        """
    )


def validate_output(connection):
    output = sql_path(OUTPUT_PATH)

    summary = connection.execute(
        f"""
        SELECT
            COUNT(*) AS selected_pairs,
            SUM(label) AS positive_pairs,
            SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS negative_pairs,
            COUNT(DISTINCT source1_entity_id) AS source1_entities,
            SUM(CASE WHEN dataset_split = 'train' THEN 1 ELSE 0 END) AS train_pairs,
            SUM(CASE WHEN dataset_split = 'validation' THEN 1 ELSE 0 END) AS validation_pairs
        FROM read_parquet('{output}')
        """
    ).fetchdf()

    duplicate_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT source1_entity_id, candidate_entity_id, COUNT(*) AS n
            FROM read_parquet('{output}')
            GROUP BY source1_entity_id, candidate_entity_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    split_leakage = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT source1_entity_id, COUNT(DISTINCT dataset_split) AS n
            FROM read_parquet('{output}')
            GROUP BY source1_entity_id
            HAVING COUNT(DISTINCT dataset_split) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count != 0:
        raise ValueError(f"Duplicate candidate pairs found: {duplicate_count:,}")

    if split_leakage != 0:
        raise ValueError(f"Source 1 split leakage found: {split_leakage:,}")

    print("\nPhase 3A summary:")
    print(summary.to_string(index=False))
    print(f"Duplicate candidate pairs: {duplicate_count:,}")
    print(f"Source 1 split leakage: {split_leakage:,}")
    print(f"Created: {OUTPUT_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Create labelled training pairs from Phase 2D candidates."
    )
    parser.add_argument("--memory-limit", default="6GB")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--temp-limit", default="30GB")
    parser.add_argument("--negatives-per-matched", type=int, default=10)
    parser.add_argument("--negatives-per-singleton", type=int, default=5)
    args = parser.parse_args()

    validate_inputs()
    connection = create_connection(
        args.memory_limit,
        args.threads,
        args.temp_limit,
    )

    try:
        build_training_pairs(
            connection,
            args.negatives_per_matched,
            args.negatives_per_singleton,
        )
        validate_output(connection)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
