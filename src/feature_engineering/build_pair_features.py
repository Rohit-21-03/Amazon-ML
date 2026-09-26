import argparse
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "train"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"

LABELED_PAIRS = TRAIN_DIR / "train_labeled_pairs.parquet"
SOURCE1 = TRAIN_DIR / "train_source1_normalized.parquet"
SOURCE2 = TRAIN_DIR / "train_source2_normalized.parquet"
SOURCE3 = TRAIN_DIR / "train_source3_normalized.parquet"
OUTPUT = TRAIN_DIR / "train_pair_features.parquet"
REPORT = REPORTS_DIR / "phase3_feature_summary.csv"


def sql_path(path):
    return Path(path).resolve().as_posix().replace("'", "''")


def validate_inputs():
    required = (LABELED_PAIRS, SOURCE1, SOURCE2, SOURCE3)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def create_connection(memory_limit, threads, temp_limit):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = OUTPUTS_DIR / "duckdb_temp_phase3b"
    temp_dir.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect(str(OUTPUTS_DIR / "phase3_pair_features.duckdb"))
    connection.execute(f"SET memory_limit = '{memory_limit}'")
    connection.execute(f"SET threads = {threads}")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute(f"SET temp_directory = '{sql_path(temp_dir)}'")
    connection.execute(f"SET max_temp_directory_size = '{temp_limit}'")
    connection.execute("SET enable_progress_bar = true")
    return connection


def build_features(connection):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        OUTPUT.unlink()

    pairs = sql_path(LABELED_PAIRS)
    s1 = sql_path(SOURCE1)
    s2 = sql_path(SOURCE2)
    s3 = sql_path(SOURCE3)
    output = sql_path(OUTPUT)

    print("Joining selected pairs with normalized source records...")
    print("Calculating pairwise name and address features...")

    connection.execute(f"""
        COPY (
            WITH source1 AS (
                SELECT
                    CAST(entity_id AS VARCHAR) AS source1_entity_id,
                    COALESCE(CAST(name_normalized AS VARCHAR), '') AS s1_name,
                    COALESCE(CAST(name_core AS VARCHAR), '') AS s1_name_core,
                    COALESCE(CAST(address_normalized AS VARCHAR), '') AS s1_address,
                    COALESCE(CAST(country_normalized AS VARCHAR), '') AS s1_country
                FROM read_parquet('{s1}')
            ),
            targets AS (
                SELECT
                    CAST(entity_id AS VARCHAR) AS candidate_entity_id,
                    'S2' AS candidate_source,
                    COALESCE(CAST(name_normalized AS VARCHAR), '') AS target_name,
                    COALESCE(CAST(name_core AS VARCHAR), '') AS target_name_core,
                    COALESCE(CAST(address_normalized AS VARCHAR), '') AS target_address,
                    COALESCE(CAST(country_normalized AS VARCHAR), '') AS target_country
                FROM read_parquet('{s2}')
                UNION ALL
                SELECT
                    CAST(entity_id AS VARCHAR),
                    'S3',
                    COALESCE(CAST(name_normalized AS VARCHAR), ''),
                    COALESCE(CAST(name_core AS VARCHAR), ''),
                    COALESCE(CAST(address_normalized AS VARCHAR), ''),
                    COALESCE(CAST(country_normalized AS VARCHAR), '')
                FROM read_parquet('{s3}')
            ),
            joined AS (
                SELECT
                    p.*,
                    s.s1_name,
                    s.s1_name_core,
                    s.s1_address,
                    s.s1_country,
                    t.target_name,
                    t.target_name_core,
                    t.target_address,
                    t.target_country
                FROM read_parquet('{pairs}') AS p
                INNER JOIN source1 AS s
                    ON CAST(p.source1_entity_id AS VARCHAR) = s.source1_entity_id
                INNER JOIN targets AS t
                    ON CAST(p.candidate_entity_id AS VARCHAR) = t.candidate_entity_id
                    AND CAST(p.candidate_source AS VARCHAR) = t.candidate_source
            ),
            raw_features AS (
                SELECT
                    *,
                    LENGTH(s1_name) AS s1_name_length,
                    LENGTH(target_name) AS target_name_length,
                    LENGTH(s1_name_core) AS s1_core_length,
                    LENGTH(target_name_core) AS target_core_length,
                    LENGTH(s1_address) AS s1_address_length,
                    LENGTH(target_address) AS target_address_length,
                    CASE WHEN s1_country = target_country THEN 1 ELSE 0 END AS country_equal,
                    CASE WHEN s1_name = '' THEN 1 ELSE 0 END AS s1_name_missing,
                    CASE WHEN target_name = '' THEN 1 ELSE 0 END AS target_name_missing,
                    CASE WHEN s1_address = '' THEN 1 ELSE 0 END AS s1_address_missing,
                    CASE WHEN target_address = '' THEN 1 ELSE 0 END AS target_address_missing,
                    CASE WHEN candidate_source = 'S2' THEN 1 ELSE 0 END AS candidate_is_s2,
                    jaro_winkler_similarity(s1_name, target_name) AS name_jaro_winkler,
                    jaro_winkler_similarity(s1_name_core, target_name_core) AS core_jaro_winkler,
                    CASE
                        WHEN s1_address = '' OR target_address = '' THEN 0.0
                        ELSE jaro_winkler_similarity(s1_address, target_address)
                    END AS address_jaro_winkler,
                    levenshtein(s1_name, target_name) AS name_levenshtein_distance,
                    levenshtein(s1_name_core, target_name_core) AS core_levenshtein_distance,
                    CASE
                        WHEN s1_address = '' OR target_address = '' THEN 9999
                        ELSE levenshtein(s1_address, target_address)
                    END AS address_levenshtein_distance,
                    CASE
                        WHEN s1_name_core <> '' AND target_name_core <> ''
                             AND (contains(s1_name_core, target_name_core)
                                  OR contains(target_name_core, s1_name_core))
                        THEN 1 ELSE 0
                    END AS core_contains,
                    CASE
                        WHEN s1_address <> '' AND target_address <> ''
                             AND (contains(s1_address, target_address)
                                  OR contains(target_address, s1_address))
                        THEN 1 ELSE 0
                    END AS address_contains
                FROM joined
            )
            SELECT
                source1_entity_id,
                candidate_entity_id,
                candidate_source,
                label,
                dataset_split,
                entity_has_true_match,
                candidate_rank,
                candidate_is_s2,
                exact_name,
                exact_core,
                exact_address,
                shared_token_count,
                shared_address_tokens,
                token_blocking_score,
                address_blocking_score,
                exact_blocking_score,
                combined_blocking_score,
                name_fuzzy_similarity,
                country_equal,
                s1_name_missing,
                target_name_missing,
                s1_address_missing,
                target_address_missing,
                name_jaro_winkler,
                core_jaro_winkler,
                address_jaro_winkler,
                name_levenshtein_distance,
                core_levenshtein_distance,
                address_levenshtein_distance,
                1.0 - (
                    name_levenshtein_distance * 1.0
                    / GREATEST(s1_name_length, target_name_length, 1)
                ) AS name_levenshtein_similarity,
                1.0 - (
                    core_levenshtein_distance * 1.0
                    / GREATEST(s1_core_length, target_core_length, 1)
                ) AS core_levenshtein_similarity,
                CASE
                    WHEN s1_address_missing = 1 OR target_address_missing = 1 THEN 0.0
                    ELSE 1.0 - (
                        address_levenshtein_distance * 1.0
                        / GREATEST(s1_address_length, target_address_length, 1)
                    )
                END AS address_levenshtein_similarity,
                ABS(s1_name_length - target_name_length) AS name_length_difference,
                ABS(s1_core_length - target_core_length) AS core_length_difference,
                ABS(s1_address_length - target_address_length) AS address_length_difference,
                core_contains,
                address_contains
            FROM raw_features
        ) TO '{output}' (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 100000
        )
    """)


def validate_and_report(connection):
    output = sql_path(OUTPUT)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    missing_labels = connection.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{output}') WHERE label IS NULL
    """).fetchone()[0]

    duplicates = connection.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT source1_entity_id, candidate_entity_id, COUNT(*) AS n
            FROM read_parquet('{output}')
            GROUP BY source1_entity_id, candidate_entity_id
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    summary = connection.execute(f"""
        SELECT
            dataset_split,
            label,
            COUNT(*) AS pair_count,
            AVG(name_jaro_winkler) AS avg_name_jaro,
            AVG(core_jaro_winkler) AS avg_core_jaro,
            AVG(address_jaro_winkler) AS avg_address_jaro,
            AVG(name_levenshtein_similarity) AS avg_name_levenshtein,
            AVG(address_levenshtein_similarity) AS avg_address_levenshtein
        FROM read_parquet('{output}')
        GROUP BY dataset_split, label
        ORDER BY dataset_split, label
    """).fetchdf()

    if missing_labels:
        raise ValueError(f"Missing labels found: {missing_labels:,}")
    if duplicates:
        raise ValueError(f"Duplicate pairs found: {duplicates:,}")

    summary.to_csv(REPORT, index=False)
    print("\nPhase 3B feature summary:")
    print(summary.to_string(index=False))
    print(f"Missing labels: {missing_labels:,}")
    print(f"Duplicate pairs: {duplicates:,}")
    print(f"Feature file: {OUTPUT}")
    print(f"Report: {REPORT}")


def main():
    parser = argparse.ArgumentParser(description="Build pairwise training features.")
    parser.add_argument("--memory-limit", default="6GB")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--temp-limit", default="30GB")
    args = parser.parse_args()

    validate_inputs()
    connection = create_connection(args.memory_limit, args.threads, args.temp_limit)
    try:
        build_features(connection)
        validate_and_report(connection)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
