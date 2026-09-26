import argparse
import json
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SUBMISSION_DIR = PROJECT_ROOT / "submission"
REPORTS_DIR = PROJECT_ROOT / "reports"

SOURCE1_PATH = TEST_DIR / "test_source1_normalized.parquet"
SOURCE2_PATH = TEST_DIR / "test_source2_normalized.parquet"
SOURCE3_PATH = TEST_DIR / "test_source3_normalized.parquet"
CANDIDATE_PATH = TEST_DIR / "test_candidate_pairs_address.parquet"
SCORED_PATH = TEST_DIR / "test_scored_pairs.parquet"

MATCHING_PATH = SUBMISSION_DIR / "matching_results.tsv"
CANDIDATE_TSV_PATH = SUBMISSION_DIR / "candidate_pairs.tsv"
VALIDATION_REPORT = REPORTS_DIR / "phase6_submission_validation.json"


def sql_path(path):
    return Path(path).resolve().as_posix().replace("'", "''")


def validate_inputs():
    required = (
        SOURCE1_PATH,
        SOURCE2_PATH,
        SOURCE3_PATH,
        CANDIDATE_PATH,
        SCORED_PATH,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def create_connection(memory_limit, threads, temp_limit):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = OUTPUTS_DIR / "duckdb_temp_phase6"
    temp_dir.mkdir(parents=True, exist_ok=True)

    database_path = OUTPUTS_DIR / "phase6_submission.duckdb"
    connection = duckdb.connect(str(database_path))
    connection.execute(f"SET memory_limit = '{memory_limit}'")
    connection.execute(f"SET threads = {threads}")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute(f"SET temp_directory = '{sql_path(temp_dir)}'")
    connection.execute(f"SET max_temp_directory_size = '{temp_limit}'")
    connection.execute("SET enable_progress_bar = true")
    return connection


def prepare_views(connection):
    s1 = sql_path(SOURCE1_PATH)
    s2 = sql_path(SOURCE2_PATH)
    s3 = sql_path(SOURCE3_PATH)
    candidates = sql_path(CANDIDATE_PATH)
    scored = sql_path(SCORED_PATH)

    connection.execute(f"""
        CREATE OR REPLACE VIEW source1 AS
        SELECT CAST(entity_id AS VARCHAR) AS source1_entity_id
        FROM read_parquet('{s1}')
    """)

    connection.execute(f"""
        CREATE OR REPLACE VIEW valid_targets AS
        SELECT CAST(entity_id AS VARCHAR) AS candidate_entity_id
        FROM read_parquet('{s2}')
        UNION ALL
        SELECT CAST(entity_id AS VARCHAR) AS candidate_entity_id
        FROM read_parquet('{s3}')
    """)

    connection.execute(f"""
        CREATE OR REPLACE VIEW candidates AS
        SELECT DISTINCT
            CAST(source1_entity_id AS VARCHAR) AS source1_entity_id,
            CAST(candidate_entity_id AS VARCHAR) AS candidate_entity_id
        FROM read_parquet('{candidates}')
    """)

    connection.execute(f"""
        CREATE OR REPLACE VIEW accepted_predictions AS
        SELECT DISTINCT
            CAST(source1_entity_id AS VARCHAR) AS source1_entity_id,
            CAST(candidate_entity_id AS VARCHAR) AS candidate_entity_id,
            CAST(match_probability AS DOUBLE) AS match_probability
        FROM read_parquet('{scored}')
        WHERE predicted_match = 1
    """)


def validate_before_export(connection):
    stats = {}
    stats["source1_rows"] = connection.execute(
        "SELECT COUNT(*) FROM source1"
    ).fetchone()[0]
    stats["distinct_source1_ids"] = connection.execute(
        "SELECT COUNT(DISTINCT source1_entity_id) FROM source1"
    ).fetchone()[0]
    stats["candidate_pairs"] = connection.execute(
        "SELECT COUNT(*) FROM candidates"
    ).fetchone()[0]
    stats["accepted_predictions"] = connection.execute(
        "SELECT COUNT(*) FROM accepted_predictions"
    ).fetchone()[0]
    stats["invalid_candidate_source1_ids"] = connection.execute("""
        SELECT COUNT(*)
        FROM candidates c
        LEFT JOIN source1 s USING (source1_entity_id)
        WHERE s.source1_entity_id IS NULL
    """).fetchone()[0]
    stats["invalid_candidate_target_ids"] = connection.execute("""
        SELECT COUNT(*)
        FROM candidates c
        LEFT JOIN valid_targets t USING (candidate_entity_id)
        WHERE t.candidate_entity_id IS NULL
    """).fetchone()[0]
    stats["predictions_not_in_candidates"] = connection.execute("""
        SELECT COUNT(*)
        FROM accepted_predictions p
        LEFT JOIN candidates c
          ON p.source1_entity_id = c.source1_entity_id
         AND p.candidate_entity_id = c.candidate_entity_id
        WHERE c.candidate_entity_id IS NULL
    """).fetchone()[0]

    if stats["source1_rows"] != stats["distinct_source1_ids"]:
        raise ValueError("Test Source 1 contains duplicate entity IDs.")
    if stats["invalid_candidate_source1_ids"]:
        raise ValueError(
            f"Candidate pairs contain {stats['invalid_candidate_source1_ids']:,} invalid Source 1 IDs."
        )
    if stats["invalid_candidate_target_ids"]:
        raise ValueError(
            f"Candidate pairs contain {stats['invalid_candidate_target_ids']:,} invalid target IDs."
        )
    if stats["predictions_not_in_candidates"]:
        raise ValueError(
            f"Accepted predictions missing from candidates: {stats['predictions_not_in_candidates']:,}"
        )
    return stats


def export_files(connection):
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    for path in (MATCHING_PATH, CANDIDATE_TSV_PATH):
        if path.exists():
            path.unlink()

    print("Exporting candidate_pairs.tsv...")
    connection.execute(f"""
        COPY (
            SELECT source1_entity_id, candidate_entity_id
            FROM candidates
        ) TO '{sql_path(CANDIDATE_TSV_PATH)}' (
            FORMAT CSV,
            HEADER TRUE,
            DELIMITER '\t',
            QUOTE '"',
            ESCAPE '"'
        )
    """)

    print("Exporting matching_results.tsv...")
    connection.execute(f"""
        COPY (
            WITH grouped AS (
                SELECT
                    source1_entity_id,
                    string_agg(
                        candidate_entity_id,
                        ',' ORDER BY match_probability DESC, candidate_entity_id
                    ) AS matched_entity_ids
                FROM accepted_predictions
                GROUP BY source1_entity_id
            )
            SELECT
                s.source1_entity_id,
                COALESCE(g.matched_entity_ids, '') AS matched_entity_ids
            FROM source1 s
            LEFT JOIN grouped g USING (source1_entity_id)
        ) TO '{sql_path(MATCHING_PATH)}' (
            FORMAT CSV,
            HEADER TRUE,
            DELIMITER '\t',
            QUOTE '"',
            ESCAPE '"'
        )
    """)


def validate_exports(connection, initial_stats):
    matching = sql_path(MATCHING_PATH)
    candidate = sql_path(CANDIDATE_TSV_PATH)

    matching_rows = connection.execute(f"""
        SELECT COUNT(*)
        FROM read_csv(
            '{matching}', delim='\t', header=true, all_varchar=true,
            columns={{'source1_entity_id':'VARCHAR','matched_entity_ids':'VARCHAR'}}
        )
    """).fetchone()[0]

    distinct_matching_ids = connection.execute(f"""
        SELECT COUNT(DISTINCT source1_entity_id)
        FROM read_csv(
            '{matching}', delim='\t', header=true, all_varchar=true,
            columns={{'source1_entity_id':'VARCHAR','matched_entity_ids':'VARCHAR'}}
        )
    """).fetchone()[0]

    candidate_rows = connection.execute(f"""
        SELECT COUNT(*)
        FROM read_csv(
            '{candidate}', delim='\t', header=true, all_varchar=true,
            columns={{'source1_entity_id':'VARCHAR','candidate_entity_id':'VARCHAR'}}
        )
    """).fetchone()[0]

    report = dict(initial_stats)
    report.update({
        "matching_result_rows": matching_rows,
        "distinct_matching_source1_ids": distinct_matching_ids,
        "candidate_tsv_rows": candidate_rows,
        "matching_file_bytes": MATCHING_PATH.stat().st_size,
        "candidate_file_bytes": CANDIDATE_TSV_PATH.stat().st_size,
    })

    if matching_rows != initial_stats["source1_rows"]:
        raise ValueError(
            f"matching_results.tsv row mismatch: expected {initial_stats['source1_rows']:,}, "
            f"found {matching_rows:,}"
        )
    if distinct_matching_ids != initial_stats["distinct_source1_ids"]:
        raise ValueError("matching_results.tsv contains missing or duplicate Source 1 IDs.")
    if candidate_rows != initial_stats["candidate_pairs"]:
        raise ValueError(
            f"candidate_pairs.tsv row mismatch: expected {initial_stats['candidate_pairs']:,}, "
            f"found {candidate_rows:,}"
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nPhase 6 complete.")
    for key, value in report.items():
        if isinstance(value, int):
            print(f"{key}: {value:,}")
        else:
            print(f"{key}: {value}")
    print(f"Matching file: {MATCHING_PATH}")
    print(f"Candidate file: {CANDIDATE_TSV_PATH}")
    print(f"Validation report: {VALIDATION_REPORT}")


def main():
    parser = argparse.ArgumentParser(description="Generate and validate final submission TSV files.")
    parser.add_argument("--memory-limit", default="6GB")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--temp-limit", default="30GB")
    args = parser.parse_args()

    validate_inputs()
    connection = create_connection(args.memory_limit, args.threads, args.temp_limit)
    try:
        prepare_views(connection)
        stats = validate_before_export(connection)
        export_files(connection)
        validate_exports(connection, stats)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
