import argparse
import csv
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
REPORT_PATH = REPORTS_DIR / "phase6_submission_validation.json"


def sql_path(path):
    return Path(path).resolve().as_posix().replace("'", "''")


def validate_inputs():
    required = [SOURCE1_PATH, SOURCE2_PATH, SOURCE3_PATH, CANDIDATE_PATH, SCORED_PATH]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def connect(memory_limit, threads, temp_limit):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    temp = OUTPUTS_DIR / "duckdb_temp_phase6_batched"
    temp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(OUTPUTS_DIR / "phase6_batched.duckdb"))
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{sql_path(temp)}'")
    con.execute(f"SET max_temp_directory_size='{temp_limit}'")
    return con


def prepare(con, buckets):
    print("Preparing bucketed Source 1 and accepted predictions...")
    con.execute(f"""
        CREATE OR REPLACE TABLE source1_work AS
        SELECT CAST(entity_id AS VARCHAR) source1_entity_id,
               hash(CAST(entity_id AS VARCHAR)) % {buckets} bucket
        FROM read_parquet('{sql_path(SOURCE1_PATH)}')
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE accepted_work AS
        SELECT CAST(source1_entity_id AS VARCHAR) source1_entity_id,
               CAST(candidate_entity_id AS VARCHAR) candidate_entity_id,
               CAST(match_probability AS DOUBLE) match_probability,
               hash(CAST(source1_entity_id AS VARCHAR)) % {buckets} bucket
        FROM read_parquet('{sql_path(SCORED_PATH)}')
        WHERE predicted_match=1
    """)


def validate_before(con):
    stats = {}
    stats["source1_rows"] = con.execute("SELECT COUNT(*) FROM source1_work").fetchone()[0]
    stats["distinct_source1_ids"] = con.execute(
        "SELECT COUNT(DISTINCT source1_entity_id) FROM source1_work"
    ).fetchone()[0]
    stats["candidate_pairs"] = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{sql_path(CANDIDATE_PATH)}')"
    ).fetchone()[0]
    stats["accepted_predictions"] = con.execute("SELECT COUNT(*) FROM accepted_work").fetchone()[0]
    stats["predictions_not_in_candidates"] = con.execute(f"""
        SELECT COUNT(*)
        FROM accepted_work p
        LEFT JOIN read_parquet('{sql_path(CANDIDATE_PATH)}') c
          ON p.source1_entity_id=CAST(c.source1_entity_id AS VARCHAR)
         AND p.candidate_entity_id=CAST(c.candidate_entity_id AS VARCHAR)
        WHERE c.candidate_entity_id IS NULL
    """).fetchone()[0]
    if stats["source1_rows"] != stats["distinct_source1_ids"]:
        raise ValueError("Duplicate Source 1 IDs detected.")
    if stats["predictions_not_in_candidates"]:
        raise ValueError("Accepted predictions are missing from candidate pairs.")
    return stats


def export_candidate_pairs(con):
    if CANDIDATE_TSV_PATH.exists() and CANDIDATE_TSV_PATH.stat().st_size > 0:
        print("candidate_pairs.tsv already exists; keeping it.")
        return
    print("Exporting candidate_pairs.tsv...")
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        COPY (
          SELECT CAST(source1_entity_id AS VARCHAR) source1_entity_id,
                 CAST(candidate_entity_id AS VARCHAR) candidate_entity_id
          FROM read_parquet('{sql_path(CANDIDATE_PATH)}')
        ) TO '{sql_path(CANDIDATE_TSV_PATH)}'
        (FORMAT CSV, HEADER TRUE, DELIMITER '\t', QUOTE '"', ESCAPE '"')
    """)


def export_matching_batched(con, buckets):
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    if MATCHING_PATH.exists():
        MATCHING_PATH.unlink()

    total_rows = 0
    with MATCHING_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])

        for bucket in range(buckets):
            rows = con.execute(f"""
                WITH grouped AS (
                    SELECT source1_entity_id,
                           string_agg(candidate_entity_id, ',' ORDER BY match_probability DESC,
                                      candidate_entity_id) matched_entity_ids
                    FROM accepted_work
                    WHERE bucket={bucket}
                    GROUP BY source1_entity_id
                )
                SELECT s.source1_entity_id,
                       COALESCE(g.matched_entity_ids, '') matched_entity_ids
                FROM source1_work s
                LEFT JOIN grouped g USING(source1_entity_id)
                WHERE s.bucket={bucket}
            """).fetchall()
            writer.writerows(rows)
            total_rows += len(rows)
            if (bucket + 1) % 8 == 0 or bucket + 1 == buckets:
                print(f"Buckets: {bucket + 1}/{buckets} | matching rows: {total_rows:,}")
    return total_rows


def finish(con, stats, matching_rows):
    candidate_tsv_rows = con.execute(f"""
        SELECT COUNT(*) FROM read_csv(
          '{sql_path(CANDIDATE_TSV_PATH)}', delim='\t', header=true, all_varchar=true,
          columns={{'source1_entity_id':'VARCHAR','candidate_entity_id':'VARCHAR'}})
    """).fetchone()[0]
    if matching_rows != stats["source1_rows"]:
        raise ValueError(
            f"matching_results row mismatch: expected {stats['source1_rows']:,}, got {matching_rows:,}"
        )
    if candidate_tsv_rows != stats["candidate_pairs"]:
        raise ValueError(
            f"candidate_pairs row mismatch: expected {stats['candidate_pairs']:,}, got {candidate_tsv_rows:,}"
        )
    stats.update({
        "matching_result_rows": matching_rows,
        "candidate_tsv_rows": candidate_tsv_rows,
        "matching_file_bytes": MATCHING_PATH.stat().st_size,
        "candidate_file_bytes": CANDIDATE_TSV_PATH.stat().st_size,
    })
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("\nPhase 6 completed successfully.")
    print(json.dumps(stats, indent=2))
    print(f"Matching file: {MATCHING_PATH}")
    print(f"Candidate file: {CANDIDATE_TSV_PATH}")
    print(f"Report: {REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Memory-safe bucketed Phase 6 export.")
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--temp-limit", default="30GB")
    parser.add_argument("--buckets", type=int, default=128)
    args = parser.parse_args()
    validate_inputs()
    con = connect(args.memory_limit, args.threads, args.temp_limit)
    try:
        prepare(con, args.buckets)
        stats = validate_before(con)
        export_candidate_pairs(con)
        rows = export_matching_batched(con, args.buckets)
        finish(con, stats, rows)
    finally:
        con.close()


if __name__ == "__main__":
    main()
