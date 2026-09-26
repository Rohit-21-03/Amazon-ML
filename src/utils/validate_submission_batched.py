import argparse
import csv
import json
import shutil
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_PARQUET = PROJECT_ROOT / "data" / "processed" / "test" / "test_candidate_pairs_address.parquet"
WORK_ROOT = PROJECT_ROOT / "outputs" / "validator_batched"
REPORT_PATH = PROJECT_ROOT / "reports" / "final_submission_validation.json"


def sql_path(path):
    return Path(path).resolve().as_posix().replace("'", "''")


def find_test_file(test_dir, filename):
    direct = test_dir / filename
    if direct.exists():
        return direct
    matches = list(test_dir.rglob(filename))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {test_dir}")
    raise ValueError(f"Multiple files named {filename} found under {test_dir}")


def create_connection(memory_limit, threads, temp_limit):
    temp_dir = PROJECT_ROOT / "outputs" / "duckdb_temp_validator_batched"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(PROJECT_ROOT / "outputs" / "submission_validator_batched.duckdb"))
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{sql_path(temp_dir)}'")
    con.execute(f"SET max_temp_directory_size='{temp_limit}'")
    return con


def partition_candidates(con, buckets, rebuild):
    candidate_dir = WORK_ROOT / "candidate_buckets"
    marker = candidate_dir / "_SUCCESS"
    if rebuild and candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    if marker.exists():
        print("Candidate buckets already exist; reusing them.")
        return candidate_dir
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    print("Partitioning candidate Parquet once for memory-safe validation...")
    con.execute(f"""
        COPY (
            SELECT
                CAST(source1_entity_id AS VARCHAR) AS source1_entity_id,
                CAST(candidate_entity_id AS VARCHAR) AS candidate_entity_id,
                hash(CAST(source1_entity_id AS VARCHAR)) % {buckets} AS bucket
            FROM read_parquet('{sql_path(CANDIDATE_PARQUET)}')
        ) TO '{sql_path(candidate_dir)}' (
            FORMAT PARQUET,
            PARTITION_BY (bucket),
            COMPRESSION ZSTD,
            OVERWRITE_OR_IGNORE TRUE
        )
    """)
    marker.write_text("ok", encoding="utf-8")
    return candidate_dir


def stream_matching_to_buckets(matching_path, buckets):
    prediction_dir = WORK_ROOT / "prediction_buckets"
    if prediction_dir.exists():
        shutil.rmtree(prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    handles = []
    writers = []
    for bucket in range(buckets):
        handle = (prediction_dir / f"pred_{bucket:04d}.tsv").open(
            "w", newline="", encoding="utf-8"
        )
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_id"])
        handles.append(handle)
        writers.append(writer)

    matching_rows = 0
    predicted_links = 0
    duplicate_source_rows = 0
    seen_source_ids = set()

    print("Streaming matching_results.tsv into validation buckets...")
    try:
        with matching_path.open("r", newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source, delimiter="\t")
            expected = ["source1_entity_id", "matched_entity_ids"]
            if reader.fieldnames != expected:
                raise ValueError(
                    f"Incorrect matching header: {reader.fieldnames}; expected {expected}"
                )
            for row in reader:
                source_id = row["source1_entity_id"].strip()
                matching_rows += 1
                if source_id in seen_source_ids:
                    duplicate_source_rows += 1
                else:
                    seen_source_ids.add(source_id)
                values = row["matched_entity_ids"].strip()
                if not values:
                    continue
                bucket = hash_string_bucket(source_id, buckets)
                local_seen = set()
                for candidate_id in values.split(","):
                    candidate_id = candidate_id.strip()
                    if not candidate_id:
                        continue
                    if candidate_id in local_seen:
                        writers[bucket].writerow([source_id, candidate_id])
                    else:
                        local_seen.add(candidate_id)
                        writers[bucket].writerow([source_id, candidate_id])
                    predicted_links += 1
                if matching_rows % 250000 == 0:
                    print(f"Matching rows streamed: {matching_rows:,}")
    finally:
        for handle in handles:
            handle.close()

    return prediction_dir, matching_rows, predicted_links, duplicate_source_rows, len(seen_source_ids)


def hash_string_bucket(value, buckets):
    # Bucket assignment is resolved by DuckDB later. This temporary Python bucket
    # only distributes files evenly and does not need to match DuckDB hash.
    import zlib
    return zlib.crc32(value.encode("utf-8")) % buckets


def validate_source_rows(con, source1_path, matching_rows, distinct_matching, duplicate_source_rows):
    source_stats = con.execute(f"""
        SELECT COUNT(*) AS rows, COUNT(DISTINCT CAST(entity_id AS VARCHAR)) AS distinct_ids
        FROM read_csv('{sql_path(source1_path)}', delim='\\t', header=true, all_varchar=true)
    """).fetchone()
    source_rows, source_distinct = map(int, source_stats)

    missing_or_unknown = None
    if matching_rows != source_rows or distinct_matching != source_distinct or duplicate_source_rows:
        missing_or_unknown = 1
    else:
        missing_or_unknown = 0
    return source_rows, source_distinct, missing_or_unknown


def validate_prediction_buckets(con, prediction_dir, candidate_dir, buckets):
    totals = {
        "duplicate_predicted_links": 0,
        "predictions_not_in_candidates": 0,
    }
    files = sorted(prediction_dir.glob("pred_*.tsv"))
    for index, pred_file in enumerate(files, start=1):
        # Each prediction file is independently small. Join against the partitioned
        # candidate dataset, while DuckDB prunes to matching Source 1 IDs.
        result = con.execute(f"""
            WITH p AS (
                SELECT source1_entity_id, candidate_entity_id
                FROM read_csv(
                    '{sql_path(pred_file)}', delim='\\t', header=true, all_varchar=true,
                    columns={{'source1_entity_id':'VARCHAR','candidate_entity_id':'VARCHAR'}}
                )
            ),
            duplicate_count AS (
                SELECT COUNT(*) AS n FROM (
                    SELECT source1_entity_id, candidate_entity_id, COUNT(*) AS c
                    FROM p GROUP BY 1,2 HAVING COUNT(*) > 1
                )
            ),
            missing_count AS (
                SELECT COUNT(*) AS n
                FROM p
                LEFT JOIN read_parquet('{sql_path(candidate_dir)}/**/*.parquet', hive_partitioning=true) c
                  ON p.source1_entity_id=c.source1_entity_id
                 AND p.candidate_entity_id=c.candidate_entity_id
                WHERE c.candidate_entity_id IS NULL
            )
            SELECT
                (SELECT n FROM duplicate_count),
                (SELECT n FROM missing_count)
        """).fetchone()
        totals["duplicate_predicted_links"] += int(result[0])
        totals["predictions_not_in_candidates"] += int(result[1])
        if index % 8 == 0 or index == len(files):
            print(f"Prediction buckets validated: {index}/{len(files)}")
    return totals


def main():
    parser = argparse.ArgumentParser(description="Memory-safe submission validator")
    parser.add_argument("--matching", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--test-dir", required=True, type=Path)
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--temp-limit", default="30GB")
    parser.add_argument("--buckets", type=int, default=64)
    parser.add_argument("--rebuild-candidate-buckets", action="store_true")
    args = parser.parse_args()

    for path in (args.matching, args.candidate, args.test_dir, CANDIDATE_PARQUET):
        if not path.exists():
            raise FileNotFoundError(f"Missing path: {path}")

    source1 = find_test_file(args.test_dir, "test_source1.tsv")
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    con = create_connection(args.memory_limit, args.threads, args.temp_limit)

    try:
        candidate_dir = partition_candidates(
            con, args.buckets, args.rebuild_candidate_buckets
        )
        (prediction_dir, matching_rows, predicted_links,
         duplicate_source_rows, distinct_matching) = stream_matching_to_buckets(
            args.matching, args.buckets
        )
        source_rows, source_distinct, source_mismatch = validate_source_rows(
            con, source1, matching_rows, distinct_matching, duplicate_source_rows
        )
        validation = validate_prediction_buckets(
            con, prediction_dir, candidate_dir, args.buckets
        )

        candidate_rows_tsv = con.execute(f"""
            SELECT COUNT(*) FROM read_csv(
                '{sql_path(args.candidate)}', delim='\\t', header=true, all_varchar=true,
                columns={{'source1_entity_id':'VARCHAR','candidate_entity_id':'VARCHAR'}})
        """).fetchone()[0]
        candidate_rows_parquet = pq.ParquetFile(CANDIDATE_PARQUET).metadata.num_rows

        report = {
            "source1_rows": source_rows,
            "source1_distinct": source_distinct,
            "matching_rows": matching_rows,
            "matching_distinct": distinct_matching,
            "duplicate_matching_source_rows": duplicate_source_rows,
            "predicted_links": predicted_links,
            "candidate_tsv_rows": int(candidate_rows_tsv),
            "candidate_parquet_rows": int(candidate_rows_parquet),
            **validation,
        }
        failures = []
        if source_mismatch:
            failures.append("matching_results.tsv does not contain exactly one row per Source 1 ID")
        if report["candidate_tsv_rows"] != report["candidate_parquet_rows"]:
            failures.append("candidate_pairs.tsv row count differs from frozen candidate Parquet")
        for key in ("duplicate_predicted_links", "predictions_not_in_candidates"):
            if report[key] != 0:
                failures.append(f"{key}={report[key]:,}")
        report["status"] = "PASS" if not failures else "FAIL"
        report["failures"] = failures

        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"Report: {REPORT_PATH}")
        if failures:
            raise SystemExit(1)
        print("\nPASS")
    finally:
        con.close()


if __name__ == "__main__":
    main()
