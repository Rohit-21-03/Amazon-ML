import argparse
import json
from pathlib import Path

import duckdb


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


def connect(memory_limit, threads, temp_limit, project_root):
    outputs = project_root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    temp_dir = outputs / "duckdb_temp_validator"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(outputs / "submission_validator.duckdb"))
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{sql_path(temp_dir)}'")
    con.execute(f"SET max_temp_directory_size='{temp_limit}'")
    return con


def main():
    parser = argparse.ArgumentParser(description="Validate matching_results.tsv and candidate_pairs.tsv")
    parser.add_argument("--matching", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--test-dir", required=True, type=Path)
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--temp-limit", default="30GB")
    parser.add_argument("--report", type=Path, default=Path("reports/final_submission_validation.json"))
    args = parser.parse_args()

    for path in (args.matching, args.candidate, args.test_dir):
        if not path.exists():
            raise FileNotFoundError(f"Missing path: {path}")

    source1 = find_test_file(args.test_dir, "test_source1.tsv")
    source2 = find_test_file(args.test_dir, "test_source2.tsv")
    source3 = find_test_file(args.test_dir, "test_source3.tsv")
    project_root = Path.cwd()
    con = connect(args.memory_limit, args.threads, args.temp_limit, project_root)

    try:
        con.execute(f"""
            CREATE OR REPLACE VIEW source1 AS
            SELECT CAST(entity_id AS VARCHAR) source1_entity_id
            FROM read_csv('{sql_path(source1)}', delim='\\t', header=true, all_varchar=true)
        """)
        con.execute(f"""
            CREATE OR REPLACE VIEW targets AS
            SELECT CAST(entity_id AS VARCHAR) candidate_entity_id
            FROM read_csv('{sql_path(source2)}', delim='\\t', header=true, all_varchar=true)
            UNION ALL
            SELECT CAST(entity_id AS VARCHAR) candidate_entity_id
            FROM read_csv('{sql_path(source3)}', delim='\\t', header=true, all_varchar=true)
        """)
        con.execute(f"""
            CREATE OR REPLACE VIEW matching AS
            SELECT CAST(source1_entity_id AS VARCHAR) source1_entity_id,
                   COALESCE(CAST(matched_entity_ids AS VARCHAR), '') matched_entity_ids
            FROM read_csv(
                '{sql_path(args.matching)}', delim='\\t', header=true, all_varchar=true,
                columns={{'source1_entity_id':'VARCHAR','matched_entity_ids':'VARCHAR'}})
        """)
        con.execute(f"""
            CREATE OR REPLACE VIEW candidates AS
            SELECT CAST(source1_entity_id AS VARCHAR) source1_entity_id,
                   CAST(candidate_entity_id AS VARCHAR) candidate_entity_id
            FROM read_csv(
                '{sql_path(args.candidate)}', delim='\\t', header=true, all_varchar=true,
                columns={{'source1_entity_id':'VARCHAR','candidate_entity_id':'VARCHAR'}})
        """)

        print("Checking row counts and duplicate IDs...")
        report = con.execute("""
            SELECT
              (SELECT COUNT(*) FROM source1) source1_rows,
              (SELECT COUNT(DISTINCT source1_entity_id) FROM source1) source1_distinct,
              (SELECT COUNT(*) FROM matching) matching_rows,
              (SELECT COUNT(DISTINCT source1_entity_id) FROM matching) matching_distinct,
              (SELECT COUNT(*) FROM candidates) candidate_rows,
              (SELECT COUNT(*) FROM (SELECT source1_entity_id, candidate_entity_id, COUNT(*) n
                 FROM candidates GROUP BY 1,2 HAVING COUNT(*)>1)) duplicate_candidate_pairs
        """).fetchdf().iloc[0].to_dict()

        print("Checking submitted IDs...")
        report["missing_source1_rows"] = con.execute("""
            SELECT COUNT(*) FROM source1 s
            LEFT JOIN matching m USING(source1_entity_id)
            WHERE m.source1_entity_id IS NULL
        """).fetchone()[0]
        report["unknown_matching_source1_ids"] = con.execute("""
            SELECT COUNT(*) FROM matching m
            LEFT JOIN source1 s USING(source1_entity_id)
            WHERE s.source1_entity_id IS NULL
        """).fetchone()[0]
        report["unknown_candidate_source1_ids"] = con.execute("""
            SELECT COUNT(*) FROM candidates c
            LEFT JOIN source1 s USING(source1_entity_id)
            WHERE s.source1_entity_id IS NULL
        """).fetchone()[0]
        report["unknown_candidate_target_ids"] = con.execute("""
            SELECT COUNT(*) FROM candidates c
            LEFT JOIN targets t USING(candidate_entity_id)
            WHERE t.candidate_entity_id IS NULL
        """).fetchone()[0]

        print("Expanding predicted match lists...")
        con.execute("""
            CREATE OR REPLACE TEMP TABLE predictions AS
            SELECT m.source1_entity_id, TRIM(x.candidate_entity_id) candidate_entity_id
            FROM matching m,
            UNNEST(string_split(m.matched_entity_ids, ',')) x(candidate_entity_id)
            WHERE m.matched_entity_ids <> '' AND TRIM(x.candidate_entity_id) <> ''
        """)
        report["predicted_links"] = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        report["duplicate_predicted_links"] = con.execute("""
            SELECT COUNT(*) FROM (
              SELECT source1_entity_id, candidate_entity_id, COUNT(*) n
              FROM predictions GROUP BY 1,2 HAVING COUNT(*)>1)
        """).fetchone()[0]
        report["unknown_predicted_target_ids"] = con.execute("""
            SELECT COUNT(*) FROM predictions p
            LEFT JOIN targets t USING(candidate_entity_id)
            WHERE t.candidate_entity_id IS NULL
        """).fetchone()[0]
        report["predictions_not_in_candidates"] = con.execute("""
            SELECT COUNT(*) FROM predictions p
            LEFT JOIN candidates c
              ON p.source1_entity_id=c.source1_entity_id
             AND p.candidate_entity_id=c.candidate_entity_id
            WHERE c.candidate_entity_id IS NULL
        """).fetchone()[0]

        integer_report = {k: int(v) for k, v in report.items()}
        failures = []
        if integer_report["source1_rows"] != integer_report["source1_distinct"]:
            failures.append("Raw test Source 1 contains duplicate IDs")
        if integer_report["matching_rows"] != integer_report["source1_rows"]:
            failures.append("matching_results.tsv does not contain exactly one row per Source 1 ID")
        if integer_report["matching_distinct"] != integer_report["source1_distinct"]:
            failures.append("matching_results.tsv contains duplicate or missing Source 1 IDs")
        zero_checks = [
            "duplicate_candidate_pairs", "missing_source1_rows", "unknown_matching_source1_ids",
            "unknown_candidate_source1_ids", "unknown_candidate_target_ids",
            "duplicate_predicted_links", "unknown_predicted_target_ids",
            "predictions_not_in_candidates",
        ]
        for key in zero_checks:
            if integer_report[key] != 0:
                failures.append(f"{key} = {integer_report[key]:,}")

        integer_report["status"] = "PASS" if not failures else "FAIL"
        integer_report["failures"] = failures
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(integer_report, indent=2), encoding="utf-8")

        print(json.dumps(integer_report, indent=2))
        print(f"Report: {args.report}")
        if failures:
            raise SystemExit(1)
        print("\nPASS")
    finally:
        con.close()


if __name__ == "__main__":
    main()
