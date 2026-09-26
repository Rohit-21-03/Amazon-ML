from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train"
    / "train_candidate_pairs_address.parquet"
)

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train"
    / "train_ground_truth_links.parquet"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "phase2_candidate_recall.csv"
)


def sql_path(path):
    return path.resolve().as_posix().replace("'", "''")


def evaluate():
    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError(
            f"Candidate file missing: {CANDIDATE_PATH}"
        )

    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            f"Ground-truth links missing: "
            f"{GROUND_TRUTH_PATH}"
        )

    connection = duckdb.connect()

    connection.execute(
        "SET memory_limit = '6GB'"
    )

    connection.execute(
        "SET threads = 4"
    )

    candidates = sql_path(CANDIDATE_PATH)
    ground_truth = sql_path(GROUND_TRUTH_PATH)

    overall = connection.execute(
        f"""
        WITH gt AS (
            SELECT DISTINCT
                source1_entity_id,
                candidate_entity_id,
                candidate_source
            FROM read_parquet('{ground_truth}')
        ),

        candidates AS (
            SELECT DISTINCT
                source1_entity_id,
                candidate_entity_id
            FROM read_parquet('{candidates}')
        ),

        recovered AS (
            SELECT
                gt.source1_entity_id,
                gt.candidate_entity_id,
                gt.candidate_source,
                CASE
                    WHEN candidates.candidate_entity_id
                        IS NOT NULL
                    THEN 1
                    ELSE 0
                END AS recovered
            FROM gt
            LEFT JOIN candidates
                ON gt.source1_entity_id
                    = candidates.source1_entity_id
                AND gt.candidate_entity_id
                    = candidates.candidate_entity_id
        )

        SELECT
            'overall' AS group_name,
            COUNT(*) AS true_links,
            SUM(recovered) AS recovered_links,
            SUM(recovered) * 1.0 / COUNT(*)
                AS candidate_recall
        FROM recovered
        """
    ).fetchdf()

    by_source = connection.execute(
        f"""
        WITH gt AS (
            SELECT DISTINCT
                source1_entity_id,
                candidate_entity_id,
                candidate_source
            FROM read_parquet('{ground_truth}')
        ),

        candidates AS (
            SELECT DISTINCT
                source1_entity_id,
                candidate_entity_id
            FROM read_parquet('{candidates}')
        ),

        recovered AS (
            SELECT
                gt.candidate_source,
                CASE
                    WHEN candidates.candidate_entity_id
                        IS NOT NULL
                    THEN 1
                    ELSE 0
                END AS recovered
            FROM gt
            LEFT JOIN candidates
                ON gt.source1_entity_id
                    = candidates.source1_entity_id
                AND gt.candidate_entity_id
                    = candidates.candidate_entity_id
        )

        SELECT
            candidate_source AS group_name,
            COUNT(*) AS true_links,
            SUM(recovered) AS recovered_links,
            SUM(recovered) * 1.0 / COUNT(*)
                AS candidate_recall
        FROM recovered
        GROUP BY candidate_source
        ORDER BY candidate_source
        """
    ).fetchdf()

    entity_coverage = connection.execute(
        f"""
        WITH gt_entities AS (
            SELECT DISTINCT source1_entity_id
            FROM read_parquet('{ground_truth}')
        ),

        candidate_entities AS (
            SELECT DISTINCT source1_entity_id
            FROM read_parquet('{candidates}')
        )

        SELECT
            COUNT(*) AS entities_with_true_matches,
            COUNT(candidate_entities.source1_entity_id)
                AS entities_with_candidates,
            COUNT(candidate_entities.source1_entity_id)
                * 1.0 / COUNT(*)
                AS entity_candidate_coverage
        FROM gt_entities
        LEFT JOIN candidate_entities
            USING (source1_entity_id)
        """
    ).fetchdf()

    report = pd.concat(
        [overall, by_source],
        ignore_index=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        REPORT_PATH,
        index=False,
    )

    print()
    print("Candidate link recall:")
    print(report.to_string(index=False))

    print()
    print("Entity candidate coverage:")
    print(entity_coverage.to_string(index=False))

    print()
    print(f"Report saved: {REPORT_PATH}")

    connection.close()


if __name__ == "__main__":
    evaluate()