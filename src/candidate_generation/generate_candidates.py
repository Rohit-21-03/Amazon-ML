import argparse
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATABASE_DIR = PROJECT_ROOT / "outputs"

DEFAULT_MEMORY_LIMIT = "6GB"
DEFAULT_THREADS = 4
DEFAULT_MAX_BLOCK_SIZE = 200
DEFAULT_MAX_CANDIDATES = 250


def sql_path(path):
    return path.resolve().as_posix().replace("'", "''")


def get_paths(split):
    split_dir = PROCESSED_DIR / split

    source1_path = (
        split_dir
        / f"{split}_source1_normalized.parquet"
    )

    source2_path = (
        split_dir
        / f"{split}_source2_normalized.parquet"
    )

    source3_path = (
        split_dir
        / f"{split}_source3_normalized.parquet"
    )

    output_path = (
        split_dir
        / f"{split}_candidate_pairs.parquet"
    )

    return {
        "source1": source1_path,
        "source2": source2_path,
        "source3": source3_path,
        "output": output_path,
    }


def validate_inputs(paths):
    missing_files = [
        str(path)
        for key, path in paths.items()
        if key != "output" and not path.exists()
    ]

    if missing_files:
        missing_text = "\n".join(missing_files)

        raise FileNotFoundError(
            "Missing Phase 1 files:\n"
            f"{missing_text}"
        )


def create_connection(
    database_path,
    memory_limit,
    threads,
):
    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = duckdb.connect(
        str(database_path)
    )

    connection.execute(
        f"SET memory_limit = '{memory_limit}'"
    )

    connection.execute(
        f"SET threads = {threads}"
    )

    connection.execute(
        "SET preserve_insertion_order = false"
    )

    temp_directory = (
        PROJECT_ROOT
        / "outputs"
        / "duckdb_temp"
    )

    temp_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection.execute(
        f"SET temp_directory = "
        f"'{sql_path(temp_directory)}'"
    )

    return connection


def create_source_views(connection, paths):
    source1 = sql_path(paths["source1"])
    source2 = sql_path(paths["source2"])
    source3 = sql_path(paths["source3"])

    connection.execute(
        f"""
        CREATE OR REPLACE VIEW source1 AS
        SELECT
            CAST(entity_id AS VARCHAR) AS source1_entity_id,
            CAST(name_normalized AS VARCHAR) AS name_normalized,
            CAST(name_core AS VARCHAR) AS name_core,
            CAST(address_normalized AS VARCHAR)
                AS address_normalized,
            CAST(country_normalized AS VARCHAR)
                AS country_normalized
        FROM read_parquet('{source1}')
        """
    )

    connection.execute(
        f"""
        CREATE OR REPLACE VIEW targets AS
        SELECT
            CAST(entity_id AS VARCHAR) AS candidate_entity_id,
            'S2' AS candidate_source,
            CAST(name_normalized AS VARCHAR) AS name_normalized,
            CAST(name_core AS VARCHAR) AS name_core,
            CAST(address_normalized AS VARCHAR)
                AS address_normalized,
            CAST(country_normalized AS VARCHAR)
                AS country_normalized
        FROM read_parquet('{source2}')

        UNION ALL

        SELECT
            CAST(entity_id AS VARCHAR) AS candidate_entity_id,
            'S3' AS candidate_source,
            CAST(name_normalized AS VARCHAR) AS name_normalized,
            CAST(name_core AS VARCHAR) AS name_core,
            CAST(address_normalized AS VARCHAR)
                AS address_normalized,
            CAST(country_normalized AS VARCHAR)
                AS country_normalized
        FROM read_parquet('{source3}')
        """
    )


def create_block_statistics(
    connection,
    max_block_size,
):
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE valid_name_blocks AS
        SELECT
            country_normalized,
            name_normalized
        FROM targets
        WHERE
            name_normalized IS NOT NULL
            AND name_normalized <> ''
            AND LENGTH(name_normalized) >= 3
        GROUP BY
            country_normalized,
            name_normalized
        HAVING COUNT(*) <= {max_block_size}
        """
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE valid_core_blocks AS
        SELECT
            country_normalized,
            name_core
        FROM targets
        WHERE
            name_core IS NOT NULL
            AND name_core <> ''
            AND LENGTH(name_core) >= 4
        GROUP BY
            country_normalized,
            name_core
        HAVING COUNT(*) <= {max_block_size}
        """
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE valid_address_blocks AS
        SELECT
            country_normalized,
            address_normalized
        FROM targets
        WHERE
            address_normalized IS NOT NULL
            AND address_normalized <> ''
            AND LENGTH(address_normalized) >= 8
        GROUP BY
            country_normalized,
            address_normalized
        HAVING COUNT(*) <= {max_block_size}
        """
    )


def create_raw_candidates(connection):
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE raw_candidates AS

        SELECT
            s.source1_entity_id,
            t.candidate_entity_id,
            t.candidate_source,
            1 AS exact_name,
            0 AS exact_core,
            0 AS exact_address
        FROM source1 AS s
        INNER JOIN valid_name_blocks AS b
            ON s.country_normalized = b.country_normalized
            AND s.name_normalized = b.name_normalized
        INNER JOIN targets AS t
            ON b.country_normalized = t.country_normalized
            AND b.name_normalized = t.name_normalized

        UNION ALL

        SELECT
            s.source1_entity_id,
            t.candidate_entity_id,
            t.candidate_source,
            0 AS exact_name,
            1 AS exact_core,
            0 AS exact_address
        FROM source1 AS s
        INNER JOIN valid_core_blocks AS b
            ON s.country_normalized = b.country_normalized
            AND s.name_core = b.name_core
        INNER JOIN targets AS t
            ON b.country_normalized = t.country_normalized
            AND b.name_core = t.name_core

        UNION ALL

        SELECT
            s.source1_entity_id,
            t.candidate_entity_id,
            t.candidate_source,
            0 AS exact_name,
            0 AS exact_core,
            1 AS exact_address
        FROM source1 AS s
        INNER JOIN valid_address_blocks AS b
            ON s.country_normalized = b.country_normalized
            AND s.address_normalized = b.address_normalized
        INNER JOIN targets AS t
            ON b.country_normalized = t.country_normalized
            AND b.address_normalized = t.address_normalized
        """
    )


def write_candidates(
    connection,
    output_path,
    max_candidates,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        output_path.unlink()

    output_sql_path = sql_path(output_path)

    connection.execute(
        f"""
        COPY (
            WITH combined AS (
                SELECT
                    source1_entity_id,
                    candidate_entity_id,
                    candidate_source,
                    MAX(exact_name) AS exact_name,
                    MAX(exact_core) AS exact_core,
                    MAX(exact_address) AS exact_address
                FROM raw_candidates
                GROUP BY
                    source1_entity_id,
                    candidate_entity_id,
                    candidate_source
            ),

            scored AS (
                SELECT
                    *,
                    (
                        exact_name * 4
                        + exact_core * 2
                        + exact_address * 3
                    ) AS blocking_score
                FROM combined
            ),

            ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY source1_entity_id
                        ORDER BY
                            blocking_score DESC,
                            exact_name DESC,
                            exact_address DESC,
                            candidate_entity_id
                    ) AS candidate_rank
                FROM scored
            )

            SELECT
                source1_entity_id,
                candidate_entity_id,
                candidate_source,
                exact_name,
                exact_core,
                exact_address,
                blocking_score,
                candidate_rank
            FROM ranked
            WHERE candidate_rank <= {max_candidates}
        )
        TO '{output_sql_path}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 100000
        )
        """
    )


def print_summary(connection, output_path):
    output_sql_path = sql_path(output_path)

    summary = connection.execute(
        f"""
        SELECT
            COUNT(*) AS total_candidate_pairs,
            COUNT(DISTINCT source1_entity_id)
                AS covered_source1_entities,
            AVG(candidate_count)
                AS average_candidates_for_covered_entities,
            MAX(candidate_count)
                AS maximum_candidates
        FROM (
            SELECT
                source1_entity_id,
                COUNT(*) AS candidate_count
            FROM read_parquet('{output_sql_path}')
            GROUP BY source1_entity_id
        )
        """
    ).fetchdf()

    source_counts = connection.execute(
        f"""
        SELECT
            candidate_source,
            COUNT(*) AS candidate_pairs
        FROM read_parquet('{output_sql_path}')
        GROUP BY candidate_source
        ORDER BY candidate_source
        """
    ).fetchdf()

    rule_counts = connection.execute(
        f"""
        SELECT
            SUM(exact_name) AS exact_name_pairs,
            SUM(exact_core) AS exact_core_pairs,
            SUM(exact_address) AS exact_address_pairs
        FROM read_parquet('{output_sql_path}')
        """
    ).fetchdf()

    print()
    print("Candidate summary:")
    print(summary.to_string(index=False))

    print()
    print("Candidate source distribution:")
    print(source_counts.to_string(index=False))

    print()
    print("Blocking-rule coverage:")
    print(rule_counts.to_string(index=False))


def generate_candidates(
    split,
    memory_limit,
    threads,
    max_block_size,
    max_candidates,
):
    paths = get_paths(split)
    validate_inputs(paths)

    database_path = (
        DATABASE_DIR
        / f"phase2_{split}.duckdb"
    )

    connection = create_connection(
        database_path=database_path,
        memory_limit=memory_limit,
        threads=threads,
    )

    try:
        print("Creating source views...")
        create_source_views(connection, paths)

        print("Calculating block frequencies...")
        create_block_statistics(
            connection,
            max_block_size=max_block_size,
        )

        print("Creating exact-match candidates...")
        create_raw_candidates(connection)

        print("Combining and ranking candidates...")
        write_candidates(
            connection,
            output_path=paths["output"],
            max_candidates=max_candidates,
        )

        print_summary(
            connection,
            paths["output"],
        )

        print()
        print(
            f"Candidate file created: "
            f"{paths['output']}"
        )

    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate exact blocking candidates "
            "for Amazon ML Challenge."
        )
    )

    parser.add_argument(
        "--split",
        choices=["train", "test"],
        required=True,
    )

    parser.add_argument(
        "--memory-limit",
        default=DEFAULT_MEMORY_LIMIT,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
    )

    parser.add_argument(
        "--max-block-size",
        type=int,
        default=DEFAULT_MAX_BLOCK_SIZE,
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
    )

    args = parser.parse_args()

    generate_candidates(
        split=args.split,
        memory_limit=args.memory_limit,
        threads=args.threads,
        max_block_size=args.max_block_size,
        max_candidates=args.max_candidates,
    )


if __name__ == "__main__":
    main()