import argparse
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

STOP_TOKENS = {
    "a", "an", "and", "co", "company", "corp", "corporation",
    "for", "in", "inc", "limited", "llc", "llp", "ltd", "of",
    "private", "pvt", "service", "services", "the", "to",
}


def sql_path(path):
    return Path(path).resolve().as_posix().replace("'", "''")


def get_paths(split):
    base = PROCESSED_DIR / split
    return {
        "source1": base / f"{split}_source1_normalized.parquet",
        "source2": base / f"{split}_source2_normalized.parquet",
        "source3": base / f"{split}_source3_normalized.parquet",
        "exact": base / f"{split}_candidate_pairs.parquet",
        "expanded": base / f"{split}_candidate_pairs_expanded.parquet",
    }


def validate_inputs(paths):
    required = ("source1", "source2", "source3", "exact")
    missing = [str(paths[key]) for key in required if not paths[key].exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def connect_db(split, memory_limit, threads, temp_limit):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = OUTPUTS_DIR / "duckdb_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    db_path = OUTPUTS_DIR / f"phase2_expansion_{split}.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET threads = {threads}")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET temp_directory = '{sql_path(temp_dir)}'")
    con.execute(f"SET max_temp_directory_size = '{temp_limit}'")
    con.execute("SET enable_progress_bar = true")
    return con


def create_views(con, paths):
    s1 = sql_path(paths["source1"])
    s2 = sql_path(paths["source2"])
    s3 = sql_path(paths["source3"])
    exact = sql_path(paths["exact"])

    con.execute(f"""
        CREATE OR REPLACE VIEW source1 AS
        SELECT CAST(entity_id AS VARCHAR) AS source1_entity_id,
               CAST(name_core AS VARCHAR) AS name_core,
               CAST(country_normalized AS VARCHAR) AS country_normalized
        FROM read_parquet('{s1}')
    """)

    con.execute(f"""
        CREATE OR REPLACE VIEW targets AS
        SELECT CAST(entity_id AS VARCHAR) AS candidate_entity_id,
               'S2' AS candidate_source,
               CAST(name_core AS VARCHAR) AS name_core,
               CAST(country_normalized AS VARCHAR) AS country_normalized
        FROM read_parquet('{s2}')
        UNION ALL
        SELECT CAST(entity_id AS VARCHAR) AS candidate_entity_id,
               'S3' AS candidate_source,
               CAST(name_core AS VARCHAR) AS name_core,
               CAST(country_normalized AS VARCHAR) AS country_normalized
        FROM read_parquet('{s3}')
    """)

    con.execute(f"""
        CREATE OR REPLACE VIEW exact_candidates AS
        SELECT CAST(source1_entity_id AS VARCHAR) AS source1_entity_id,
               CAST(candidate_entity_id AS VARCHAR) AS candidate_entity_id,
               CAST(candidate_source AS VARCHAR) AS candidate_source,
               CAST(exact_name AS INTEGER) AS exact_name,
               CAST(exact_core AS INTEGER) AS exact_core,
               CAST(exact_address AS INTEGER) AS exact_address,
               CAST(blocking_score AS DOUBLE) AS exact_blocking_score
        FROM read_parquet('{exact}')
    """)


def create_stop_tokens(con):
    con.execute("CREATE OR REPLACE TEMP TABLE stop_tokens(token VARCHAR)")
    values = ",".join(f"('{token}')" for token in sorted(STOP_TOKENS))
    con.execute(f"INSERT INTO stop_tokens VALUES {values}")


def create_token_tables(con, max_token_frequency):
    print("Extracting and filtering target tokens...")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE valid_target_tokens AS
        WITH raw AS (
            SELECT DISTINCT candidate_entity_id, candidate_source,
                   country_normalized, token
            FROM targets,
            UNNEST(string_split(name_core, ' ')) AS u(token)
            WHERE name_core IS NOT NULL
              AND name_core <> ''
              AND token IS NOT NULL
              AND token <> ''
              AND LENGTH(token) >= 3
              AND NOT regexp_matches(token, '^[0-9]+$')
              AND token NOT IN (SELECT token FROM stop_tokens)
        ),
        frequencies AS (
            SELECT country_normalized, token, COUNT(*) AS token_count
            FROM raw
            GROUP BY country_normalized, token
            HAVING COUNT(*) <= {max_token_frequency}
        )
        SELECT raw.candidate_entity_id, raw.candidate_source,
               raw.country_normalized, raw.token, frequencies.token_count
        FROM raw
        JOIN frequencies
          ON raw.country_normalized = frequencies.country_normalized
         AND raw.token = frequencies.token
    """)

    print("Extracting Source 1 tokens...")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE source1_tokens AS
        SELECT DISTINCT source1_entity_id, country_normalized, token
        FROM source1,
        UNNEST(string_split(name_core, ' ')) AS u(token)
        WHERE name_core IS NOT NULL
          AND name_core <> ''
          AND token IS NOT NULL
          AND token <> ''
          AND LENGTH(token) >= 3
          AND NOT regexp_matches(token, '^[0-9]+$')
          AND token NOT IN (SELECT token FROM stop_tokens)
    """)


def create_token_candidates(con):
    print("Creating rare-token candidates...")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE token_candidates AS
        SELECT s.source1_entity_id,
               t.candidate_entity_id,
               t.candidate_source,
               COUNT(DISTINCT s.token) AS shared_token_count,
               SUM(1.0 / LN(2.0 + t.token_count)) AS token_blocking_score
        FROM source1_tokens AS s
        JOIN valid_target_tokens AS t
          ON s.country_normalized = t.country_normalized
         AND s.token = t.token
        GROUP BY s.source1_entity_id,
                 t.candidate_entity_id,
                 t.candidate_source
    """)


def write_expanded(con, output_path, max_candidates):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    out = sql_path(output_path)

    print("Combining and ranking candidates...")
    con.execute(f"""
        COPY (
            WITH combined AS (
                SELECT source1_entity_id, candidate_entity_id, candidate_source,
                       exact_name, exact_core, exact_address,
                       0 AS shared_token_count,
                       0.0 AS token_blocking_score,
                       exact_blocking_score
                FROM exact_candidates
                UNION ALL
                SELECT source1_entity_id, candidate_entity_id, candidate_source,
                       0, 0, 0, shared_token_count,
                       token_blocking_score, 0.0
                FROM token_candidates
            ),
            aggregated AS (
                SELECT source1_entity_id, candidate_entity_id, candidate_source,
                       MAX(exact_name) AS exact_name,
                       MAX(exact_core) AS exact_core,
                       MAX(exact_address) AS exact_address,
                       MAX(shared_token_count) AS shared_token_count,
                       MAX(token_blocking_score) AS token_blocking_score,
                       MAX(exact_blocking_score) AS exact_blocking_score
                FROM combined
                GROUP BY source1_entity_id, candidate_entity_id, candidate_source
            ),
            scored AS (
                SELECT *,
                       exact_name * 100.0
                       + exact_address * 80.0
                       + exact_core * 60.0
                       + shared_token_count * 10.0
                       + token_blocking_score AS combined_blocking_score
                FROM aggregated
            ),
            ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY source1_entity_id
                           ORDER BY combined_blocking_score DESC,
                                    exact_name DESC,
                                    exact_address DESC,
                                    exact_core DESC,
                                    shared_token_count DESC,
                                    candidate_entity_id
                       ) AS candidate_rank
                FROM scored
            )
            SELECT source1_entity_id, candidate_entity_id, candidate_source,
                   exact_name, exact_core, exact_address,
                   shared_token_count, token_blocking_score,
                   exact_blocking_score, combined_blocking_score,
                   candidate_rank
            FROM ranked
            WHERE candidate_rank <= {max_candidates}
        ) TO '{out}' (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 100000
        )
    """)


def print_summary(con, output_path):
    out = sql_path(output_path)
    summary = con.execute(f"""
        WITH counts AS (
            SELECT source1_entity_id, COUNT(*) AS candidate_count
            FROM read_parquet('{out}')
            GROUP BY source1_entity_id
        )
        SELECT SUM(candidate_count) AS total_candidate_pairs,
               COUNT(*) AS covered_source1_entities,
               AVG(candidate_count) AS average_candidates,
               MAX(candidate_count) AS maximum_candidates
        FROM counts
    """).fetchdf()
    print("\nExpanded candidate summary:")
    print(summary.to_string(index=False))


def run(split, memory_limit, threads, temp_limit,
        max_token_frequency, max_candidates):
    paths = get_paths(split)
    validate_inputs(paths)
    con = connect_db(split, memory_limit, threads, temp_limit)
    try:
        create_views(con, paths)
        create_stop_tokens(con)
        create_token_tables(con, max_token_frequency)
        create_token_candidates(con)
        write_expanded(con, paths["expanded"], max_candidates)
        print_summary(con, paths["expanded"])
        print(f"\nExpanded candidate file created: {paths['expanded']}")
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(
        description="Expand exact candidates with rare business-name tokens."
    )
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--memory-limit", default="6GB")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--temp-limit", default="30GB")
    parser.add_argument("--max-token-frequency", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=200)
    args = parser.parse_args()

    run(
        split=args.split,
        memory_limit=args.memory_limit,
        threads=args.threads,
        temp_limit=args.temp_limit,
        max_token_frequency=args.max_token_frequency,
        max_candidates=args.max_candidates,
    )


if __name__ == "__main__":
    main()
