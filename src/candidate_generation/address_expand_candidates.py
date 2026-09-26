import argparse
from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

ADDRESS_STOP_TOKENS = {
    "and", "apt", "ave", "bldg", "blvd", "building", "dr", "drive",
    "fl", "floor", "hwy", "in", "lane", "ln", "main", "near", "of",
    "rd", "road", "st", "ste", "street", "suite", "the", "to", "unit"
}


def sql_path(path):
    return Path(path).resolve().as_posix().replace("'", "''")


def get_paths(split):
    base = PROCESSED_DIR / split
    return {
        "source1": base / f"{split}_source1_normalized.parquet",
        "source2": base / f"{split}_source2_normalized.parquet",
        "source3": base / f"{split}_source3_normalized.parquet",
        "phase2c": base / f"{split}_candidate_pairs_fuzzy.parquet",
        "output": base / f"{split}_candidate_pairs_address.parquet",
    }


def validate_inputs(paths):
    required = ("source1", "source2", "source3", "phase2c")
    missing = [str(paths[key]) for key in required if not paths[key].exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def connect_db(split, memory_limit, threads, temp_limit):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = OUTPUTS_DIR / "duckdb_temp_2d"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(OUTPUTS_DIR / f"phase2d_{split}.duckdb"))
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
    phase2c = sql_path(paths["phase2c"])

    con.execute(f"""
        CREATE OR REPLACE VIEW source1 AS
        SELECT CAST(entity_id AS VARCHAR) AS source1_entity_id,
               CAST(name_core AS VARCHAR) AS name_core,
               CAST(address_normalized AS VARCHAR) AS address_normalized,
               CAST(country_normalized AS VARCHAR) AS country_normalized
        FROM read_parquet('{s1}')
    """)

    con.execute(f"""
        CREATE OR REPLACE VIEW targets AS
        SELECT CAST(entity_id AS VARCHAR) AS candidate_entity_id,
               'S2' AS candidate_source,
               CAST(name_core AS VARCHAR) AS name_core,
               CAST(address_normalized AS VARCHAR) AS address_normalized,
               CAST(country_normalized AS VARCHAR) AS country_normalized
        FROM read_parquet('{s2}')
        UNION ALL
        SELECT CAST(entity_id AS VARCHAR), 'S3',
               CAST(name_core AS VARCHAR),
               CAST(address_normalized AS VARCHAR),
               CAST(country_normalized AS VARCHAR)
        FROM read_parquet('{s3}')
    """)

    con.execute(f"""
        CREATE OR REPLACE VIEW phase2c AS
        SELECT * FROM read_parquet('{phase2c}')
    """)


def create_stop_tokens(con):
    con.execute("CREATE OR REPLACE TEMP TABLE address_stop_tokens(token VARCHAR)")
    values = ",".join(f"('{token}')" for token in sorted(ADDRESS_STOP_TOKENS))
    con.execute(f"INSERT INTO address_stop_tokens VALUES {values}")


def create_address_tokens(con, max_token_frequency):
    print("Extracting and filtering target address tokens...")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE valid_target_address_tokens AS
        WITH raw AS (
            SELECT DISTINCT candidate_entity_id, candidate_source,
                   country_normalized, token
            FROM targets,
            UNNEST(string_split(address_normalized, ' ')) AS u(token)
            WHERE address_normalized IS NOT NULL
              AND address_normalized <> ''
              AND token IS NOT NULL
              AND token <> ''
              AND LENGTH(token) >= 3
              AND token NOT IN (SELECT token FROM address_stop_tokens)
        ), frequencies AS (
            SELECT country_normalized, token, COUNT(*) AS token_count
            FROM raw
            GROUP BY country_normalized, token
            HAVING COUNT(*) <= {max_token_frequency}
        )
        SELECT r.candidate_entity_id, r.candidate_source,
               r.country_normalized, r.token, f.token_count
        FROM raw r
        JOIN frequencies f
          ON r.country_normalized = f.country_normalized
         AND r.token = f.token
    """)

    print("Extracting Source 1 address tokens...")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE source1_address_tokens AS
        SELECT DISTINCT source1_entity_id, country_normalized, token
        FROM source1,
        UNNEST(string_split(address_normalized, ' ')) AS u(token)
        WHERE address_normalized IS NOT NULL
          AND address_normalized <> ''
          AND token IS NOT NULL
          AND token <> ''
          AND LENGTH(token) >= 3
          AND token NOT IN (SELECT token FROM address_stop_tokens)
    """)


def create_address_candidates(con, min_shared_tokens):
    print("Creating address-assisted candidates...")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE address_candidates AS
        SELECT s.source1_entity_id,
               t.candidate_entity_id,
               t.candidate_source,
               COUNT(DISTINCT s.token) AS shared_address_tokens,
               SUM(1.0 / LN(2.0 + t.token_count)) AS address_blocking_score
        FROM source1_address_tokens s
        JOIN valid_target_address_tokens t
          ON s.country_normalized = t.country_normalized
         AND s.token = t.token
        GROUP BY s.source1_entity_id,
                 t.candidate_entity_id,
                 t.candidate_source
        HAVING COUNT(DISTINCT s.token) >= {min_shared_tokens}
    """)


def write_output(con, paths, max_candidates):
    output = paths["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    print("Merging Phase 2C and address candidates...")
    con.execute(f"""
        COPY (
            WITH unioned AS (
                SELECT source1_entity_id, candidate_entity_id, candidate_source,
                       exact_name, exact_core, exact_address,
                       shared_token_count, token_blocking_score,
                       exact_blocking_score, combined_blocking_score,
                       name_fuzzy_similarity,
                       0 AS shared_address_tokens,
                       0.0 AS address_blocking_score
                FROM phase2c
                UNION ALL
                SELECT source1_entity_id, candidate_entity_id, candidate_source,
                       0, 0, 0, 0, 0.0, 0.0,
                       address_blocking_score * 20.0,
                       0.0,
                       shared_address_tokens,
                       address_blocking_score
                FROM address_candidates
            ), aggregated AS (
                SELECT source1_entity_id, candidate_entity_id, candidate_source,
                       MAX(exact_name) AS exact_name,
                       MAX(exact_core) AS exact_core,
                       MAX(exact_address) AS exact_address,
                       MAX(shared_token_count) AS shared_token_count,
                       MAX(token_blocking_score) AS token_blocking_score,
                       MAX(exact_blocking_score) AS exact_blocking_score,
                       MAX(combined_blocking_score) AS combined_blocking_score,
                       MAX(name_fuzzy_similarity) AS name_fuzzy_similarity,
                       MAX(shared_address_tokens) AS shared_address_tokens,
                       MAX(address_blocking_score) AS address_blocking_score
                FROM unioned
                GROUP BY source1_entity_id, candidate_entity_id, candidate_source
            ), ranked AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY source1_entity_id
                    ORDER BY exact_name DESC, exact_address DESC, exact_core DESC,
                             name_fuzzy_similarity DESC,
                             shared_address_tokens DESC,
                             address_blocking_score DESC,
                             combined_blocking_score DESC,
                             candidate_entity_id
                ) AS candidate_rank
                FROM aggregated
            )
            SELECT * FROM ranked
            WHERE candidate_rank <= {max_candidates}
        ) TO '{sql_path(output)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """)


def print_summary(con, output_path):
    result = con.execute(f"""
        WITH counts AS (
            SELECT source1_entity_id, COUNT(*) AS candidate_count
            FROM read_parquet('{sql_path(output_path)}')
            GROUP BY source1_entity_id
        )
        SELECT SUM(candidate_count) AS total_candidate_pairs,
               COUNT(*) AS covered_source1_entities,
               AVG(candidate_count) AS average_candidates,
               MAX(candidate_count) AS maximum_candidates
        FROM counts
    """).fetchdf()
    print("\nPhase 2D summary:")
    print(result.to_string(index=False))
    print(f"\nCreated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2D address-assisted candidate recovery")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--memory-limit", default="6GB")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--temp-limit", default="30GB")
    parser.add_argument("--max-token-frequency", type=int, default=100)
    parser.add_argument("--min-shared-tokens", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=300)
    args = parser.parse_args()

    paths = get_paths(args.split)
    validate_inputs(paths)
    con = connect_db(args.split, args.memory_limit, args.threads, args.temp_limit)
    try:
        create_views(con, paths)
        create_stop_tokens(con)
        create_address_tokens(con, args.max_token_frequency)
        create_address_candidates(con, args.min_shared_tokens)
        write_output(con, paths, args.max_candidates)
        print_summary(con, paths["output"])
    finally:
        con.close()


if __name__ == "__main__":
    main()
