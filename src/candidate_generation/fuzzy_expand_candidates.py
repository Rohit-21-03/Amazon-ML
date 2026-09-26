import argparse
from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def sql_path(path):
    return Path(path).resolve().as_posix().replace("'", "''")


def paths_for(split):
    base = PROCESSED_DIR / split
    return {
        "s1": base / f"{split}_source1_normalized.parquet",
        "s2": base / f"{split}_source2_normalized.parquet",
        "s3": base / f"{split}_source3_normalized.parquet",
        "phase2b": base / f"{split}_candidate_pairs_expanded.parquet",
        "output": base / f"{split}_candidate_pairs_fuzzy.parquet",
    }


def validate(paths):
    missing = [str(paths[k]) for k in ("s1", "s2", "s3", "phase2b") if not paths[k].exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def connect(split, memory_limit, threads, temp_limit):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    temp = OUTPUTS_DIR / "duckdb_temp_2c"
    temp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(OUTPUTS_DIR / f"phase2c_{split}.duckdb"))
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET threads = {threads}")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET temp_directory = '{sql_path(temp)}'")
    con.execute(f"SET max_temp_directory_size = '{temp_limit}'")
    con.execute("SET enable_progress_bar = true")
    return con


def create_views(con, p):
    con.execute(f"""
        CREATE OR REPLACE VIEW s1 AS
        SELECT CAST(entity_id AS VARCHAR) source1_entity_id,
               CAST(name_core AS VARCHAR) name_core,
               CAST(country_normalized AS VARCHAR) country_normalized
        FROM read_parquet('{sql_path(p['s1'])}')
        WHERE name_core IS NOT NULL AND LENGTH(name_core) >= 4
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW targets AS
        SELECT CAST(entity_id AS VARCHAR) candidate_entity_id, 'S2' candidate_source,
               CAST(name_core AS VARCHAR) name_core,
               CAST(country_normalized AS VARCHAR) country_normalized
        FROM read_parquet('{sql_path(p['s2'])}')
        WHERE name_core IS NOT NULL AND LENGTH(name_core) >= 4
        UNION ALL
        SELECT CAST(entity_id AS VARCHAR), 'S3', CAST(name_core AS VARCHAR),
               CAST(country_normalized AS VARCHAR)
        FROM read_parquet('{sql_path(p['s3'])}')
        WHERE name_core IS NOT NULL AND LENGTH(name_core) >= 4
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW phase2b AS
        SELECT * FROM read_parquet('{sql_path(p['phase2b'])}')
    """)


def create_fuzzy_candidates(con, max_prefix_frequency, min_similarity):
    print("Creating prefix blocks...")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE valid_target_prefixes AS
        WITH x AS (
            SELECT country_normalized,
                   LEFT(REPLACE(name_core, ' ', ''), 4) prefix4,
                   COUNT(*) n
            FROM targets
            GROUP BY country_normalized, prefix4
        )
        SELECT country_normalized, prefix4
        FROM x
        WHERE LENGTH(prefix4) = 4 AND n <= {max_prefix_frequency}
    """)

    print("Scoring fuzzy name candidates...")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fuzzy_candidates AS
        WITH blocked AS (
            SELECT s.source1_entity_id, t.candidate_entity_id, t.candidate_source,
                   s.name_core source_name, t.name_core candidate_name
            FROM s1 s
            JOIN valid_target_prefixes b
              ON s.country_normalized = b.country_normalized
             AND LEFT(REPLACE(s.name_core, ' ', ''), 4) = b.prefix4
            JOIN targets t
              ON t.country_normalized = b.country_normalized
             AND LEFT(REPLACE(t.name_core, ' ', ''), 4) = b.prefix4
            WHERE ABS(LENGTH(s.name_core) - LENGTH(t.name_core)) <= 8
        )
        SELECT source1_entity_id, candidate_entity_id, candidate_source,
               jaro_winkler_similarity(source_name, candidate_name) name_fuzzy_similarity
        FROM blocked
        WHERE jaro_winkler_similarity(source_name, candidate_name) >= {min_similarity}
    """)


def write_output(con, p, max_candidates):
    out = p["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    print("Merging Phase 2B and fuzzy candidates...")
    con.execute(f"""
        COPY (
          WITH unioned AS (
            SELECT source1_entity_id, candidate_entity_id, candidate_source,
                   exact_name, exact_core, exact_address, shared_token_count,
                   token_blocking_score, exact_blocking_score,
                   combined_blocking_score,
                   0.0 name_fuzzy_similarity
            FROM phase2b
            UNION ALL
            SELECT source1_entity_id, candidate_entity_id, candidate_source,
                   0, 0, 0, 0, 0.0, 0.0,
                   name_fuzzy_similarity * 50.0,
                   name_fuzzy_similarity
            FROM fuzzy_candidates
          ), agg AS (
            SELECT source1_entity_id, candidate_entity_id, candidate_source,
                   MAX(exact_name) exact_name, MAX(exact_core) exact_core,
                   MAX(exact_address) exact_address,
                   MAX(shared_token_count) shared_token_count,
                   MAX(token_blocking_score) token_blocking_score,
                   MAX(exact_blocking_score) exact_blocking_score,
                   MAX(combined_blocking_score) combined_blocking_score,
                   MAX(name_fuzzy_similarity) name_fuzzy_similarity
            FROM unioned
            GROUP BY source1_entity_id, candidate_entity_id, candidate_source
          ), ranked AS (
            SELECT *, ROW_NUMBER() OVER (
              PARTITION BY source1_entity_id
              ORDER BY exact_name DESC, exact_address DESC, exact_core DESC,
                       name_fuzzy_similarity DESC, combined_blocking_score DESC,
                       candidate_entity_id
            ) candidate_rank
            FROM agg
          )
          SELECT * FROM ranked WHERE candidate_rank <= {max_candidates}
        ) TO '{sql_path(out)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """)


def summary(con, p):
    df = con.execute(f"""
      WITH c AS (
        SELECT source1_entity_id, COUNT(*) n
        FROM read_parquet('{sql_path(p['output'])}') GROUP BY 1
      )
      SELECT SUM(n) total_candidate_pairs, COUNT(*) covered_source1_entities,
             AVG(n) average_candidates, MAX(n) maximum_candidates FROM c
    """).fetchdf()
    print("\nPhase 2C summary:")
    print(df.to_string(index=False))
    print(f"\nCreated: {p['output']}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2C fuzzy name candidate recovery")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--memory-limit", default="6GB")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--temp-limit", default="30GB")
    parser.add_argument("--max-prefix-frequency", type=int, default=300)
    parser.add_argument("--min-similarity", type=float, default=0.88)
    parser.add_argument("--max-candidates", type=int, default=250)
    a = parser.parse_args()
    p = paths_for(a.split)
    validate(p)
    con = connect(a.split, a.memory_limit, a.threads, a.temp_limit)
    try:
        create_views(con, p)
        create_fuzzy_candidates(con, a.max_prefix_frequency, a.min_similarity)
        write_output(con, p, a.max_candidates)
        summary(con, p)
    finally:
        con.close()


if __name__ == "__main__":
    main()
