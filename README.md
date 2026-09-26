# Amazon ML Challenge 2026: Business Entity Resolution

## Overview

This repository contains an end-to-end machine learning pipeline for large-scale business entity resolution across three data sources.

The objective is to identify which records in Source 2 and Source 3 refer to the same real-world business as each Source 1 record. The system processes more than 12 million source records, generates a manageable set of candidate pairs, calculates pairwise similarity features, trains a LightGBM classifier, optimizes an entity-level F0.5 threshold, performs test inference, and generates validated submission files.

The project was designed around two constraints:

1. A brute-force comparison between Source 1 and all Source 2 and Source 3 records is computationally infeasible.
2. The F0.5 metric gives more importance to precision than recall, so candidate recall and final prediction precision must be balanced carefully.

---

## Final Pipeline

```text
Raw TSV files
    |
    v
Phase 0: Dataset profiling and ground-truth analysis
    |
    v
Phase 1: Text normalization and Parquet conversion
    |
    v
Phase 2: Multi-stage candidate generation
    |-- 2A Exact blocking
    |-- 2B Rare name-token blocking
    |-- 2C Fuzzy name blocking
    `-- 2D Address-token blocking
    |
    v
Phase 3: Labeled pair creation and feature engineering
    |
    v
Phase 4: LightGBM training and threshold optimization
    |
    v
Phase 5: Test candidate generation, features, and inference
    |
    v
Phase 6: Submission generation
    |
    v
Phase 7: Memory-safe structural validation
```

---

## Dataset Summary

### Training data

| Source | Rows |
|---|---:|
| Source 1 | 2,206,821 |
| Source 2 | 5,034,616 |
| Source 3 | 5,285,603 |
| Ground-truth rows | 2,206,821 |

Total source records exceed 12.5 million.

### Test data

The final test Source 1 file contains:

```text
1,732,544 records
```

### Missing-value observations

```text
Source 1 missing names:        0
Source 1 missing addresses:    0
Source 2 missing names:        2
Source 2 missing addresses:    168,967
Source 3 missing names:        13
Source 3 missing addresses:    175,916
```

### Training ground-truth distribution

```text
Total Source 1 entities:  2,206,821
No matches:                 123,247
Exactly 1 match:            119,157
Exactly 2 matches:          375,212
3 or more matches:        1,589,205
Average matches/entity:         3.46
Maximum matches/entity:           11
```

Only about 5.58% of training Source 1 entities are singletons. Most entities have multiple true links, so candidate-generation recall is critical.

---

## Why Candidate Generation Is Required

A direct comparison would require approximately:

```text
2.2 million Source 1 records
x
10.3 million Source 2 and Source 3 records
=
more than 22 trillion pair comparisons
```

This is impractical. The pipeline therefore uses blocking and retrieval rules to produce a smaller candidate set before applying the classifier.

A true match omitted during candidate generation cannot be recovered by the classification model. Candidate recall is therefore the upper bound of the matching system's link recall.

---

## Repository Structure

```text
amazon-ml-2026/
|
|-- data/
|   |-- raw/
|   |   |-- train/
|   |   |   |-- train_source1.tsv
|   |   |   |-- train_source2.tsv
|   |   |   |-- train_source3.tsv
|   |   |   `-- train_ground_truth.tsv
|   |   `-- test/
|   |       |-- test_source1.tsv
|   |       |-- test_source2.tsv
|   |       `-- test_source3.tsv
|   |
|   `-- processed/
|       |-- train/
|       |   |-- train_source1_normalized.parquet
|       |   |-- train_source2_normalized.parquet
|       |   |-- train_source3_normalized.parquet
|       |   |-- train_ground_truth_links.parquet
|       |   |-- train_candidate_pairs.parquet
|       |   |-- train_candidate_pairs_expanded.parquet
|       |   |-- train_candidate_pairs_fuzzy.parquet
|       |   |-- train_candidate_pairs_address.parquet
|       |   |-- train_labeled_pairs.parquet
|       |   |-- train_pair_features.parquet
|       |   `-- validation_scored_pairs.parquet
|       `-- test/
|           |-- test_source1_normalized.parquet
|           |-- test_source2_normalized.parquet
|           |-- test_source3_normalized.parquet
|           |-- test_candidate_pairs.parquet
|           |-- test_candidate_pairs_expanded.parquet
|           |-- test_candidate_pairs_fuzzy.parquet
|           |-- test_candidate_pairs_address.parquet
|           |-- test_pair_features.parquet
|           `-- test_scored_pairs.parquet
|
|-- models/
|   |-- lightgbm_baseline.txt
|   |-- feature_columns.json
|   `-- best_threshold.json
|
|-- reports/
|   |-- phase1_validation.csv
|   |-- phase2A_exact_candidate_recall.csv
|   |-- phase2B_token_candidate_recall.csv
|   |-- phase2C_fuzzy_candidate_recall.csv
|   |-- phase2D_address_candidate_recall.csv
|   |-- phase3_feature_summary.csv
|   |-- phase4A_pair_metrics.json
|   |-- phase4A_feature_importance.csv
|   |-- phase4B_threshold_search.csv
|   |-- phase4B_validation_metrics.json
|   |-- phase6_submission_validation.json
|   `-- final_submission_validation.json
|
|-- src/
|   |-- analysis/
|   |-- profiling/
|   |-- utils/
|   |   |-- data_loader.py
|   |   `-- validate_submission_batched.py
|   |-- preprocessing/
|   |   |-- text_normalizer.py
|   |   |-- preprocess_sources.py
|   |   `-- validate_preprocessing.py
|   |-- candidate_generation/
|   |   |-- generate_candidates.py
|   |   |-- prepare_ground_truth.py
|   |   |-- evaluate_candidate_recall.py
|   |   |-- expand_candidates.py
|   |   |-- fuzzy_expand_candidates.py
|   |   `-- address_expand_candidates.py
|   |-- feature_engineering/
|   |   |-- build_training_pairs.py
|   |   |-- build_pair_features.py
|   |   `-- build_test_pair_features.py
|   |-- modeling/
|   |   |-- train_lightgbm.py
|   |   |-- optimize_threshold.py
|   |   `-- score_test_pairs.py
|   `-- submission/
|       |-- generate_submission.py
|       `-- generate_submission_batched.py
|
|-- submission/
|   |-- matching_results.tsv
|   `-- candidate_pairs.tsv
|
|-- outputs/
|-- notebooks/
|-- requirements.txt
`-- README.md
```

All Python package folders should contain an empty `__init__.py` file.

---

## Dependencies

The implementation uses:

```txt
pandas
numpy
pyarrow
duckdb
lightgbm
tqdm
matplotlib
seaborn
```

Install dependencies from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

A minimal `requirements.txt` is:

```txt
pandas
numpy
pyarrow
duckdb
lightgbm
tqdm
matplotlib
seaborn
```

Python, package, LightGBM, and DuckDB versions should be recorded before final packaging:

```powershell
python --version
pip freeze > environment-lock.txt
python -c "import lightgbm, duckdb; print('LightGBM:', lightgbm.__version__); print('DuckDB:', duckdb.__version__)"
```

---

# Detailed Methodology

## Phase 0: Data Understanding

### Purpose

Phase 0 profiles the data before modeling. It establishes dataset scale, missing values, duplicate behavior, country distribution, ground-truth match counts, and common name and address tokens.

### Files

```text
src/utils/data_loader.py
src/profiling/source_profiler.py
src/analysis/ground_truth_analysis.py
src/analysis/country_analysis.py
src/analysis/token_analysis.py
```

### Analyses performed

- Row counts
- Unique entity IDs
- Duplicate entity IDs
- Missing business names
- Missing business addresses
- Missing countries
- Duplicate names and addresses
- Country distribution
- Match-count distribution
- Singleton frequency
- Common business-name tokens
- Common address tokens
- Manual ground-truth inspection

### Important findings

Frequently occurring legal-name tokens included:

```text
limited
private
llc
inc
ltd
pvt
corp
co
and
&
```

Frequently occurring address words included:

```text
road
street
unit
drive
avenue
```

Country was selected as a safe initial blocking attribute because comparisons should remain within the same normalized country.

---

## Phase 1: Text Normalization

### Purpose

Phase 1 normalizes business names, business addresses, and countries while retaining all original fields. Raw TSV files are converted to compressed Parquet files for efficient downstream processing.

### Files

```text
src/preprocessing/text_normalizer.py
src/preprocessing/preprocess_sources.py
src/preprocessing/validate_preprocessing.py
```

### Name normalization

The normalizer performs:

1. Unicode NFKC normalization
2. Whitespace trimming
3. Case folding
4. `&` replacement with `and`
5. Punctuation removal
6. Repeated-whitespace reduction
7. Legal-suffix removal in a separate core-name representation

Two name fields are retained:

```text
name_normalized
name_core
```

Example:

```text
Original:         Custom Wealth Services LLC
name_normalized:  custom wealth services llc
name_core:        custom wealth services
```

Legal suffixes removed from `name_core` include:

```text
ag, co, company, corp, corporation, gmbh, inc, incorporated,
limited, llc, llp, lp, ltd, pc, plc, private, pvt, sa, sarl, sas
```

The complete normalized name is preserved because legal terms can sometimes help distinguish entities.

### Address normalization

Address terms are standardized, including:

```text
apartment -> apt
avenue    -> ave
building  -> bldg
boulevard -> blvd
drive     -> dr
floor     -> fl
highway   -> hwy
lane      -> ln
mount     -> mt
road      -> rd
saint     -> st
street    -> st
suite     -> ste
```

### Generated columns

```text
entity_id
business_name
business_address
country
name_normalized
name_core
address_normalized
country_normalized
name_missing
address_missing
name_length
address_length
```

### Processing method

Files are read in chunks and written as compressed Parquet to avoid loading the complete multi-million-row dataset into memory.

### Commands

```powershell
python -m src.preprocessing.preprocess_sources --split train --chunk-size 250000
python -m src.preprocessing.preprocess_sources --split test --chunk-size 250000
python -m src.preprocessing.validate_preprocessing
```

---

## Phase 2: Candidate Generation

Candidate generation is implemented as four cumulative stages. Each stage preserves candidates from the previous stage and adds another recovery mechanism.

DuckDB is used to scan and write Parquet files, perform joins, aggregate blocking signals, and spill larger-than-memory operations to disk.

## Phase 2A: Exact Blocking

### Rules

Candidate pairs are created within the same country using:

```text
Exact normalized business name
Exact core business name
Exact normalized address
```

Very frequent blocks are excluded to prevent candidate explosion.

### Blocking signals

```text
exact_name
exact_core
exact_address
blocking_score
candidate_rank
```

### Parameters used

```text
Maximum block size:       200
Maximum candidates/S1:   250
```

### Training candidate recall

```text
Overall: 49.1556%
S2:      53.4048%
S3:      45.1770%
Entity candidate coverage: 93.8106%
```

### Commands

```powershell
python -m src.candidate_generation.generate_candidates --split train --memory-limit 6GB --threads 2 --max-block-size 200 --max-candidates 250
python -m src.candidate_generation.prepare_ground_truth
python -m src.candidate_generation.evaluate_candidate_recall
```

## Phase 2B: Rare Name-Token Blocking

### Method

Business core names are split into tokens. Common and low-information tokens are removed. Source 1 and target records are joined within the same country when they share a sufficiently rare name token.

Example stop tokens:

```text
a, an, and, co, company, corp, corporation, for, in, inc,
limited, llc, llp, ltd, of, private, pvt, service, services, the, to
```

### Features added

```text
shared_token_count
token_blocking_score
```

The token blocking score gives greater weight to rarer tokens.

### Parameters used

```text
Maximum token frequency: 100
Maximum candidates/S1:   200
```

### Training candidate recall

```text
Overall: 56.4211%
S2:      59.9371%
S3:      53.1289%
Entity candidate coverage: 95.4943%
```

### Command

```powershell
python -m src.candidate_generation.expand_candidates --split train --memory-limit 6GB --threads 2 --temp-limit 30GB --max-token-frequency 100 --max-candidates 200
```

## Phase 2C: Fuzzy Name Blocking

### Method

Fuzzy candidate recovery is restricted before similarity calculation to prevent all-to-all comparisons.

Records must satisfy:

```text
Same normalized country
Same four-character compact-name prefix
Prefix block frequency <= 300
Absolute name-length difference <= 8
Jaro-Winkler similarity >= 0.88
```

### Feature added

```text
name_fuzzy_similarity
```

### Parameters used

```text
Maximum prefix frequency: 300
Minimum Jaro-Winkler similarity: 0.88
Maximum candidates/S1: 250
```

### Training candidate recall

```text
Overall: 59.5518%
S2:      62.7996%
S3:      56.5109%
Entity candidate coverage: 96.2419%
```

### Command

```powershell
python -m src.candidate_generation.fuzzy_expand_candidates --split train --memory-limit 6GB --threads 2 --temp-limit 30GB --max-prefix-frequency 300 --min-similarity 0.88 --max-candidates 250
```

## Phase 2D: Address-Token Blocking

### Method

Normalized addresses are split into tokens. Generic address words are removed, and low-frequency address tokens are used for same-country candidate recovery.

Example address stop tokens:

```text
and, apt, ave, bldg, blvd, building, dr, drive, fl, floor,
hwy, in, lane, ln, main, near, of, rd, road, st, ste,
street, suite, the, to, unit
```

### Features added

```text
shared_address_tokens
address_blocking_score
```

### Parameters used

```text
Maximum address-token frequency: 100
Minimum shared address tokens:   1
Maximum candidates/S1:           300
```

### Training candidate recall

```text
Overall: 72.6667%
S2:      75.1403%
S3:      70.3507%
Entity candidate coverage: 97.6465%
```

Phase 2D recovered 5,550,550 of 7,638,365 training ground-truth links represented in the evaluation.

### Command

```powershell
python -m src.candidate_generation.address_expand_candidates --split train --memory-limit 6GB --threads 2 --temp-limit 30GB --max-token-frequency 100 --min-shared-tokens 1 --max-candidates 300
```

The Phase 2D output is frozen as the baseline modeling candidate set:

```text
data/processed/train/train_candidate_pairs_address.parquet
```

---

## Phase 3: Labeled Pairs and Feature Engineering

## Phase 3A: Labeled Training Pairs

### Labeling

Each candidate pair is compared against the exploded ground-truth link table:

```text
label = 1 if the pair exists in ground truth
label = 0 otherwise
```

### Negative sampling

The complete candidate set can contain hundreds of millions of negative pairs. The initial training dataset therefore retains:

```text
All recovered positive pairs
10 hard negatives per Source 1 entity with true matches
5 negatives per singleton Source 1 entity
```

Hard negatives are ranked using exact-match flags, fuzzy-name similarity, address-token overlap, blocking scores, and candidate rank.

### Train/validation split

The split is deterministic and performed by Source 1 entity ID:

```text
80% train
20% validation
```

All candidate pairs belonging to a Source 1 entity remain in the same split, preventing entity leakage.

### Command

```powershell
python -m src.feature_engineering.build_training_pairs --memory-limit 6GB --threads 2 --temp-limit 30GB --negatives-per-matched 10 --negatives-per-singleton 5
```

## Phase 3B: Pairwise Features

The feature builder joins selected candidate pairs to normalized Source 1, Source 2, and Source 3 records.

### Identity and control columns

```text
source1_entity_id
candidate_entity_id
candidate_source
label
dataset_split
entity_has_true_match
```

### Candidate-generation features

```text
candidate_rank
candidate_is_s2
exact_name
exact_core
exact_address
shared_token_count
shared_address_tokens
token_blocking_score
address_blocking_score
exact_blocking_score
combined_blocking_score
name_fuzzy_similarity
```

### Missingness and country features

```text
country_equal
s1_name_missing
target_name_missing
s1_address_missing
target_address_missing
```

### Jaro-Winkler features

```text
name_jaro_winkler
core_jaro_winkler
address_jaro_winkler
```

### Levenshtein distance features

```text
name_levenshtein_distance
core_levenshtein_distance
address_levenshtein_distance
```

### Normalized Levenshtein similarity features

```text
name_levenshtein_similarity
core_levenshtein_similarity
address_levenshtein_similarity
```

Each similarity is calculated as:

```text
1 - edit_distance / maximum_string_length
```

### Length-difference features

```text
name_length_difference
core_length_difference
address_length_difference
```

### Containment features

```text
core_contains
address_contains
```

These indicate whether one normalized string contains the other.

### Commands

```powershell
python -m src.feature_engineering.build_pair_features --memory-limit 6GB --threads 2 --temp-limit 30GB
```

### Validation checks

```text
Missing labels: 0
Duplicate candidate pairs: 0
No Source 1 entity leakage across train and validation
```

---

## Phase 4: Matching Model

## Model

The baseline classifier is LightGBM with a binary objective.

LightGBM was selected because it:

- Handles nonlinear interactions between similarity features
- Trains efficiently on millions of rows
- Supports class weighting
- Provides feature importance
- Produces probabilities for threshold optimization
- Saves a compact model artifact

## Training configuration

```text
Objective:              binary
Metrics:                binary log loss, AUC
Learning rate:          0.04
Number of leaves:       63
Minimum data in leaf:   100
Feature fraction:       0.90
Bagging fraction:       0.90
Bagging frequency:      1
L1 regularization:      0.1
L2 regularization:      1.0
Maximum boosting rounds: 1200
Early-stopping patience: 100
Seed:                   42
```

Positive-class weighting is calculated from the training split:

```text
scale_pos_weight = negative_pairs / positive_pairs
```

### Command

```powershell
python -m src.modeling.train_lightgbm --threads 4 --num-boost-round 1200 --early-stopping-rounds 100
```

## Phase 4A validation at threshold 0.5

```text
True positives:   1,076,726
False positives:     97,512
False negatives:     33,426
True negatives:   2,899,382
Precision:          0.91696
Recall:             0.96989
Pairwise F0.5:      0.92708
Best iteration:        1200
```

The pairwise F0.5 is a diagnostic metric. The competition-oriented threshold selection uses entity-level macro F0.5.

## Feature importance observations

The strongest baseline features included:

```text
address_jaro_winkler
candidate_rank
address_levenshtein_similarity
core_levenshtein_similarity
core_jaro_winkler
address_levenshtein_distance
candidate_is_s2
combined_blocking_score
name_jaro_winkler
address_length_difference
```

Address similarity was the strongest signal in the first trained baseline.

## Phase 4B: Threshold Optimization

The default probability threshold of `0.5` is not assumed to be optimal.

Thresholds are evaluated from:

```text
0.300 to 0.950 in increments of 0.025
```

For each threshold:

1. Candidate probabilities are converted into predicted links.
2. Predictions are grouped by Source 1 entity.
3. True positives, false positives, and false negatives are calculated per entity.
4. Entity-level F0.5 is calculated.
5. Scores are averaged across all validation Source 1 entities.
6. A true singleton predicted as empty receives a score of 1.

The F0.5 formula is:

```text
F0.5 = 1.25 * TP / (1.25 * TP + FP + 0.25 * FN)
```

The selected threshold is stored in:

```text
models/best_threshold.json
```

### Command

```powershell
python -m src.modeling.optimize_threshold --memory-limit 6GB --threads 2 --temp-limit 30GB --start 0.30 --stop 0.95 --step 0.025
```

---

## Phase 5: Test Inference

The complete candidate-generation pipeline is rerun on test data using the same settings as training.

### Phase 5A: Exact candidates

```powershell
python -m src.candidate_generation.generate_candidates --split test --memory-limit 6GB --threads 2 --max-block-size 200 --max-candidates 250
```

### Phase 5B: Name-token expansion

```powershell
python -m src.candidate_generation.expand_candidates --split test --memory-limit 6GB --threads 2 --temp-limit 30GB --max-token-frequency 100 --max-candidates 200
```

### Phase 5C: Fuzzy-name expansion

```powershell
python -m src.candidate_generation.fuzzy_expand_candidates --split test --memory-limit 6GB --threads 2 --temp-limit 30GB --max-prefix-frequency 300 --min-similarity 0.88 --max-candidates 250
```

### Phase 5D: Address-token expansion

```powershell
python -m src.candidate_generation.address_expand_candidates --split test --memory-limit 6GB --threads 2 --temp-limit 30GB --max-token-frequency 100 --min-shared-tokens 1 --max-candidates 300
```

### Phase 5E: Test pair features

The exact 31 model features and their order are verified against:

```text
models/feature_columns.json
```

```powershell
python -m src.feature_engineering.build_test_pair_features --memory-limit 6GB --threads 2 --temp-limit 30GB
```

Required checks:

```text
Candidate rows = feature rows
Duplicate pairs = 0
All 31 model features are present
```

### Phase 5F: Test scoring

The saved LightGBM model and optimized threshold are loaded. The test feature Parquet is scored in batches to control memory use.

```powershell
python -m src.modeling.score_test_pairs --batch-size 250000 --threads 4
```

Output:

```text
data/processed/test/test_scored_pairs.parquet
```

Columns:

```text
source1_entity_id
candidate_entity_id
candidate_source
match_probability
predicted_match
```

---

## Phase 6: Submission Generation

The test candidate set and accepted model predictions are converted into:

```text
submission/matching_results.tsv
submission/candidate_pairs.tsv
```

### `matching_results.tsv`

Contains exactly one row per test Source 1 entity:

```text
source1_entity_id    matched_entity_ids
```

`matched_entity_ids` is a comma-separated list. Empty predictions are represented by an empty value.

### `candidate_pairs.tsv`

Contains one candidate pair per row:

```text
source1_entity_id    candidate_entity_id
```

Every predicted link in `matching_results.tsv` must exist in `candidate_pairs.tsv`.

### Memory-safe generation

The first global aggregation required too much memory. The final exporter partitions Source 1 IDs into 128 hash buckets and creates `matching_results.tsv` one bucket at a time.

```powershell
python -m src.submission.generate_submission_batched --memory-limit 4GB --threads 1 --temp-limit 30GB --buckets 128
```

### Final generated counts

```text
Test Source 1 rows:          1,732,544
Candidate pairs:            85,192,266
Accepted predicted links:   65,755,368
Matching-result rows:        1,732,544
Predictions outside candidates:       0
```

### Current baseline warning

The accepted prediction count corresponds to approximately 37.96 predicted links per test Source 1 entity, while the training ground-truth average was approximately 3.46. The generated files are structurally valid, but this volume suggests that further precision-oriented tuning may improve the leaderboard score.

---

## Phase 7: Structural Validation

The original global validator exceeded memory while expanding tens of millions of predictions. A memory-safe batched validator was therefore implemented.

The validator:

1. Confirms exactly one result row per test Source 1 ID.
2. Confirms no duplicate Source 1 result rows.
3. Confirms candidate TSV row count matches the frozen candidate Parquet.
4. Streams the matching file into smaller buckets.
5. Checks duplicate predicted links.
6. Confirms every predicted link exists in the candidate set.
7. Writes a final JSON validation report.

### Command

```powershell
python -m src.utils.validate_submission_batched `
  --matching .\submission\matching_results.tsv `
  --candidate .\submission\candidate_pairs.tsv `
  --test-dir .\data\raw\test `
  --memory-limit 4GB `
  --threads 1 `
  --temp-limit 30GB `
  --buckets 64 `
  --rebuild-candidate-buckets
```

### Final validation result

```text
Source 1 rows:                   1,732,544
Distinct Source 1 IDs:          1,732,544
Matching rows:                  1,732,544
Distinct matching IDs:          1,732,544
Duplicate matching Source rows:         0
Predicted links:               65,755,368
Candidate TSV rows:            85,192,266
Candidate Parquet rows:        85,192,266
Duplicate predicted links:              0
Predictions outside candidates:          0
Status: PASS
```

Final report:

```text
reports/final_submission_validation.json
```

---

# End-to-End Reproduction Order

Run commands from the project root.

## 1. Install dependencies

```powershell
pip install -r requirements.txt
```

## 2. Normalize train and test data

```powershell
python -m src.preprocessing.preprocess_sources --split train --chunk-size 250000
python -m src.preprocessing.preprocess_sources --split test --chunk-size 250000
python -m src.preprocessing.validate_preprocessing
```

## 3. Generate and evaluate training candidates

```powershell
python -m src.candidate_generation.generate_candidates --split train --memory-limit 6GB --threads 2 --max-block-size 200 --max-candidates 250
python -m src.candidate_generation.prepare_ground_truth
python -m src.candidate_generation.evaluate_candidate_recall

python -m src.candidate_generation.expand_candidates --split train --memory-limit 6GB --threads 2 --temp-limit 30GB --max-token-frequency 100 --max-candidates 200

python -m src.candidate_generation.fuzzy_expand_candidates --split train --memory-limit 6GB --threads 2 --temp-limit 30GB --max-prefix-frequency 300 --min-similarity 0.88 --max-candidates 250

python -m src.candidate_generation.address_expand_candidates --split train --memory-limit 6GB --threads 2 --temp-limit 30GB --max-token-frequency 100 --min-shared-tokens 1 --max-candidates 300
```

Update `CANDIDATE_PATH` in `evaluate_candidate_recall.py` to the stage being evaluated before each recall evaluation.

## 4. Build training pairs and features

```powershell
python -m src.feature_engineering.build_training_pairs --memory-limit 6GB --threads 2 --temp-limit 30GB --negatives-per-matched 10 --negatives-per-singleton 5

python -m src.feature_engineering.build_pair_features --memory-limit 6GB --threads 2 --temp-limit 30GB
```

## 5. Train model and optimize threshold

```powershell
python -m src.modeling.train_lightgbm --threads 4 --num-boost-round 1200 --early-stopping-rounds 100

python -m src.modeling.optimize_threshold --memory-limit 6GB --threads 2 --temp-limit 30GB --start 0.30 --stop 0.95 --step 0.025
```

## 6. Generate test candidates

```powershell
python -m src.candidate_generation.generate_candidates --split test --memory-limit 6GB --threads 2 --max-block-size 200 --max-candidates 250

python -m src.candidate_generation.expand_candidates --split test --memory-limit 6GB --threads 2 --temp-limit 30GB --max-token-frequency 100 --max-candidates 200

python -m src.candidate_generation.fuzzy_expand_candidates --split test --memory-limit 6GB --threads 2 --temp-limit 30GB --max-prefix-frequency 300 --min-similarity 0.88 --max-candidates 250

python -m src.candidate_generation.address_expand_candidates --split test --memory-limit 6GB --threads 2 --temp-limit 30GB --max-token-frequency 100 --min-shared-tokens 1 --max-candidates 300
```

## 7. Build test features and score candidates

```powershell
python -m src.feature_engineering.build_test_pair_features --memory-limit 6GB --threads 2 --temp-limit 30GB

python -m src.modeling.score_test_pairs --batch-size 250000 --threads 4
```

## 8. Generate submission files

```powershell
python -m src.submission.generate_submission_batched --memory-limit 4GB --threads 1 --temp-limit 30GB --buckets 128
```

## 9. Validate final files

```powershell
python -m src.utils.validate_submission_batched `
  --matching .\submission\matching_results.tsv `
  --candidate .\submission\candidate_pairs.tsv `
  --test-dir .\data\raw\test `
  --memory-limit 4GB `
  --threads 1 `
  --temp-limit 30GB `
  --buckets 64 `
  --rebuild-candidate-buckets
```

Required final output:

```text
PASS
```

---

## Memory and Disk Management

The dataset and candidate sets are large. The implementation uses:

- Chunked TSV preprocessing
- Parquet compression
- DuckDB projection and streaming
- Explicit memory limits
- Disk spilling through `temp_directory`
- Reduced thread counts for memory-heavy operations
- Hash-bucketed generation and validation
- Batched LightGBM inference

Important distinction:

```text
memory_limit             controls DuckDB-managed RAM usage
max_temp_directory_size  controls temporary disk-spill usage
```

Temporary files can be removed after successful stages, but frozen Parquet files, model files, reports, and submission artifacts should be preserved.

---

## Key Artifacts to Preserve

```text
models/lightgbm_baseline.txt
models/feature_columns.json
models/best_threshold.json

data/processed/train/train_candidate_pairs_address.parquet
data/processed/train/train_pair_features.parquet
data/processed/test/test_candidate_pairs_address.parquet
data/processed/test/test_pair_features.parquet
data/processed/test/test_scored_pairs.parquet

submission/matching_results.tsv
submission/candidate_pairs.tsv

reports/final_submission_validation.json
requirements.txt
environment-lock.txt
README.md
```

---

## Submission Notes

For the leaderboard submission, use:

```text
submission/matching_results.tsv
```

Retain the following for reproducibility and final review:

```text
submission/candidate_pairs.tsv
src/
models/
reports/
requirements.txt
environment-lock.txt
README.md
```

Do not open the complete TSV outputs in spreadsheet software because the files contain millions of rows.

---

## Known Limitations

1. Phase 2D candidate recall is approximately 72.67%, so candidate generation still misses some true links.
2. The model was trained on sampled hard negatives rather than every negative candidate pair.
3. France is not represented in training but may appear in test, so generalization relies on text similarities rather than learned country identity.
4. The first baseline predicts substantially more links per entity than observed in training ground truth.
5. The `candidate_pairs.tsv` file is large because up to 300 candidates are retained per covered Source 1 record.
6. No transformer embeddings or semantic retrieval models are used in the baseline.

---

## Recommended Next Improvements

### Precision improvements

- Tune a stricter threshold on full validation candidate distributions.
- Add a per-entity maximum predicted-match count.
- Tune separate thresholds for Source 2 and Source 3.
- Tune thresholds by missing-address state.
- Calibrate LightGBM probabilities.
- Add an explicit abstention rule for low-confidence entities.

### Candidate recall improvements

- Character n-gram TF-IDF retrieval
- Postal-code extraction and blocking
- Numeric address-token matching
- Phonetic name features
- Abbreviation dictionaries derived from training links
- Separate retrieval indexes by country and target source
- Approximate nearest-neighbor retrieval on lexical or semantic embeddings

### Model improvements

- Retrain using a broader set of hard negatives from the full inference candidate distribution.
- Compare LightGBM with XGBoost or CatBoost.
- Add calibrated model ensembles.
- Add entity-level and source-level aggregate features.
- Add top-1 versus top-2 probability margins.

---

## Reproducibility Checklist

Before sharing or evaluation, verify:

```text
[ ] Raw file names and folder locations match this README
[ ] requirements.txt is present
[ ] environment-lock.txt is generated
[ ] Every src package contains __init__.py
[ ] Phase 1 normalized Parquet files exist
[ ] Final Phase 2D candidate files exist
[ ] LightGBM model and feature list exist
[ ] Optimized threshold exists
[ ] Final test scored-pair file exists
[ ] matching_results.tsv exists
[ ] candidate_pairs.tsv exists
[ ] Final validator returns PASS
[ ] final_submission_validation.json is preserved
```

---

## Final Status

```text
End-to-end pipeline:        Complete
Model training:             Complete
Threshold optimization:     Complete
Test inference:             Complete
Submission generation:      Complete
Structural validation:      PASS
Leaderboard optimization:   Baseline complete; further tuning possible
```

This repository represents a complete, reproducible baseline for large-scale business entity resolution using normalized lexical features, multi-stage blocking, LightGBM classification, entity-level F0.5 threshold selection, and memory-safe inference and validation.
