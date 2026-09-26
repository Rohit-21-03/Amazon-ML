from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]

TRAIN_DIR = ROOT_DIR / "data" / "raw" / "train"
TEST_DIR = ROOT_DIR / "data" / "raw" / "test"


def load_source1(train=True):
    path = (
        TRAIN_DIR / "train_source1.tsv"
        if train
        else TEST_DIR / "test_source1.tsv"
    )

    return pd.read_csv(path, sep="\t")


def load_source2(train=True):
    path = (
        TRAIN_DIR / "train_source2.tsv"
        if train
        else TEST_DIR / "test_source2.tsv"
    )

    return pd.read_csv(path, sep="\t")


def load_source3(train=True):
    path = (
        TRAIN_DIR / "train_source3.tsv"
        if train
        else TEST_DIR / "test_source3.tsv"
    )

    return pd.read_csv(path, sep="\t")


def load_ground_truth():
    return pd.read_csv(
        TRAIN_DIR / "train_ground_truth.tsv",
        sep="\t"
    )