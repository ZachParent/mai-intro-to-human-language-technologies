import pandas as pd
from pathlib import Path
from typing import List
from zp_ihlt_project.config import (
    INPUT_FILENAME_PATTERN,
    GS_FILENAME_PATTERN,
    TRAIN_DATA_DIR,
    TEST_DATA_DIR,
)


def read_data_file(filepath: str) -> pd.DataFrame:
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        data = [line.strip().split("\t") for line in lines]
        dt = pd.DataFrame(data, columns=["s1", "s2"])
    return dt


def load_all_data_files(data_dir: Path) -> List[pd.DataFrame]:
    input_filenames = sorted(data_dir.glob(INPUT_FILENAME_PATTERN.format("*")))
    gs_filenames = sorted(data_dir.glob(GS_FILENAME_PATTERN.format("*")))
    gs_filenames = [f for f in gs_filenames if "ALL" not in f.name]

    dfs = []
    for input_filename, gs_filename in zip(input_filenames, gs_filenames):
        curr_df = read_data_file(input_filename)
        curr_gs = pd.read_csv(gs_filename, sep="\t", header=None)
        curr_df["gs"] = curr_gs[0]
        curr_df["dataset"] = input_filename.name.split(".")[-2]
        dfs.append(curr_df)
    return dfs


def load_all_data(data_dir: Path) -> pd.DataFrame:
    dfs = load_all_data_files(data_dir)
    return pd.concat(dfs, ignore_index=True)


def load_train_data() -> pd.DataFrame:
    return load_all_data(TRAIN_DATA_DIR)


def load_test_data() -> pd.DataFrame:
    return load_all_data(TEST_DATA_DIR)
