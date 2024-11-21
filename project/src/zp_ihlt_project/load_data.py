import pandas as pd
from pathlib import Path
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


def load_all_data(data_dir: Path) -> pd.DataFrame:
    input_filenames = sorted(data_dir.glob(INPUT_FILENAME_PATTERN.format("*")))
    gs_filenames = sorted(data_dir.glob(GS_FILENAME_PATTERN.format("*")))
    gs_filenames = [f for f in gs_filenames if "ALL" not in f.name]

    dfs = []
    gs_dfs = []
    for input_filename, gs_filename in zip(input_filenames, gs_filenames):
        print(f"Processing {input_filename}")

        curr_df = read_data_file(input_filename)
        curr_df.columns = ["s1", "s2"]  # Set column names

        curr_gs = pd.read_csv(gs_filename, sep="\t", header=None)

        dfs.append(curr_df)
        gs_dfs.append(curr_gs)

    dt = pd.concat(dfs, ignore_index=True)
    gs = pd.concat(gs_dfs, ignore_index=True)
    dt["gs"] = gs[0]
    return dt


def load_train_data() -> pd.DataFrame:
    return load_all_data(TRAIN_DATA_DIR)


def load_test_data() -> pd.DataFrame:
    return load_all_data(TEST_DATA_DIR)
