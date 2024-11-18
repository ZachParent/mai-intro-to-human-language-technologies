# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv-ihlt
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from zp_ihlt_project.config import TRAIN_DATA_DIR
from zp_ihlt_project.feature_extraction import generate_valid_permutations, apply_steps_and_compare


# %%
# Simple file reader
def read_data_file(filepath: str) -> pd.DataFrame:
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        data = [line.strip().split("\t") for line in lines]
        dt = pd.DataFrame(data, columns=["s1", "s2"])
    return dt


# %%
INPUT_FILENAME_PATTERN = "STS.input.{}.txt"
GS_FILENAME_PATTERN = "STS.gs.{}.txt"

input_filenames = sorted(TRAIN_DATA_DIR.glob(INPUT_FILENAME_PATTERN.format("*")))
gs_filenames = sorted(TRAIN_DATA_DIR.glob(GS_FILENAME_PATTERN.format("*")))
print(input_filenames)
print(gs_filenames)

dfs = []
gs_dfs = []
for input_filename, gs_filename in zip(input_filenames, gs_filenames):
    print(f"Processing {input_filename}")
    
    # Read the current file
    curr_df = read_data_file(input_filename)
    curr_df.columns = ["s1", "s2"]  # Set column names
    
    # Read corresponding gold standard
    curr_gs = pd.read_csv(gs_filename, sep="\t", header=None)
    
    # Add to our lists
    dfs.append(curr_df)
    gs_dfs.append(curr_gs)

# Concatenate all DataFrames at once
dt = pd.concat(dfs, ignore_index=True)
gs = pd.concat(gs_dfs, ignore_index=True)
dt["gs"] = gs[0]

# %%
print(dt.count())
dt.head()


# %%
valid_permutations = generate_valid_permutations()
features = []
feature_steps = []
for i, perm in enumerate(valid_permutations):
    features.append(f"score_{i}")
    feature_steps.append(perm)
    dt[f"score_{i}"] = apply_steps_and_compare(dt.s1, dt.s2, perm)
dt.head()


# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    dt[features], dt.gs, test_size=0.2, random_state=42
)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
pearsonr(y_test, preds)

# %% [markdown]
# `PearsonRResult(statistic=0.7804839292242243, pvalue=8.04461291278962e-93)`
#
# Achieves 0.78 on the train set!
