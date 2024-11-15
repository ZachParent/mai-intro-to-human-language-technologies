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
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from nltk.metrics.distance import jaccard_distance
import spacy
import zp_ihlt_project
from zp_ihlt_project.config import TRAIN_DATA_DIR
from zp_ihlt_project.feature_extraction import generate_valid_permutations, apply_steps_and_compare
import importlib

# %%
BASE_PATH = "./"
DEMO_S_IDX = 79
# N_SAMPLES = np.iinfo(np.int32).max
N_SAMPLES = 100

# %%
dt = pd.read_csv(
    f"{TRAIN_DATA_DIR}/STS.input.SMTeuroparl.txt", sep="\t", header=None
)
dt.columns = ["s1", "s2"]
gs = pd.read_csv(f"{TRAIN_DATA_DIR}/STS.gs.SMTeuroparl.txt", sep="\t", header=None)
dt["gs"] = gs[0]
dt = dt.iloc[:N_SAMPLES]
gs = gs.iloc[:N_SAMPLES]

# %%
valid_permutations = generate_valid_permutations()
for i, perm in enumerate(valid_permutations):
    dt[f"score_{i}"] = apply_steps_and_compare(dt.s1, dt.s2, perm)
features = [f"score_{i}" for i in range(len(valid_permutations))]
dt.head()


# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dt[features], dt.gs, test_size=0.2, random_state=42)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
pearsonr(y_test, preds)
