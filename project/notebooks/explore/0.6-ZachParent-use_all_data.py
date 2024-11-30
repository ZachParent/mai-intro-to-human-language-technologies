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
from zp_ihlt_project.load_data import load_train_data
from zp_ihlt_project.feature_extraction import generate_valid_permutations, apply_steps_and_compare

# %%
dt = load_train_data()

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
