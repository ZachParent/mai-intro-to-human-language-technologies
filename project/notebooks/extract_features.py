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

from zp_ihlt_project.feature_extraction import generate_valid_permutations, apply_steps_and_compare
from zp_ihlt_project.load_data import load_train_data, load_test_data


# %%
all_train_dt = load_train_data()
all_test_dt = load_test_data()

# %%
valid_permutations = generate_valid_permutations()


# %%
metrics = ['jaccard', 'cosine', 'euclidean', 'manhattan']
feature_names = []
feature_steps = []
features = []
test_features = []
print(f"Generating {len(valid_permutations) * len(metrics)} features ({len(valid_permutations)} permutations * {len(metrics)} metrics)")
for i, perm in enumerate(valid_permutations):
    print(f"Generating features for permutation {i+1} of {len(valid_permutations)}")
    for metric in metrics:
        feature_names.append(f"score_{metric}_{i}")
        feature_steps.append(perm)
        features.append(apply_steps_and_compare(all_train_dt.s1, all_train_dt.s2, perm, metric))
        test_features.append(apply_steps_and_compare(all_test_dt.s1, all_test_dt.s2, perm, metric))
all_train_dt = all_train_dt.assign(**{name: feature for name, feature in zip(feature_names, features)})
all_test_dt = all_test_dt.assign(**{name: feature for name, feature in zip(feature_names, test_features)})
all_train_dt.head()

# %%
all_train_dt.to_csv("../data/processed/train_data_with_features.csv", index=False)
all_test_dt.to_csv("../data/processed/test_data_with_features.csv", index=False)
