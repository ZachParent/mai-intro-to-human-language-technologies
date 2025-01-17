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
import spacy
from typing import Tuple
import nltk
import numpy as np
import pandas as pd
from functools import partial

from zp_ihlt_project.feature_extraction import generate_valid_permutations, apply_steps_and_compare, get_word_ngrams, get_character_ngrams, sentence_to_doc, get_tokens, get_token_text
from zp_ihlt_project.load_data import load_train_data, load_test_data, load_all_data_files
from zp_ihlt_project.config import TRAIN_DATA_DIR


# %%
dfs = load_all_data_files(TRAIN_DATA_DIR)

# %%
dt = dfs[2]

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
    for metric in metrics:
        feature_names.append(f"score_{metric}_{i}")
        feature_steps.append(perm)
        features.append(apply_steps_and_compare(dt.s1, dt.s2, perm, metric))
        # test_features.append(apply_steps_and_compare(test_dt.s1, test_dt.s2, perm, metric))
dt = dt.assign(**{name: feature for name, feature in zip(feature_names, features)})
# test_dt = test_dt.assign(**{name: feature for name, feature in zip(feature_names, test_features)})
dt.head()

# %%
from sklearn.feature_selection import SelectKBest, mutual_info_regression

kbest = SelectKBest(mutual_info_regression, k=1000).fit(dt[feature_names], dt.gs)
selected_features = np.array(feature_names)[kbest.get_support()]
selected_steps = [feature_steps[i] for i in np.where(kbest.get_support())[0]]
[[step.__name__ for step in steps] for steps in selected_steps]

# %%
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

X_train, X_test, y_train, y_test = train_test_split(dt[selected_features], dt.gs, test_size=0.2, random_state=42)

# Define parameter grid
param_grid = {
    'hidden_layer_sizes': [(800, 160), (800, 80)],
    'activation': ['tanh'],
    'solver': ['adam']
}

# Create MLPRegressor
mlp = MLPRegressor(max_iter=500, random_state=42)

# Perform grid search with 5-fold cross validation
grid_search = GridSearchCV(
    mlp, 
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# Fit grid search
grid_search.fit(X_train, y_train)

# Print best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best RMSE:", np.sqrt(-grid_search.best_score_))

best_mlp = grid_search.best_estimator_
preds = best_mlp.predict(X_train)
pearsonr(y_train, preds)

# %%
best_mlp = grid_search.best_estimator_
preds = best_mlp.predict(X_test[selected_features])
pearsonr(y_test, preds)

