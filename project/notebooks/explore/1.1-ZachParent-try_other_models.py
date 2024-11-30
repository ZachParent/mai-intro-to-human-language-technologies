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
from zp_ihlt_project.load_data import load_train_data, load_test_data
from zp_ihlt_project.feature_extraction import generate_valid_permutations, apply_steps_and_compare

# %%
train_df = load_train_data()
test_df = load_test_data()

# %%
valid_permutations = generate_valid_permutations()
features = []
feature_steps = []
for i, perm in enumerate(valid_permutations):
    features.append(f"score_{i}")
    feature_steps.append(perm)
    train_df[f"score_{i}"] = apply_steps_and_compare(train_df.s1, train_df.s2, perm)
    test_df[f"score_{i}"] = apply_steps_and_compare(test_df.s1, test_df.s2, perm)
train_df.head()


# %%
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

X_train = train_df[features]
y_train = train_df.gs
X_test = test_df[features]
y_test = test_df.gs

# Define parameter grid
param_grid = {
    'hidden_layer_sizes': [(200,40), (200, 20), (400, 40), (200, 80), (400, 80), (400, 20)],
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
preds = best_mlp.predict(X_test)
pearsonr(y_test, preds)

# %% [markdown]
# `PearsonRResult(statistic=0.653254868028039, pvalue=0.0)`
#
# Achieves 0.65 on the test set
