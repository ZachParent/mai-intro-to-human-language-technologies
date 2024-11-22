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
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split
from scipy.stats import pearsonr

from zp_ihlt_project.config import TRAIN_DATA_DIR, TEST_DATA_DIR


# %%
all_train_dt = pd.read_csv("../data/processed/train_data_with_features.csv")
all_test_dt = pd.read_csv("../data/processed/test_data_with_features.csv")
feature_names = [col for col in all_train_dt.columns if col.startswith("score_")]

# %%
from sklearn.feature_selection import SelectKBest, mutual_info_regression

kbest = SelectKBest(mutual_info_regression, k=10000).fit(all_train_dt[feature_names], all_train_dt.gs)
selected_features = np.array(feature_names)[kbest.get_support()]

# %%
train_datasets = all_train_dt.dataset.unique().tolist()
test_datasets = all_test_dt.dataset.unique().tolist()

# %%
from sklearn.ensemble import RandomForestRegressor

dt = all_train_dt

X_train, X_val, y_train, y_val = train_test_split(dt[selected_features], dt.gs, test_size=0.2, random_state=42)
X_test = all_test_dt[selected_features]
y_test = all_test_dt.gs

results = []

# Define parameter grid
param_grid = {}

model = RandomForestRegressor()

# Perform grid search with 5-fold cross validation
grid_search = GridSearchCV(
    model, 
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

best_model = grid_search.best_estimator_
preds = best_model.predict(X_train)
results.append(pearsonr(y_train, preds)[0])

preds = best_model.predict(X_val)
results.append(pearsonr(y_val, preds)[0])

preds = best_model.predict(X_test)
results.append(pearsonr(y_test, preds)[0])

results = pd.DataFrame([results], columns=["train_pearson", "val_pearson", "test_pearson"], index=["all"])
results

# %%
best_model.fit(dt[selected_features], dt.gs)


# %%
train_results = []
datasets = train_datasets
for dataset in datasets:
    dataset_results = []
    dataset_results.append(dataset)
    dt = all_train_dt[all_train_dt.dataset == dataset]
    test_dt = all_test_dt[all_test_dt.dataset == dataset]

    X_train, X_val, y_train, y_val = train_test_split(dt[selected_features], dt.gs, test_size=0.2, random_state=42)
    X_test = test_dt[selected_features]
    y_test = test_dt.gs

    preds = best_model.predict(X_train)
    dataset_results.append(pearsonr(y_train, preds)[0])

    preds = best_model.predict(X_val)
    dataset_results.append(pearsonr(y_val, preds)[0])

    preds = best_model.predict(X_test[selected_features])
    dataset_results.append(pearsonr(y_test, preds)[0])

    train_results.append(dataset_results)

train_results = pd.DataFrame(train_results, columns=["dataset", "train_pearson", "val_pearson", "test_pearson"])
train_results
# %%
test_results = []
datasets = test_datasets
for dataset in datasets:
    dataset_results = []
    dataset_results.append(dataset)
    test_dt = all_test_dt[all_test_dt.dataset == dataset]

    X_test = test_dt[selected_features]
    y_test = test_dt.gs

    preds = best_model.predict(X_test[selected_features])
    dataset_results.append(pearsonr(y_test, preds)[0])

    test_results.append(dataset_results)

preds = best_model.predict(all_test_dt[selected_features])
test_results.append(['all', pearsonr(all_test_dt.gs, preds)[0]])
test_results = pd.DataFrame(test_results, columns=["dataset", "test_pearson"])
test_results
# %%
results_to_beat = pd.DataFrame(np.array([[.683, .873, .528, .664, .493, 0.823]]).T, index=[*test_datasets, 'all'], columns=["pearson_to_beat"])
results_to_beat

