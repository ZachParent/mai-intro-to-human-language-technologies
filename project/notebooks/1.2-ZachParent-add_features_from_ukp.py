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

from zp_ihlt_project.feature_extraction import generate_valid_permutations, apply_steps_and_compare, get_word_ngrams
from zp_ihlt_project.load_data import load_train_data

nltk.download("wordnet")
nltk.download("omw-1.4")

# %%
dt = load_train_data()


# %%
def get_character_ngrams(sentence: str, n: int = 3) -> Tuple[str, ...]:
    ngrams = [sentence[i:i+n] for i in range(len(sentence)-n+1)]
    return tuple(ngram for ngram in ngrams)


get_character_ngrams(dt.s1.iloc[0])


# %%
def get_word_ngrams(sentence: str, n: int = 3) -> Tuple[Tuple[str, ...], ...]:
    words = sentence.split()
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return tuple(ngram for ngram in ngrams)


get_word_ngrams(dt.s1.iloc[0])



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
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

X_train, X_test, y_train, y_test = train_test_split(dt[features], dt.gs, test_size=0.2, random_state=42)

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
