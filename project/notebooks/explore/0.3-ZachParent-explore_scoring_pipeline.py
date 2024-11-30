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
from zp_ihlt_project.feature_extraction import generate_valid_permutations, get_token_text
import importlib

# %%
BASE_PATH = "./"
DEMO_S_IDX = 79
# N_SAMPLES = np.iinfo(np.int32).max
N_SAMPLES = 10

# %% [markdown]
# ### Load the data

# %%
dt = pd.read_csv(
    f"{TRAIN_DATA_DIR}/STS.input.SMTeuroparl.txt", sep="\t", header=None
)
dt.columns = ["s1", "s2"]
gs = pd.read_csv(f"{TRAIN_DATA_DIR}/STS.gs.SMTeuroparl.txt", sep="\t", header=None)
dt["gs"] = gs[0]
dt = dt.iloc[:N_SAMPLES]
gs = gs.iloc[:N_SAMPLES]
dt.head()


# %% [markdown]
# ## Tools

# %% [markdown]
# ### Previous NLP tools

# %% [markdown]
# ### Scoring and pipeline methods

# %%
# Scoring methods
def jaccard_vector(s1, s2):
    return pd.concat([s1, s2], axis=1).apply(
        lambda x: 1 - jaccard_distance(set(x.iloc[0]), set(x.iloc[1])), axis=1
    )


def score_jaccard_vector(jaccard_vector, gold_standard=gs):
    return pearsonr(gold_standard[0], jaccard_vector)[0]


# %%
def apply_steps_to_sentence_incrementally(sentence, steps):
    for i in range(len(steps)):
        sentence = steps[i](sentence)
        yield sentence

def apply_steps_to_sentence(sentence, steps):
    return list(apply_steps_to_sentence_incrementally(sentence, steps))[-1]

def apply_steps_and_score_incrementally(s1_values, s2_values, steps):
    for i in range(len(steps)):
        s1_values = s1_values.apply(steps[i])
        s2_values = s2_values.apply(steps[i])
        if s1_values[0].__class__ != list or s1_values[0][0].__class__ != str:
            s1_tokens = s1_values.apply(get_token_text)
            s2_tokens = s2_values.apply(get_token_text)
        else:
            s1_tokens = s1_values
            s2_tokens = s2_values

        yield score_jaccard_vector(jaccard_vector(s1_tokens, s2_tokens))

def apply_steps_and_score(s1_values, s2_values, steps):
    return list(apply_steps_and_score_incrementally(s1_values, s2_values, steps))[-1]




# %%
# Generate all valid permutations
valid_permutations = generate_valid_permutations()
list(apply_steps_to_sentence(dt.s1.iloc[0], valid_permutations[0]))


# %%
for i, perm in enumerate(valid_permutations):
    dt[f"score_{i}"] = apply_steps_and_score(dt.s1, dt.s2, perm)
dt.head()

