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
from zp_ihlt_project.feature_extraction import generate_valid_permutations
import importlib

# %%
nlp = spacy.load("en_core_web_sm")

# %%
BASE_PATH = "./"
DEMO_S_IDX = 79
N_SAMPLES = np.iinfo(np.int32).max

# %%
assert BASE_PATH is not None, "BASE_PATH is not set"

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

# %%
# Generate all valid permutations
valid_permutations = generate_valid_permutations()
print(len(valid_permutations))

# Example of processing a sentence with valid permutations
sentence = "The European Union is a political and economic union of 27 European countries."
for perm in valid_permutations:
    result = sentence
    print(f"Processing with order {', '.join(func.__name__ for func in perm)}: {result}")
    for func in perm:
        result = func(result)
    print(f"Processed: {result}")
