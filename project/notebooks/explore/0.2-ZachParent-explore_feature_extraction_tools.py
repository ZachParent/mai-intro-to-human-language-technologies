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
from scipy.stats import pearsonr
from nltk.metrics.distance import jaccard_distance
from zp_ihlt_project.feature_extraction import generate_valid_permutations

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
