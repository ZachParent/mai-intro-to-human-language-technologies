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
from zp_ihlt_project.feature_generation import generate_valid_permutations
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
sentence = "This is a sample sentence."
for perm in valid_permutations:
    result = sentence
    print(f"Processing with order {', '.join(func.__name__ for func in perm)}: {result}")
    for func in perm:
        result = func(result)
    print(f"Processed: {result}")


# %% [markdown]
# ### Scoring and pipeline methods

# %%
# Scoring methods
def jaccard_vector(s1, s2):
    return pd.concat([s1, s2], axis=1).apply(
        lambda x: jaccard_distance(set(x.iloc[0]), set(x.iloc[1])), axis=1
    )


def score_jaccard_vector(jaccard_vector, gold_standard=gs):
    return pearsonr(gold_standard[0], jaccard_vector)[0]


# %%
def apply_steps_to_sentence(sentence, steps):
    for i in range(len(steps)):
        sentence = steps[i](sentence)
        yield sentence

def apply_steps_and_score(s1_values, s2_values, steps):
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



# %% [markdown]
# ### Named Entity (NE) chunking
#
# This is the key new step.

# %%
def chunk_NEs(doc):
    with doc.retokenize() as retokenizer:
        tokens = [token for token in doc]
        for ent in doc.ents:
            retokenizer.merge(doc[ent.start:ent.end], 
                attrs={"LEMMA": " ".join([tokens[i].text 
                                    for i in range(ent.start, ent.end)])})
    return [token for token in doc]


# %% [markdown]
# ## Results including Named Entity chunking

# %%
steps = [
    sentence_to_doc,
    chunk_NEs,
    remove_stopwords,
    lemmatize_tokens,
    lower,
    remove_non_alnum
]
results = pd.DataFrame(
    { 'score':
        apply_steps_and_score(dt["s1"], dt["s2"], steps), 
        's1[0]': apply_steps_to_sentence(dt["s1"].iloc[DEMO_S_IDX], steps),
        's2[0]': apply_steps_to_sentence(dt["s2"].iloc[DEMO_S_IDX], steps),
    }, 
    index=[step.__name__ for step in steps]
)
results

# %% [markdown]
# ## Analysis

# %%
plt.figure(figsize=(10, 6))
plt.plot(results['score'], color="blue", label="With NE chunking")
plt.plot(previous_results['score'], color="red", label="Previous results")
plt.title("Jaccard Distance Scores after Each Preprocessing Step")
plt.xlabel("Preprocessing Step")
plt.ylabel("Pearson correlation with Gold Standard")
plt.xticks(rotation=45)
plt.gca().invert_yaxis()  # Invert the y-axis
plt.tight_layout()
plt.legend()
# Annotate chart values
for i, v in enumerate(results["score"].values):
    plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", color="white")
for i, v in enumerate(previous_results["score"].values):
    plt.text(i + 1 if i > 0 else i, v, f"{v:.3f}", ha="center", va="bottom", color="white")
plt.show()
