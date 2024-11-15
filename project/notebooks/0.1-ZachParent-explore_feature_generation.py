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
# Spacy methods
def get_tokens(doc):
    return [token for token in doc]

def sentence_to_doc(sentence):
    return nlp(sentence)

def get_pos_tags(doc_or_tokens):
    return [token.pos_ for token in doc_or_tokens]

def lemmatize_tokens(tokens):
    return [token.lemma_ for token in tokens]

def get_token_text(doc_or_tokens):
    if isinstance(doc_or_tokens[0], spacy.tokens.token.Token):
        return [token.text for token in doc_or_tokens]
    elif isinstance(doc_or_tokens[0], tuple):
        return [token[0] for token in doc_or_tokens]
    else:
        raise ValueError("Invalid input type")

def remove_non_alnum(words : list[str]):
    return [word for word in words if word.isalnum()]

def lower(words : list[str]):
    return [word.lower() for word in words]

def remove_stopwords(doc_or_tokens):
    return [token for token in doc_or_tokens if not token.is_stop]


# %%
import itertools
from typing import List, Callable, Dict, Tuple

class PosTag(str):
    """A special string type for part-of-speech tags"""
    pass

class Word(str):
    """A special string type for words"""
    pass

# List of processing functions
functions = [
    sentence_to_doc,
    get_pos_tags,
    lemmatize_tokens,
    get_token_text,
    remove_non_alnum,
    lower,
    # remove_stopwords
]

# List of processing functions with their input and output types
function_input_output_types: Dict[str, Tuple[Callable, Tuple[type, type]]] = {
    sentence_to_doc.__name__: (str, spacy.tokens.doc.Doc),
    get_pos_tags.__name__: (spacy.tokens.doc.Doc, List[PosTag]),
    get_token_text.__name__: (spacy.tokens.doc.Doc, List[Word]),
    lemmatize_tokens.__name__: (spacy.tokens.doc.Doc, List[Word]),
    remove_non_alnum.__name__: (List[Word], List[Word]),
    lower.__name__: (List[Word], List[Word]),
    # remove_stopwords.__name__: (List[Word], List[Word]),
}

# Function to check if a permutation is valid based on input/output types
def is_valid_permutation(perm: List[str]) -> bool:
    if perm[0].__name__ != sentence_to_doc.__name__:
        return False
    if function_input_output_types[perm[-1].__name__][1] not in [List[Word], List[PosTag]]:
        return False
    for i in range(len(perm) - 1):
        _, current_func_output_type = function_input_output_types[perm[i].__name__]
        next_func_input_type, _ = function_input_output_types[perm[i + 1].__name__]
        # Check if the output type of the current function matches the input type of the next function
        if current_func_output_type != next_func_input_type:
            return False
    return True

# Generate all valid permutations
valid_permutations = []
for n in range(1, len(functions) + 1):
    for perm in itertools.permutations(functions, n):
        if is_valid_permutation(perm):
            valid_permutations.append(perm)
print(len(valid_permutations))

# Example of processing a sentence with valid permutations
sentence = "This is a sample sentence."
for perm in valid_permutations:
    result = sentence
    for func in perm:
        result = func(result)
    print(f"Processed with order {', '.join(func.__name__ for func in perm)}: {result}")


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
