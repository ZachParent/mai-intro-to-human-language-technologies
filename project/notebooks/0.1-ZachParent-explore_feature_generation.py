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
def sentence_to_doc(sentence):
    return nlp(sentence)

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
# ## Previous Results

# %%
steps = [
    sentence_to_doc,
    remove_stopwords,
    lemmatize_tokens,
    lower,
    remove_non_alnum
]

previous_results = pd.DataFrame(
    { 'score':
        apply_steps_and_score(dt["s1"], dt["s2"], steps), 
        f's1[{DEMO_S_IDX}]': apply_steps_to_sentence(dt["s1"].iloc[DEMO_S_IDX], steps),
        f's2[{DEMO_S_IDX}]': apply_steps_to_sentence(dt["s2"].iloc[DEMO_S_IDX], steps),
    }, 
    index=[step.__name__ for step in steps]
)
previous_results

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

# %%
steps = [
    sentence_to_doc,
    chunk_NEs,
    remove_stopwords,
    lemmatize_tokens,
    lower,
    remove_non_alnum
]

steps_to_skip = [
    chunk_NEs,
    remove_stopwords,
    lower,
    remove_non_alnum,
]

leave_one_out_results = pd.DataFrame(columns=["score"])

for step_to_skip in steps_to_skip:
    steps_to_use = [step for step in steps if step != step_to_skip]
    print(f"Skipping {step_to_skip.__name__}")
    scores = list(apply_steps_and_score(dt["s1"], dt["s2"], steps_to_use))
    leave_one_out_results.loc[step_to_skip.__name__] = scores[-1]
leave_one_out_results.loc["full_pipeline"] = results.loc[:,"score"].iloc[-1]

leave_one_out_results = leave_one_out_results
print("Leave one out results:")
leave_one_out_results

# %%
# Plot bar chart for leave-one-out results and differences from full pipeline
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

fig.suptitle("Analysis of Leave-One-Out Configurations", fontsize=16)

# Original bar chart
leave_one_out_results.plot(kind="bar", color=["skyblue"], ax=ax1)
ax1.set_title(
    "Jaccard Distance Scores for While Skipping One of Each Preprocessing Step"
)
ax1.set_ylabel("Pearson correlation with Gold Standard", fontsize=12)
ax1.invert_yaxis()

# Annotate original bar chart values
for i, v in enumerate(leave_one_out_results["score"].values):
    ax1.text(i, v, f"{v:.3f}", ha="center", va="top", color="black")

# Calculate and plot differences from full pipeline
full_pipeline_score = leave_one_out_results.T["full_pipeline"].values[0]
differences = leave_one_out_results.subtract(full_pipeline_score)
differences.plot(kind="bar", color=["lightgreen"], ax=ax2)
ax2.set_title("Importance of each preprocessing step")
ax2.set_xlabel("Step Skipped", fontsize=12)
ax2.set_ylabel("Difference in Pearson correlation", fontsize=12)
ax2.tick_params(axis="x", rotation=45)

# Annotate difference bar chart values
ax2.axhline(0, color="white", linewidth=0.5)
for i, v in enumerate(differences.values):
    ax2.text(
        i,
        v,
        f"{v[0]:.3f}",
        ha="center",
        va="bottom" if v[0] > 0 else "top",
        color="white",
    )

plt.tight_layout()
plt.show()
