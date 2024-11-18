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

from zp_ihlt_project.feature_extraction import generate_valid_permutations, sentence_to_doc, get_tokens, get_pos_tags, get_token_text
from zp_ihlt_project.load_data import load_train_data

nltk.download("wordnet")
nltk.download("omw-1.4")

# %%
dt = load_train_data()

# %%
POS_TAGS_MAPPING = {
    "NOUN": "n",
    "VERB": "v",
    "ADV": "r",
}

def get_synsets(tokens: Tuple[spacy.tokens.Token, ...]) -> Tuple[str, ...]:
    pos_tags = get_pos_tags(tokens)
    words = get_token_text(tokens)
    print(list(zip(words, pos_tags)))
    synsets = [nltk.wsd.lesk(words, word, POS_TAGS_MAPPING[pos_tag]).name() if pos_tag in POS_TAGS_MAPPING else word for word, pos_tag in zip(words, pos_tags) ]
    print(synsets)
    return tuple(synset for synset in synsets)

get_synsets(get_tokens(sentence_to_doc(dt.s1.iloc[0])))

