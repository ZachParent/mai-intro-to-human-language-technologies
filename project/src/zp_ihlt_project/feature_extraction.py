import spacy
from nltk.metrics.distance import jaccard_distance
import itertools
import pandas as pd
from typing import List, Callable, Dict, Tuple
import inspect


nlp = spacy.load("en_core_web_sm")

# ====== Feature extraction methods ======

class PosTag(str):
    """A special string type for part-of-speech tags"""

    pass


class Word(str):
    """A special string type for words"""

    pass


# Spacy methods
def sentence_to_doc(sentence: str) -> spacy.tokens.doc.Doc:
    return nlp(sentence)


def get_tokens(doc: spacy.tokens.doc.Doc) -> List[spacy.tokens.token.Token]:
    return [token for token in doc]


def get_pos_tags(doc_or_tokens: List[spacy.tokens.token.Token]) -> List[PosTag]:
    return [token.pos_ for token in doc_or_tokens]


def lemmatize_tokens(tokens: List[spacy.tokens.token.Token]) -> List[Word]:
    return [token.lemma_ for token in tokens]


def get_token_text(tokens: List[spacy.tokens.token.Token]) -> List[Word]:
    return [token.text for token in tokens]


def chunk_NEs(doc: spacy.tokens.doc.Doc) -> List[spacy.tokens.token.Token]:
    with doc.retokenize() as retokenizer:
        tokens = [token for token in doc]
        for ent in doc.ents:
            retokenizer.merge(
                doc[ent.start : ent.end],
                attrs={"LEMMA": " ".join([tokens[i].text for i in range(ent.start, ent.end)])},
            )
    return [token for token in doc]


def remove_non_alnum(words: List[Word]) -> List[Word]:
    return [word for word in words if word.isalnum()]


def lower(words: List[Word]) -> List[Word]:
    return [word.lower() for word in words]


def remove_stopwords(
    doc_or_tokens: List[spacy.tokens.token.Token],
) -> List[spacy.tokens.token.Token]:
    return [token for token in doc_or_tokens if not token.is_stop]


# Automatically extract input and output types
def _extract_input_output_types(func: Callable) -> Tuple[type, type]:
    signature = inspect.signature(func)
    param_types = [param.annotation for param in signature.parameters.values()]
    return_type = signature.return_annotation
    return param_types[0], return_type


syntax_functions = [
    get_tokens,
    get_pos_tags,
    lemmatize_tokens,
    get_token_text,
]

semantic_functions = [chunk_NEs, remove_stopwords]
standard_functions = [sentence_to_doc, remove_non_alnum, lower]
all_functions = syntax_functions + semantic_functions + standard_functions

# Dictionary to hold function names and their input/output types
function_input_output_types: Dict[str, Tuple[Tuple[type, ...], type]] = {}

# Populate the dictionary with function names and their input/output types
for func in all_functions:
    input_types, output_type = _extract_input_output_types(func)
    function_input_output_types[func.__name__] = (input_types, output_type)


# Function to check if a permutation is valid based on input/output types
def _is_valid_permutation(perm: List[str]) -> bool:
    if function_input_output_types[perm[0].__name__][0] != str:
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


def generate_valid_permutations(functions: List[Callable] = all_functions) -> List[List[Callable]]:
    valid_permutations = []
    for n in range(1, len(functions) + 1):
        for perm in itertools.permutations(functions, n):
            if _is_valid_permutation(perm):
                valid_permutations.append(perm)
    return valid_permutations

# ====== Scoring methods ======

def jaccard_vector(tokens1, tokens2):
    return pd.concat([tokens1, tokens2], axis=1).apply(
        lambda x: 1 - jaccard_distance(set(x.iloc[0]), set(x.iloc[1])), axis=1
    )

def apply_steps_to_sentence_incrementally(sentence, steps):
    for i in range(len(steps)):
        sentence = steps[i](sentence)
        yield sentence

def apply_steps_to_sentence(sentence, steps):
    return list(apply_steps_to_sentence_incrementally(sentence, steps))[-1]

def apply_steps_and_compare_incrementally(s1_values, s2_values, steps):
    for i in range(len(steps)):
        s1_values = s1_values.apply(steps[i])
        s2_values = s2_values.apply(steps[i])
        if s1_values[0].__class__ != list or s1_values[0][0].__class__ != str:
            s1_tokens = s1_values.apply(get_token_text)
            s2_tokens = s2_values.apply(get_token_text)
        else:
            s1_tokens = s1_values
            s2_tokens = s2_values

        yield jaccard_vector(s1_tokens, s2_tokens)

def apply_steps_and_compare(s1_values, s2_values, steps):
    return list(apply_steps_and_compare_incrementally(s1_values, s2_values, steps))[-1]
