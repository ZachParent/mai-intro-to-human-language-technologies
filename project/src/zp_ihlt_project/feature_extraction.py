import spacy
import nltk
from nltk.metrics.distance import jaccard_distance
import itertools
import pandas as pd
from typing import List, Callable, Dict, Tuple
import inspect
from functools import cache, partial
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

nlp = spacy.load("en_core_web_sm")
nltk.download("wordnet")
nltk.download("omw-1.4")


class PosTag(str):
    pass


class Word(str):
    pass

class CharacterNgram(str):
    pass

class WordNgram(Tuple[str, ...]):
    pass

@cache
def get_character_ngrams(sentence: str, n: int = 3) -> Tuple[CharacterNgram, ...]:
    ngrams = [sentence[i:i+n] for i in range(len(sentence)-n+1)]
    return tuple(CharacterNgram(ngram) for ngram in ngrams)

def get_character_2grams(sentence: str) -> Tuple[CharacterNgram, ...]:
    return get_character_ngrams(sentence, 2)

def get_character_3grams(sentence: str) -> Tuple[CharacterNgram, ...]:
    return get_character_ngrams(sentence, 3)

def get_character_4grams(sentence: str) -> Tuple[CharacterNgram, ...]:
    return get_character_ngrams(sentence, 4)

def get_character_5grams(sentence: str) -> Tuple[CharacterNgram, ...]:
    return get_character_ngrams(sentence, 5)


@cache
def sentence_to_doc(sentence: str) -> spacy.tokens.doc.Doc:
    return nlp(sentence)


@cache
def get_tokens(doc: spacy.tokens.doc.Doc) -> Tuple[spacy.tokens.token.Token, ...]:
    return tuple(token for token in doc)


@cache
def get_pos_tags(tokens: Tuple[spacy.tokens.token.Token, ...]) -> Tuple[PosTag, ...]:
    return tuple(token.pos_ for token in tokens)


@cache
def lemmatize_tokens(tokens: Tuple[spacy.tokens.token.Token, ...]) -> Tuple[Word, ...]:
    return tuple(token.lemma_ for token in tokens)


@cache
def get_token_text(tokens: Tuple[spacy.tokens.token.Token, ...]) -> Tuple[Word, ...]:
    return tuple(token.text for token in tokens)

@cache
def get_word_ngrams(words: Tuple[Word, ...], n: int = 3) -> Tuple[WordNgram, ...]:
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return tuple(WordNgram(ngram) for ngram in ngrams)

@cache
def chunk_NEs(doc: spacy.tokens.doc.Doc) -> Tuple[spacy.tokens.token.Token, ...]:
    with doc.retokenize() as retokenizer:
        tokens = [token for token in doc]
        for ent in doc.ents:
            retokenizer.merge(
                doc[ent.start : ent.end],
                attrs={"LEMMA": " ".join([tokens[i].text for i in range(ent.start, ent.end)])},
            )
    return tuple(token for token in doc)


POS_TAGS_MAPPING = {
    "NOUN": "n",
    "VERB": "v",
    "ADV": "r",
}


@cache
def get_synsets(tokens: Tuple[spacy.tokens.token.Token, ...]) -> Tuple[Word, ...]:
    pos_tags = get_pos_tags(tokens)
    words = get_token_text(tokens)
    result = []
    for word, pos_tag in zip(words, pos_tags):
        mapped_pos_tag = POS_TAGS_MAPPING.get(pos_tag, None)
        if mapped_pos_tag is None:
            result.append(word)
            continue
        synset = nltk.wsd.lesk(words, word, mapped_pos_tag)
        if synset is None:
            result.append(word)
            continue
        result.append(synset.name())
    return tuple(result)


@cache
def remove_non_alnum(words: Tuple[Word, ...]) -> Tuple[Word, ...]:
    return tuple(word for word in words if word.isalnum())


@cache
def lower(words: Tuple[Word, ...]) -> Tuple[Word, ...]:
    return tuple(word.lower() for word in words)


@cache
def remove_stopwords(
    tokens: Tuple[spacy.tokens.token.Token, ...]
) -> Tuple[spacy.tokens.token.Token, ...]:
    return tuple(token for token in tokens if not token.is_stop)


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
    get_synsets,
]

semantic_functions = [chunk_NEs, remove_stopwords, get_word_ngrams]
standard_functions = [remove_non_alnum, lower]
all_functions = syntax_functions + semantic_functions + standard_functions

# Dictionary to hold function names and their input/output types
function_input_output_types: Dict[str, Tuple[Tuple[type, ...], type]] = {}

# Populate the dictionary with function names and their input/output types
for func in all_functions:
    input_types, output_type = _extract_input_output_types(func)
    function_input_output_types[func.__name__] = (input_types, output_type)


# Function to check if a permutation is valid based on input/output types
def _is_valid_permutation(perm: List[str]) -> bool:
    if function_input_output_types[perm[0].__name__][0] != spacy.tokens.doc.Doc:
        return False
    if function_input_output_types[perm[-1].__name__][1] not in [
        Tuple[Word, ...],
        Tuple[PosTag, ...],
        Tuple[WordNgram, ...],
    ]:
        return False
    for i in range(len(perm) - 1):
        _, current_func_output_type = function_input_output_types[perm[i].__name__]
        next_func_input_type, _ = function_input_output_types[perm[i + 1].__name__]
        if current_func_output_type != next_func_input_type:
            return False
    return True


def generate_valid_permutations(
    functions: List[Callable] = all_functions,
) -> List[Tuple[Callable, ...]]:
    valid_permutations = []
    for n in range(1, len(functions) + 1):
        for perm in itertools.permutations(functions, n):
            if _is_valid_permutation(perm):
                valid_permutations.append(perm)
    valid_permutations = [tuple([sentence_to_doc]) + perm for perm in valid_permutations]
    valid_permutations.extend([[get_character_2grams], [get_character_3grams], [get_character_4grams], [get_character_5grams]])
    return valid_permutations


# ====== Scoring methods ======


@cache
def vectorize_sets(set1: set, set2: set) -> np.ndarray:
    """Convert two sets into binary vectors using their union as vocabulary"""
    vocabulary = list(set1.union(set2))
    if not vocabulary:
        return np.array([[0], [0]])
    
    vec1 = np.array([1 if word in set1 else 0 for word in vocabulary])
    vec2 = np.array([1 if word in set2 else 0 for word in vocabulary])
    return np.vstack([vec1, vec2])

def compute_similarity(set1: set, set2: set, metric: str = 'jaccard') -> float:
    """Compute similarity between two sets using specified metric"""
    if not set1 and not set2:  # Both empty
        return 1.0
    if not set1 or not set2:  # One empty
        return 0.0
        
    if metric == 'jaccard':
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union
    
    # Convert sets to vectors for other metrics
    vectors = vectorize_sets(set1, set2)
    
    if metric == 'cosine':
        return float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])
    elif metric == 'euclidean':
        return float(1 / (1 + euclidean_distances(vectors[0:1], vectors[1:2])[0][0]))
    elif metric == 'manhattan':
        return float(1 / (1 + manhattan_distances(vectors[0:1], vectors[1:2])[0][0]))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def similarity_vector(tokens1, tokens2, metric: str = 'jaccard'):
    """Compute similarity vector using specified metric"""
    def safe_similarity(x):
        set1 = set(x.iloc[0])
        set2 = set(x.iloc[1])
        return compute_similarity(set1, set2, metric)

    return pd.concat([tokens1, tokens2], axis=1).apply(safe_similarity, axis=1)

def apply_steps_and_compare_incrementally(s1_values, s2_values, steps, metric: str = 'jaccard'):
    """Apply steps and compare incrementally using specified metric"""
    for i in range(len(steps)):
        s1_values = s1_values.apply(steps[i])
        s2_values = s2_values.apply(steps[i])
        
        if s1_values[0].__class__ == spacy.tokens.doc.Doc or s1_values[0][0].__class__ == spacy.tokens.token.Token:
            s1_tokens = s1_values.apply(get_token_text)
            s2_tokens = s2_values.apply(get_token_text)
        else:
            s1_tokens = s1_values
            s2_tokens = s2_values

        yield similarity_vector(s1_tokens, s2_tokens, metric)

def apply_steps_and_compare(s1_values, s2_values, steps, metric: str = 'jaccard'):
    """Apply steps and compare using specified metric"""
    return list(apply_steps_and_compare_incrementally(s1_values, s2_values, steps, metric))[-1]

def apply_steps_and_compare_all_metrics(s1_values, s2_values, steps):
    return [apply_steps_and_compare(s1_values, s2_values, steps, metric) for metric in ['jaccard', 'cosine', 'euclidean', 'manhattan']]

# Example usage:
"""
# In your notebook:
metrics = ['jaccard', 'cosine', 'euclidean', 'manhattan']
for metric in metrics:
    for i, perm in enumerate(valid_permutations):
        feature_name = f"score_{metric}_{i}"
        features.append(feature_name)
        feature_steps.append(perm)
        dt[feature_name] = apply_steps_and_compare(dt.s1, dt.s2, perm, metric=metric)
"""
