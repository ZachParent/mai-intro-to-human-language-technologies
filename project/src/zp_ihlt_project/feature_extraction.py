import spacy
import nltk
import itertools
import pandas as pd
from typing import List, Callable, Dict, Tuple, Iterator
import inspect
from functools import cache
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
import numpy as np

nlp = spacy.load("en_core_web_sm")
nltk.download("wordnet")
nltk.download("omw-1.4")


class PosTag(str):
    pass


class Word(str):
    pass

class Character(str):
    pass

class Ngram(Tuple[Word | Character | PosTag, ...]):
    pass

class WordPair(Tuple[Word, Word]):
    pass

@cache
def get_characters(words: Tuple[Word, ...]) -> Tuple[Character, ...]:
    sentence = " ".join(words)
    return tuple(Character(char) for char in sentence)

@cache
def get_word_pairs(words: Tuple[Word, ...]) -> Tuple[WordPair, ...]:
    return tuple(WordPair(pair) for pair in itertools.combinations(words, 2))

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
    return tuple(token.lemma_.lower() for token in tokens)


@cache
def get_token_text(tokens: Tuple[spacy.tokens.token.Token, ...]) -> Tuple[Word, ...]:
    return tuple(token.text.lower() for token in tokens)


@cache
def get_ngrams(words: Tuple[Word | Character | PosTag, ...], n: int = 3) -> Tuple[Ngram, ...]:
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return tuple(Ngram(ngram) for ngram in ngrams)

def get_2grams(words: Tuple[Word | Character | PosTag, ...]) -> Tuple[Ngram, ...]:
    return get_ngrams(words, 2)

def get_3grams(words: Tuple[Word | Character | PosTag, ...]) -> Tuple[Ngram, ...]:
    return get_ngrams(words, 3)

def get_4grams(words: Tuple[Word | Character | PosTag, ...]) -> Tuple[Ngram, ...]:
    return get_ngrams(words, 4)

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
def remove_stopwords(
    tokens: Tuple[spacy.tokens.token.Token, ...]
) -> Tuple[spacy.tokens.token.Token, ...]:
    return tuple(token for token in tokens if not token.is_stop)

@cache
def get_stopwords(tokens: Tuple[spacy.tokens.token.Token, ...]) -> Tuple[spacy.tokens.token.Token, ...]:
    return tuple(token for token in tokens if token.is_stop)


def _extract_input_output_types(func: Callable) -> Tuple[type, type]:
    signature = inspect.signature(func)
    param_types = [param.annotation for param in signature.parameters.values()]
    return_type = signature.return_annotation
    return param_types[0], return_type


# Lexical features (surface form, characters, word forms)
lexical_functions = [
    get_characters,      # Character-level patterns
    get_tokens,          # Word tokenization
    get_token_text,      # Raw word forms
    remove_non_alnum,    # Character filtering
    get_word_pairs,      # Word co-occurrences
]

# Semantic features (meaning, concepts)
semantic_functions = [
    lemmatize_tokens,    # Normalize to base meaning
    get_synsets,         # Word meanings/concepts
    chunk_NEs,           # Named entity grouping
    get_pos_tags,        # Part of speech (bridges lexical/semantic)
]

ngram_functions = [
    get_2grams,          # Bigrams
    get_3grams,          # Trigrams
    get_4grams,          # 4-grams
]

preprocessing_functions = [
    remove_stopwords,    # Filter non-content words
    get_stopwords,       # Identify non-content words
]

all_functions = lexical_functions + semantic_functions + preprocessing_functions

# Dictionary to hold function names and their input/output types
function_input_output_types: Dict[str, Tuple[Tuple[type, ...], type]] = {}

# Populate the dictionary with function names and their input/output types
for func in all_functions:
    input_types, output_type = _extract_input_output_types(func)
    function_input_output_types[func.__name__] = (input_types, output_type)


# Function to check if a permutation is valid based on input/output types
def _is_valid_permutation(perm: Tuple[Callable]) -> bool:
    if function_input_output_types[perm[0].__name__][0] != spacy.tokens.doc.Doc:
        return False
    if function_input_output_types[perm[-1].__name__][1] not in [
        Tuple[Word, ...],
        Tuple[PosTag, ...],
        Tuple[Character, ...],
    ]:
        return False
    for i in range(len(perm) - 1):
        _, current_func_output_type = function_input_output_types[perm[i].__name__]
        next_func_input_type, _ = function_input_output_types[perm[i + 1].__name__]
        if current_func_output_type != next_func_input_type:
            return False
    return True

def add_final_step(perm: Tuple[Callable]) -> Iterator[List[Callable]]:
    yield perm
    for func in ngram_functions:
        yield perm + (func,)

def generate_valid_permutations(
    functions: List[Callable] = all_functions,
) -> List[Tuple[Callable, ...]]:
    valid_permutations = []
    for n in range(1, len(functions) + 1):
        for perm in itertools.permutations(functions, n):
            if _is_valid_permutation(perm):
                valid_permutations.append(perm)
    valid_permutations = [tuple([sentence_to_doc]) + perm for perm in valid_permutations]
    valid_permutations = [new_perm for perm in valid_permutations for new_perm in add_final_step(perm)]
    return valid_permutations


# ====== Scoring methods ======


@cache
def vectorize_tuples(tup1: Tuple[str, ...], tup2: Tuple[str, ...]) -> np.ndarray:
    """Convert two tuples into binary vectors using their union as vocabulary"""
    vocabulary = tuple(sorted(set(tup1).union(set(tup2))))
    if not vocabulary:
        return np.array([[0], [0]])
    
    vec1 = np.array([1 if word in tup1 else 0 for word in vocabulary])
    vec2 = np.array([1 if word in tup2 else 0 for word in vocabulary])
    return np.vstack([vec1, vec2])

@cache
def compute_similarity(tup1: Tuple[str, ...], tup2: Tuple[str, ...], metric: str = 'jaccard') -> float:
    """Compute similarity between two tuples using specified metric"""
    if not tup1 and not tup2:  # Both empty
        return 1.0
    if not tup1 or not tup2:  # One empty
        return 0.0
        
    if metric == 'jaccard':
        set1, set2 = set(tup1), set(tup2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union
    
    # Convert tuples to vectors for other metrics
    vectors = vectorize_tuples(tup1, tup2)
    
    if metric == 'cosine':
        return float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])
    elif metric == 'euclidean':
        return float(1 / (1 + euclidean_distances(vectors[0:1], vectors[1:2])[0][0]))
    elif metric == 'manhattan':
        return float(1 / (1 + manhattan_distances(vectors[0:1], vectors[1:2])[0][0]))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def similarity_vector(tokens1: pd.Series, tokens2: pd.Series, metric: str = 'jaccard') -> pd.Series:
    """Compute similarity vector using specified metric"""
    def safe_similarity(x):
        # Convert to tuples for caching
        tup1 = tuple(sorted(x.iloc[0]))  # Sort for consistent caching
        tup2 = tuple(sorted(x.iloc[1]))
        return compute_similarity(tup1, tup2, metric)

    return pd.concat([tokens1, tokens2], axis=1).apply(safe_similarity, axis=1)

def apply_steps_and_compare_incrementally(s1_values: pd.Series, 
                                        s2_values: pd.Series, 
                                        steps: Tuple[Callable, ...], 
                                        metric: str = 'jaccard') -> Iterator[pd.Series]:
    """Apply steps and compare incrementally using specified metric"""
    for i in range(len(steps)):
        s1_values = s1_values.apply(steps[i])
        s2_values = s2_values.apply(steps[i])
        
        if s1_values.iloc[0].__class__ == spacy.tokens.doc.Doc or (
            isinstance(s1_values.iloc[0], (tuple, list)) and 
            len(s1_values.iloc[0]) > 0 and 
            isinstance(s1_values.iloc[0][0], spacy.tokens.token.Token)
        ):
            s1_tokens = s1_values.apply(get_token_text)
            s2_tokens = s2_values.apply(get_token_text)
        else:
            s1_tokens = s1_values
            s2_tokens = s2_values

        yield similarity_vector(s1_tokens, s2_tokens, metric)

def apply_steps_and_compare(s1_values: pd.Series, 
                          s2_values: pd.Series, 
                          steps: Tuple[Callable, ...], 
                          metric: str = 'jaccard') -> pd.Series:
    """Apply steps and compare using specified metric"""
    return list(apply_steps_and_compare_incrementally(s1_values, s2_values, steps, metric))[-1]

def apply_steps_and_compare_all_metrics(s1_values: pd.Series, 
                                      s2_values: pd.Series, 
                                      steps: Tuple[Callable, ...]) -> List[pd.Series]:
    """Apply steps and compare using all available metrics"""
    return [
        apply_steps_and_compare(s1_values, s2_values, steps, metric) 
        for metric in ['jaccard', 'cosine', 'euclidean', 'manhattan']
    ]


