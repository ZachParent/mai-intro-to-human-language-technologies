import spacy
import itertools
from typing import List, Callable, Dict, Tuple

nlp = spacy.load("en_core_web_sm")


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


# List of processing functions
functions = [
    sentence_to_doc,
    get_tokens,
    get_pos_tags,
    lemmatize_tokens,
    get_token_text,
    remove_non_alnum,
    lower,
    remove_stopwords,
]

# List of processing functions with their input and output types
function_input_output_types: Dict[str, Tuple[Callable, Tuple[type, type]]] = {
    sentence_to_doc.__name__: (str, spacy.tokens.doc.Doc),
    get_tokens.__name__: (spacy.tokens.doc.Doc, List[spacy.tokens.token.Token]),
    get_pos_tags.__name__: (List[spacy.tokens.token.Token], List[PosTag]),
    get_token_text.__name__: (List[spacy.tokens.token.Token], List[Word]),
    lemmatize_tokens.__name__: (List[spacy.tokens.token.Token], List[Word]),
    remove_non_alnum.__name__: (List[Word], List[Word]),
    lower.__name__: (List[Word], List[Word]),
    remove_stopwords.__name__: (List[spacy.tokens.token.Token], List[spacy.tokens.token.Token]),
}


# Function to check if a permutation is valid based on input/output types
def is_valid_permutation(perm: List[str]) -> bool:
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


def generate_valid_permutations() -> List[List[Callable]]:
    valid_permutations = []
    for n in range(1, len(functions) + 1):
        for perm in itertools.permutations(functions, n):
            if is_valid_permutation(perm):
                valid_permutations.append(perm)
    return valid_permutations
