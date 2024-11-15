import spacy
import itertools
from typing import List, Callable, Dict, Tuple
import inspect

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


# Automatically extract input and output types
def _extract_input_output_types(func: Callable) -> Tuple[type, type]:
    signature = inspect.signature(func)
    param_types = [param.annotation for param in signature.parameters.values()]
    return_type = signature.return_annotation
    return param_types[0], return_type


# List of processing functions
functions = [
    sentence_to_doc,
    get_tokens,
    get_pos_tags,
    lemmatize_tokens,
    get_token_text,
    chunk_NEs,
    remove_non_alnum,
    lower,
    remove_stopwords,
]

# Dictionary to hold function names and their input/output types
function_input_output_types: Dict[str, Tuple[Tuple[type, ...], type]] = {}

# Populate the dictionary with function names and their input/output types
for func in functions:
    input_types, output_type = _extract_input_output_types(func)
    function_input_output_types[func.__name__] = (input_types, output_type)

print(function_input_output_types)


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


def _generate_valid_permutations() -> List[List[Callable]]:
    valid_permutations = []
    for n in range(1, len(functions) + 1):
        for perm in itertools.permutations(functions, n):
            if _is_valid_permutation(perm):
                valid_permutations.append(perm)
    return valid_permutations


VALID_PERMUTATIONS = _generate_valid_permutations()
