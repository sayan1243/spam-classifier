from functools import lru_cache
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


_TOKENIZER_RESOURCE = "tokenizers/punkt_tab/english/"
_STOPWORDS_RESOURCE = "corpora/stopwords"
_STEMMER = PorterStemmer()


def ensure_nltk_resources() -> None:
    resources = (
        ("punkt_tab", _TOKENIZER_RESOURCE),
        ("stopwords", _STOPWORDS_RESOURCE),
    )
    for package, resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            if not nltk.download(package, quiet=True):
                raise LookupError(
                    f"Could not download the NLTK {package!r} data required "
                    "for SMS preprocessing."
                )
            nltk.data.find(resource)


@lru_cache(maxsize=1)
def _english_stopwords() -> frozenset[str]:
    return frozenset(stopwords.words("english"))


def transform_text(text: str) -> str:
    tokens = nltk.word_tokenize(text.lower())
    stop_words = _english_stopwords()
    words = (
        token
        for token in tokens
        if token.isalnum()
        and token not in stop_words
        and token not in string.punctuation
    )
    return " ".join(_STEMMER.stem(word) for word in words)
