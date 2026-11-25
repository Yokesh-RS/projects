import re

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stemmer = PorterStemmer()
stem_cache = {}


def clean_text(text):
    # Join words that the PDF split across two lines, e.g. "effi-\nciency"
    text = text.replace("-\n", "")

    # Remove the reference list at the end of the paper
    position = text.lower().rfind("\nreferences")
    if position > len(text) * 0.6:
        text = text[:position]

    # Replace newlines and repeated spaces with single spaces
    return " ".join(text.split())


def stem(word):
    # Stemming is slow, so remember words we have already stemmed
    if word not in stem_cache:
        stem_cache[word] = stemmer.stem(word)
    return stem_cache[word]


def get_terms(text):
    words = re.findall(r"[a-z0-9]+", text.lower())

    terms = []
    for word in words:
        if word in ENGLISH_STOP_WORDS:
            continue
        if len(word) < 2:
            continue
        # Keep years like 2023 but skip page numbers and other numbers
        if word.isdigit() and len(word) != 4:
            continue
        terms.append(stem(word))

    return terms
