import re
import unicodedata

from engine.porter import stem

STOPWORDS = set("""
a about above after again against all also am an and any are as at be because
been before being below between both but by can cannot could did do does doing
down during each et al etc few for from further had has have having he her here
hers herself him himself his how however i if in into is it its itself just me
more most my myself no nor not of off on once only or other our ours ourselves
out over own same she should so some such than that the their theirs them
themselves then there these they this those through thus to too under until up
upon us use used using very via was we were what when where which while who whom
why will with within without would yet you your yours yourself yourselves
fig figure table eq section paper doi http https www org com vol pp
""".split())

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
REFERENCES_PATTERN = re.compile(
    r"\n\s*(?:\d+\.?\s*)?(references|bibliography|literature cited)\s*\n",
    re.IGNORECASE
)


def clean_text(text):
    # NFKC also expands PDF ligatures like "ﬁ" into "fi".
    text = unicodedata.normalize("NFKC", text)

    # PDFs break words across lines: "effi-\nciency" -> "efficiency"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # The reference list adds lots of author names and venue words that are
    # not about the paper's topic, so cut it if it is near the end.
    matches = list(REFERENCES_PATTERN.finditer(text))
    if matches and matches[-1].start() > 0.6 * len(text):
        text = text[:matches[-1].start()]

    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


def normalize(token):
    if token in STOPWORDS or len(token) < 2 or len(token) > 30:
        return None
    # Keep 4-digit years, drop page numbers and other stray numbers.
    if token.isdigit() and len(token) != 4:
        return None
    if token.isalpha():
        return stem(token)
    return token


def analyze(text):
    # Positions count every token, including dropped stopwords, so that
    # phrase queries still need the words to be adjacent in the original text.
    terms = []
    for position, token in enumerate(tokenize(text)):
        term = normalize(token)
        if term:
            terms.append((term, position))
    return terms


def get_terms(text):
    return [term for term, _ in analyze(text)]
