from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from engine.preprocess import get_terms, normalize


def document_texts(index):
    return [doc["title"] + " " + text for doc, text in zip(index.documents, index.texts)]


class BooleanModel:
    # Simple AND / OR / NOT, read left to right. Words with no operator
    # between them are treated as AND.

    def __init__(self, index):
        self.index = index
        self.all_docs = set(range(index.N))

    def score(self, query):
        result = None
        operator = "AND"

        for word in query.split():
            if word in ("AND", "OR", "NOT"):
                operator = word
                continue

            term = normalize(word.lower())
            if term is None:
                continue
            docs = self.index.docs_with_term(term)

            if result is None:
                result = self.all_docs - docs if operator == "NOT" else docs
            elif operator == "OR":
                result = result | docs
            elif operator == "NOT":
                result = result - docs
            else:
                result = result & docs
            operator = "AND"

        return {doc: 1.0 for doc in result or []}


class TFIDFModel:

    def __init__(self, index):
        # Use our own analyzer so TF-IDF sees the same stemmed terms as the index.
        self.vectorizer = TfidfVectorizer(analyzer=get_terms, sublinear_tf=True)
        self.doc_vectors = self.vectorizer.fit_transform(document_texts(index))

    def score(self, query):
        query_vector = self.vectorizer.transform([query])
        # Vectors are L2-normalised, so the dot product is the cosine similarity.
        similarities = linear_kernel(query_vector, self.doc_vectors)[0]
        return {doc: s for doc, s in enumerate(similarities) if s > 0}


class BM25Model:

    def __init__(self, index, k1=1.2, b=0.75):
        self.bm25 = BM25Okapi([get_terms(text) for text in document_texts(index)], k1=k1, b=b)

    def score(self, query):
        scores = self.bm25.get_scores(get_terms(query))
        return {doc: s for doc, s in enumerate(scores) if s > 0}


MODELS = {
    "boolean": BooleanModel,
    "tfidf": TFIDFModel,
    "bm25": BM25Model
}
