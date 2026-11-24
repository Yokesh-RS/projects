import math

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


class LanguageModel:
    # Query likelihood: rank documents by log P(query | document).
    # Smoothing mixes in the whole-collection probability so a document
    # missing one query word doesn't get probability zero.
    #   dirichlet: P(w|d) = (tf + mu * P(w|C)) / (|d| + mu)
    #   jm:        P(w|d) = (1 - lam) * tf / |d| + lam * P(w|C)

    def __init__(self, index, smoothing="dirichlet", mu=2000, lam=0.7):
        self.index = index
        self.smoothing = smoothing
        self.mu = mu
        self.lam = lam

        self.doc_length = [
            index.doc_length["title"][d] + index.doc_length["body"][d] for d in range(index.N)
        ]
        self.collection_length = sum(self.doc_length)

    def term_frequencies(self, term):
        tf = {}
        for field in ["title", "body"]:
            for doc, positions in self.index.postings[field].get(term, {}).items():
                tf[doc] = tf.get(doc, 0) + len(positions)
        return tf

    def score(self, query):
        query_terms = [t for t in get_terms(query) if t in self.index.vocabulary]
        tf = {term: self.term_frequencies(term) for term in set(query_terms)}

        # Only score documents that contain at least one query word.
        candidates = set()
        for docs in tf.values():
            candidates |= set(docs)

        scores = {}
        for doc in candidates:
            length = self.doc_length[doc]
            total = 0.0
            for term in query_terms:
                count = tf[term].get(doc, 0)
                p_collection = sum(tf[term].values()) / self.collection_length

                if self.smoothing == "jm":
                    p = (1 - self.lam) * count / length + self.lam * p_collection
                else:
                    p = (count + self.mu * p_collection) / (length + self.mu)
                total += math.log(p)
            scores[doc] = total

        return scores


MODELS = {
    "boolean": BooleanModel,
    "tfidf": TFIDFModel,
    "bm25": BM25Model,
    "lm": LanguageModel,
    "lm_jm": lambda index: LanguageModel(index, smoothing="jm")
}
