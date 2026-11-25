import math

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from engine import elastic
from engine.llm import embed_query
from engine.preprocess import get_terms


def keep_terms(terms):
    # The documents are already split into terms, so TF-IDF can use them as they are
    return terms


class BooleanModel:
    # Supports AND, OR and NOT, read from left to right.
    # Two words with nothing between them are treated as AND.

    def __init__(self, index):
        self.index = index

    def score(self, query):
        all_docs = set(range(self.index.N))
        result = None
        operator = "AND"

        for word in query.split():
            if word in ["AND", "OR", "NOT"]:
                operator = word
                continue

            terms = get_terms(word)
            if not terms:
                continue
            docs = self.index.postings.get(terms[0], set())

            if result is None:
                if operator == "NOT":
                    result = all_docs - docs
                else:
                    result = docs
            elif operator == "OR":
                result = result | docs
            elif operator == "NOT":
                result = result - docs
            else:
                result = result & docs

            operator = "AND"

        scores = {}
        if result:
            for doc in result:
                scores[doc] = 1.0
        return scores


class TFIDFModel:

    def __init__(self, index):
        self.vectorizer = TfidfVectorizer(analyzer=keep_terms, sublinear_tf=True)
        self.doc_vectors = self.vectorizer.fit_transform(index.doc_terms)

    def score(self, query):
        query_vector = self.vectorizer.transform([get_terms(query)])
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]

        scores = {}
        for doc in range(len(similarities)):
            if similarities[doc] > 0:
                scores[doc] = similarities[doc]
        return scores


class BM25Model:

    def __init__(self, index):
        self.bm25 = BM25Okapi(index.doc_terms)

    def score(self, query):
        bm25_scores = self.bm25.get_scores(get_terms(query))

        scores = {}
        for doc in range(len(bm25_scores)):
            if bm25_scores[doc] > 0:
                scores[doc] = bm25_scores[doc]
        return scores


class LanguageModel:
    # Query likelihood with Dirichlet smoothing:
    # P(word | paper) = (count in paper + mu * P(word | all papers)) / (paper length + mu)
    # The score of a paper is the sum of log P(word | paper) over the query words.

    def __init__(self, index, mu=2000):
        self.index = index
        self.mu = mu

    def score(self, query):
        query_terms = []
        for term in get_terms(query):
            if term in self.index.postings:
                query_terms.append(term)

        # Only score papers that contain at least one of the query words
        candidates = set()
        for term in query_terms:
            candidates = candidates | self.index.postings[term]

        scores = {}
        for doc in candidates:
            length = len(self.index.doc_terms[doc])
            total = 0
            for term in query_terms:
                count = self.index.term_counts[doc][term]
                p_collection = self.index.collection_counts[term] / self.index.total_terms
                p = (count + self.mu * p_collection) / (length + self.mu)
                total = total + math.log(p)
            scores[doc] = total
        return scores


class ElasticModel:
    # Sends the query to Elasticsearch (which uses BM25) and returns its ranking.
    # Used as a baseline to compare our own models against.

    def __init__(self, index):
        self.doc_numbers = {}
        for doc in range(index.N):
            self.doc_numbers[index.documents[doc]["doc_id"]] = doc

    def score(self, query):
        if not query.strip():
            return {}

        # If the Elasticsearch server can't be reached, return nothing so the
        # other models in the hybrid ranking still work
        try:
            results = elastic.search(query)
        except Exception as error:
            print("Elasticsearch is not available:", error)
            return {}

        scores = {}
        for doc_id, score in results.items():
            if doc_id in self.doc_numbers:
                scores[self.doc_numbers[doc_id]] = score
        return scores


class SemanticModel:
    # Finds papers by meaning instead of exact words. Every paper is split into
    # passages, and each passage has an embedding vector (from build_embeddings.py).
    # A paper's score is the cosine similarity of its best passage to the query.

    def __init__(self, index, passages, vectors, top_n=50):
        self.index = index
        self.passages = passages
        self.top_n = top_n

        # Make every vector length 1, so a dot product is the cosine similarity
        lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = vectors / lengths

        # Which paper each passage belongs to, and which passages each paper has
        doc_numbers = {}
        for doc in range(index.N):
            doc_numbers[index.documents[doc]["doc_id"]] = doc

        self.passage_doc = []
        self.doc_passages = {}
        for i in range(len(passages)):
            doc = doc_numbers[passages[i]["doc_id"]]
            self.passage_doc.append(doc)
            if doc not in self.doc_passages:
                self.doc_passages[doc] = []
            self.doc_passages[doc].append(i)

        # Embedding a query is a call to Ollama, so remember recent ones
        self.query_cache = {}

    def similarities(self, query):
        if query not in self.query_cache:
            vector = np.array(embed_query(query), dtype=np.float32)
            self.query_cache[query] = vector / np.linalg.norm(vector)
        return self.vectors @ self.query_cache[query]

    def score(self, query):
        if not query.strip():
            return {}

        similarities = self.similarities(query)
        best = {}
        for i in range(len(similarities)):
            doc = self.passage_doc[i]
            if doc not in best or similarities[i] > best[doc]:
                best[doc] = float(similarities[i])

        # Every paper is a little similar to every query, so keep only the closest
        top_docs = sorted(best, key=best.get, reverse=True)[:self.top_n]
        scores = {}
        for doc in top_docs:
            scores[doc] = best[doc]
        return scores

    def best_passage(self, query, doc):
        similarities = self.similarities(query)
        best_index = self.doc_passages[doc][0]
        for i in self.doc_passages[doc]:
            if similarities[i] > similarities[best_index]:
                best_index = i
        return self.passages[best_index]["text"]


class CombinedModel:
    # Combines several rankings with Reciprocal Rank Fusion.
    # Each ranking gives a paper 1 / (60 + its rank), and these are added up.
    # So a paper that is near the top in most rankings ends up first.

    def __init__(self, index, models):
        self.index = index
        self.models = models

    def title_scores(self, query):
        # How many of the query words are in the title.
        # Needs at least half of them, otherwise one shared word is enough
        # to match (e.g. "e-waste" for the query "waste heat reuse").
        query_terms = set(get_terms(query))
        scores = {}
        if not query_terms:
            return scores

        for doc in range(self.index.N):
            matched = len(query_terms & self.index.title_terms[doc])
            if matched >= len(query_terms) / 2:
                scores[doc] = matched / len(query_terms)
        return scores

    def score(self, query):
        # If none of the query words are in any paper (e.g. "zzzqqq"), return
        # nothing. Otherwise semantic search would still return its closest papers.
        known_word = False
        for term in get_terms(query):
            if term in self.index.postings:
                known_word = True
        if not known_word:
            return {}

        all_scores = []
        for model in self.models:
            all_scores.append(model.score(query))
        all_scores.append(self.title_scores(query))

        final_scores = {}
        for scores in all_scores:
            ranked = sorted(scores, key=scores.get, reverse=True)
            for position in range(len(ranked)):
                doc = ranked[position]
                rank = position + 1
                final_scores[doc] = final_scores.get(doc, 0) + 1 / (60 + rank)
        return final_scores
