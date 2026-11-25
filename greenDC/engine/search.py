import html
import json
import os

import numpy as np

from engine.index import Index
from engine.llm import generate
from engine import elastic
from engine.models import (BM25Model, BooleanModel, CombinedModel, ElasticModel, LanguageModel,
                           SemanticModel, TFIDFModel)
from engine.preprocess import get_terms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_FILE = os.path.join(BASE_DIR, "data", "corpus.jsonl")
PASSAGES_FILE = os.path.join(BASE_DIR, "data", "passages.json")
VECTORS_FILE = os.path.join(BASE_DIR, "data", "passage_vectors.npy")

# Sentences with these words are usually author details, licences or links,
# not the actual content of the paper.
FRONT_MATTER_WORDS = [
    "university", "department", "college", "institute", "school", "laboratory",
    "licen", "creative commons", "copyright", "http", "www.", "@", "doi", "received",
    "keywords"
]


class SearchEngine:

    def __init__(self):
        self.index = Index(CORPUS_FILE)

        tfidf = TFIDFModel(self.index)
        bm25 = BM25Model(self.index)
        lm = LanguageModel(self.index)

        self.models = {
            "boolean": BooleanModel(self.index),
            "tfidf": tfidf,
            "bm25": bm25,
            "lm": lm,
            "combined": CombinedModel(self.index, [tfidf, bm25, lm])
        }
        self.default_model = "combined"

        # Elasticsearch is only used if its address is in .env
        if elastic.is_configured():
            self.models["elastic"] = ElasticModel(self.index)

        # Semantic search and answers need the passage embeddings made by
        # build_embeddings.py. Without them only the keyword models work.
        self.semantic = None
        if os.path.exists(VECTORS_FILE):
            with open(PASSAGES_FILE, "r", encoding="utf-8") as file:
                passages = json.load(file)
            vectors = np.load(VECTORS_FILE)

            self.semantic = SemanticModel(self.index, passages, vectors)
            self.models["semantic"] = self.semantic
            self.models["hybrid"] = CombinedModel(self.index, [tfidf, bm25, lm, self.semantic])
            self.default_model = "hybrid"

            # Hybrid with Elasticsearch added scored best in the evaluation
            if "elastic" in self.models:
                ranking_lists = [tfidf, bm25, lm, self.semantic, self.models["elastic"]]
                self.models["hybrid_es"] = CombinedModel(self.index, ranking_lists)
                self.default_model = "hybrid_es"

    def search(self, query, model_name=None, top_k=10, offset=0):
        if model_name is None:
            model_name = self.default_model

        scores = self.models[model_name].score(query)
        ranked = sorted(scores, key=scores.get, reverse=True)
        query_terms = set(get_terms(query))

        results = []
        for doc in ranked[offset:offset + top_k]:
            paper = self.index.documents[doc]
            text = self.index.texts[doc]

            results.append({
                "doc_id": paper["doc_id"],
                "title": paper["title"],
                "authors": paper["authors"],
                "year": paper["year"],
                "url": paper["url"],
                "score": round(float(scores[doc]), 4),
                "title_html": highlight(paper["title"], query_terms),
                "snippet_html": highlight(make_snippet(text, paper, query_terms), query_terms)
            })

        # The summary on the right side is only shown on the first page
        summary = None
        if offset == 0 and ranked:
            summary = self.make_summary(ranked[:3], query_terms)

        return {
            "query": query,
            "model": model_name,
            "total_hits": len(ranked),
            "results": results,
            "summary": summary
        }

    def answer(self, query, top_n=3):
        # Generate a short answer from the best passage of each of the top papers
        if self.semantic is None:
            return {"answer": "", "sources": []}

        scores = self.models[self.default_model].score(query)
        ranked = sorted(scores, key=scores.get, reverse=True)[:top_n]
        if not ranked:
            return {"answer": "", "sources": []}

        sources = []
        context = ""
        for number in range(len(ranked)):
            doc = ranked[number]
            paper = self.index.documents[doc]
            passage = self.semantic.best_passage(query, doc)
            context = context + "[" + str(number + 1) + "] " + paper["title"] + "\n" + passage + "\n\n"
            sources.append({"number": number + 1, "doc_id": paper["doc_id"], "title": paper["title"]})

        prompt = (
            "Answer the question using only the sources below. "
            "Write 3 to 5 sentences in plain language. "
            "After each fact, cite the source number in square brackets, like [1]. "
            # The model once turned "626 billion litres in 2014" into "per day",
            # so numbers must be copied exactly as the source gives them.
            "Copy every number exactly with the same unit and time period as in the source. "
            "Do not add words like 'per day' or 'per year' unless the source says them. "
            "If the sources do not answer the question, say so.\n\n"
            "Sources:\n" + context +
            "Question: " + query + "\n\nAnswer:"
        )

        return {"answer": generate(prompt), "sources": sources}

    def make_summary(self, top_docs, query_terms):
        first = top_docs[0]
        abstract = get_abstract(self.index.texts[first], self.index.documents[first])

        key_points = []
        for doc in top_docs:
            text = self.index.texts[doc]
            paper = self.index.documents[doc]
            sentence = best_sentence(text, paper, query_terms, abstract)
            if sentence:
                key_points.append({
                    "doc_id": self.index.documents[doc]["doc_id"],
                    "sentence_html": highlight(sentence, query_terms)
                })

        return {
            "abstract_html": highlight(abstract, query_terms),
            "key_points": key_points
        }


def content_start(text, paper):
    # Skip the journal header, title and authors at the top of the paper.
    # Start after the word "Abstract", or after the title and authors if there is none.
    lower = text.lower()
    title = paper["title"]

    position = lower.find("abstract", 0, len(text) // 3)
    if position != -1:
        return position + len("abstract")

    position = lower.find(title.lower()[:40])
    if position == -1 or position > 1000:
        return 0
    position = position + len(title)

    # The authors usually come right after the title
    if paper["authors"]:
        last_author = paper["authors"][-1]
        found = text.find(last_author, position, position + 300)
        if found != -1:
            position = found + len(last_author)

    # Skip marks like "1✉" until the first letter
    while position < len(text) and not text[position].isalpha():
        position = position + 1

    return position


def split_passages(text, paper, size=150):
    # The title on its own, then the paper in chunks of about 150 words
    words = text[content_start(text, paper):].split(" ")

    passages = [paper["title"]]
    for start in range(0, len(words), size):
        passages.append(" ".join(words[start:start + size]))
    return passages


def split_sentences(text):
    sentences = []
    for part in text.split(". "):
        part = part.strip(" :.-")
        if part:
            sentences.append(part + ".")
    return sentences


def is_front_matter(sentence):
    lower = sentence.lower()
    for word in FRONT_MATTER_WORDS:
        if word in lower:
            return True
    return False


def get_abstract(text, paper, max_length=1000):
    text = text[content_start(text, paper):]

    abstract = ""
    for sentence in split_sentences(text):
        lower = sentence.lower()
        if "keywords" in lower or "introduction" in lower:
            break
        if is_front_matter(sentence):
            continue

        abstract = abstract + sentence + " "
        if len(abstract) > max_length:
            break

    return abstract.strip()


def best_sentence(text, paper, query_terms, skip_text=""):
    # The sentence that contains the most query words
    text = text[content_start(text, paper):]

    best = ""
    best_count = 0
    for sentence in split_sentences(text):
        if len(sentence) < 60 or len(sentence) > 400:
            continue
        if is_front_matter(sentence):
            continue
        if sentence in skip_text:
            continue

        count = len(query_terms & set(get_terms(sentence)))
        if count > best_count:
            best = sentence
            best_count = count

    return best


def make_snippet(text, paper, query_terms, size=40):
    # The 40 words in a row that contain the most query words
    words = text[content_start(text, paper):].split(" ")

    word_terms = []
    for word in words:
        terms = get_terms(word)
        if terms:
            word_terms.append(terms[0])
        else:
            word_terms.append("")

    best_start = 0
    best_count = 0
    for i in range(len(words)):
        if word_terms[i] not in query_terms:
            continue
        count = len(query_terms & set(word_terms[i:i + size]))
        if count > best_count:
            best_start = i
            best_count = count

    return " ".join(words[best_start:best_start + size])


def highlight(text, query_terms):
    # Put <mark> around the query words. The text is escaped first so that
    # nothing in a paper can be read as HTML by the browser.
    words = []
    for word in text.split(" "):
        core = word.strip(".,;:!?()[]\"'")
        terms = get_terms(core)

        if core and terms and terms[0] in query_terms:
            start = word.find(core)
            before = word[:start]
            after = word[start + len(core):]
            words.append(html.escape(before) + "<mark>" + html.escape(core) + "</mark>" + html.escape(after))
        else:
            words.append(html.escape(word))

    return " ".join(words)
