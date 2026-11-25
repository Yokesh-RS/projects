import json
from collections import Counter

from engine.preprocess import clean_text, get_terms


class Index:

    def __init__(self, corpus_file):
        self.documents = []           # title, authors, year, url of each paper
        self.texts = []               # cleaned full text of each paper
        self.doc_terms = []           # list of terms in each paper
        self.term_counts = []         # how often each term appears in each paper
        self.title_terms = []         # set of terms in each title
        self.postings = {}            # term -> set of papers that contain it
        self.collection_counts = Counter()

        with open(corpus_file, "r", encoding="utf-8") as file:
            for line in file:
                paper = json.loads(line)
                self.add_paper(paper)

        self.N = len(self.documents)
        self.total_terms = sum(self.collection_counts.values())

    def add_paper(self, paper):
        doc = len(self.documents)
        title = paper.get("title") or ""
        text = clean_text(paper.get("full_text") or "")

        self.documents.append({
            "doc_id": paper["document_id"],
            "title": title,
            "authors": paper.get("authors", []),
            "year": paper.get("year"),
            "url": paper.get("pdf_url")
        })
        self.texts.append(text)

        terms = get_terms(title + " " + text)
        self.doc_terms.append(terms)
        self.term_counts.append(Counter(terms))
        self.title_terms.append(set(get_terms(title)))
        self.collection_counts.update(terms)

        for term in set(terms):
            if term not in self.postings:
                self.postings[term] = set()
            self.postings[term].add(doc)
