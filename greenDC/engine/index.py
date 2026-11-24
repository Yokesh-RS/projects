import gzip
import json
import pickle

from engine.preprocess import analyze, clean_text

FIELDS = ["title", "body"]


class InvertedIndex:

    def __init__(self):
        # postings[field][term] = {doc_number: [positions]}
        self.postings = {"title": {}, "body": {}}
        self.doc_length = {"title": [], "body": []}
        self.documents = []
        self.texts = []

    def add_document(self, document, body):
        doc_number = len(self.documents)
        self.documents.append(document)
        self.texts.append(body)

        for field, text in [("title", document["title"]), ("body", body)]:
            terms = analyze(text)
            self.doc_length[field].append(len(terms))

            for term, position in terms:
                self.postings[field].setdefault(term, {}).setdefault(doc_number, []).append(position)

    def finish(self):
        self.N = len(self.documents)
        self.avg_length = {
            field: sum(self.doc_length[field]) / self.N for field in FIELDS
        }
        self.vocabulary = set(self.postings["title"]) | set(self.postings["body"])

    def docs_with_term(self, term):
        return set(self.postings["title"].get(term, {})) | set(self.postings["body"].get(term, {}))

    def save(self, path):
        with gzip.open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        with gzip.open(path, "rb") as file:
            return pickle.load(file)


def build_index(corpus_file, catalogue_file):
    index = InvertedIndex()

    with open(catalogue_file, "r", encoding="utf-8") as file:
        catalogue = json.load(file)

    with open(corpus_file, "r", encoding="utf-8") as file:
        for line in file:
            paper = json.loads(line)

            document = {
                "doc_id": paper["document_id"],
                "title": paper.get("title") or "",
                "authors": paper.get("authors", []),
                "year": paper.get("year"),
                "url": paper.get("pdf_url"),
                "topics": catalogue.get(paper["document_id"], {}).get("metadata", {}).get("topics", [])
            }

            index.add_document(document, clean_text(paper.get("full_text") or ""))

    index.finish()
    return index
