import os
import time

from engine.index import build_index
from engine.search import INDEX_FILE

CORPUS_FILE = "data/corpus.jsonl"
CATALOGUE_FILE = "data/documents_catalogue.json"


if __name__ == "__main__":
    start = time.time()

    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    index = build_index(CORPUS_FILE, CATALOGUE_FILE)
    index.save(INDEX_FILE)

    print("Index built in", round(time.time() - start, 1), "seconds")
    print("Documents:", index.N)
    print("Vocabulary size:", len(index.vocabulary))
    print("Average body length:", round(index.avg_length["body"]), "terms")
    print("Saved to:", INDEX_FILE)
