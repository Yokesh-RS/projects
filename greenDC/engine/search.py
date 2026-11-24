import argparse
import os
import re
import time

from engine.index import InvertedIndex
from engine.models import MODELS
from engine.preprocess import get_terms, normalize

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_FILE = os.path.join(BASE_DIR, "data", "index", "index.pkl.gz")


class SearchEngine:

    def __init__(self, index_file=INDEX_FILE):
        self.index = InvertedIndex.load(index_file)
        self.models = {}

    def search(self, query, model_name="bm25", top_k=10):
        start = time.time()

        # Models are slow to set up (idf, norms), so build each one once.
        if model_name not in self.models:
            self.models[model_name] = MODELS[model_name](self.index)

        scores = self.models[model_name].score(query)
        # Ties (every Boolean match scores 1.0) fall back to document order.
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

        query_terms = set(get_terms(re.sub(r"\b(AND|OR|NOT)\b", " ", query)))

        results = []
        for doc, score in ranked[:top_k]:
            result = dict(self.index.documents[doc])
            result["score"] = round(score, 4)
            result["snippet"] = self.make_snippet(doc, query_terms)
            results.append(result)

        return {
            "query": query,
            "model": model_name,
            "total_hits": len(scores),
            "time_ms": round((time.time() - start) * 1000, 1),
            "results": results
        }

    def make_snippet(self, doc, query_terms, size=35):
        # Show the window of words that contains the most distinct query terms.
        words = self.index.texts[doc].split(" ")
        terms = [normalize(re.sub(r"\W", "", w.lower())) for w in words]

        best_start, best_count = 0, 0
        for i in range(len(words)):
            if terms[i] in query_terms:
                count = len(set(terms[i:i + size]) & query_terms)
                if count > best_count:
                    best_start, best_count = i, count

        start = max(0, best_start - 5)
        return " ".join(words[start:start + size])


def highlight(text, query_terms):
    words = []
    for word in text.split(" "):
        if normalize(re.sub(r"\W", "", word.lower())) in query_terms:
            word = "\033[1;33m" + word + "\033[0m"
        words.append(word)
    return " ".join(words)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("-m", "--model", default="bm25", choices=list(MODELS))
    parser.add_argument("-k", "--top", type=int, default=10)
    args = parser.parse_args()

    engine = SearchEngine()
    response = engine.search(args.query, args.model, args.top)
    query_terms = set(get_terms(args.query))

    print(f"\n{response['total_hits']} hits in {response['time_ms']} ms ({args.model})\n")
    for rank, result in enumerate(response["results"], 1):
        print(f"{rank}. [{result['doc_id']}] {result['title']} ({result['year']})  score={result['score']}")
        print("   " + highlight(result["snippet"], query_terms) + "\n")
