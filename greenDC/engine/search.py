import argparse
import html
import os
import re
import time
from collections import Counter

from engine.index import InvertedIndex
from engine.models import MODELS
from engine.preprocess import get_terms, normalize

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_FILE = os.path.join(BASE_DIR, "data", "index", "index.pkl.gz")


def word_matches(word, query_terms):
    return normalize(re.sub(r"\W", "", word.lower())) in query_terms


def highlight(text, query_terms):
    # Escape first so paper text can't inject HTML into the page.
    words = []
    for word in text.split(" "):
        if word_matches(word, query_terms):
            # Leave punctuation like "cooling," outside the highlight.
            before, core, after = re.match(r"^(\W*)(.*?)(\W*)$", word).groups()
            words.append(html.escape(before) + "<mark>" + html.escape(core) + "</mark>" + html.escape(after))
        else:
            words.append(html.escape(word))
    return " ".join(words)


class SearchEngine:

    def __init__(self, index_file=INDEX_FILE):
        self.index = InvertedIndex.load(index_file)
        self.models = {}

    def search(self, query, model_name="bm25", top_k=10, offset=0,
               topic=None, year_from=None, year_to=None):
        start = time.time()

        # Models are slow to set up (idf, norms), so build each one once.
        if model_name not in self.models:
            self.models[model_name] = MODELS[model_name](self.index)

        scores = self.models[model_name].score(query)
        # Ties (every Boolean match scores 1.0) fall back to document order.
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

        def year_ok(doc):
            year = self.index.documents[doc]["year"] or 0
            return (not year_from or year >= year_from) and (not year_to or year <= year_to)

        def topic_ok(doc):
            return not topic or topic in self.index.documents[doc]["topics"]

        # Each facet counts hits with the *other* filter applied, so picking a
        # topic doesn't hide the counts for the remaining topics.
        topic_counts = Counter()
        year_counts = Counter()
        for doc, _ in ranked:
            if year_ok(doc):
                topic_counts.update(self.index.documents[doc]["topics"])
            if topic_ok(doc):
                year_counts[self.index.documents[doc]["year"]] += 1

        ranked = [(doc, score) for doc, score in ranked if year_ok(doc) and topic_ok(doc)]
        query_terms = set(get_terms(re.sub(r"\b(AND|OR|NOT)\b", " ", query)))

        results = []
        for rank, (doc, score) in enumerate(ranked[offset:offset + top_k], offset + 1):
            result = dict(self.index.documents[doc])
            result["rank"] = rank
            result["score"] = round(float(score), 4)
            result["snippet"] = self.make_snippet(doc, query_terms)
            results.append(result)

        return {
            "query": query,
            "model": model_name,
            "query_terms": sorted(query_terms),
            "total_hits": len(ranked),
            "time_ms": round((time.time() - start) * 1000, 1),
            "topics": topic_counts.most_common(),
            "years": sorted((y, c) for y, c in year_counts.items() if y),
            "results": results
        }

    def make_snippet(self, doc, query_terms, size=40):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("-m", "--model", default="bm25", choices=list(MODELS))
    parser.add_argument("-k", "--top", type=int, default=10)
    args = parser.parse_args()

    engine = SearchEngine()
    response = engine.search(args.query, args.model, args.top)
    query_terms = set(response["query_terms"])

    print(f"\n{response['total_hits']} hits in {response['time_ms']} ms ({args.model})\n")
    for result in response["results"]:
        snippet = " ".join(
            "\033[1;33m" + w + "\033[0m" if word_matches(w, query_terms) else w
            for w in result["snippet"].split(" ")
        )
        print(f"{result['rank']}. [{result['doc_id']}] {result['title']} ({result['year']})  score={result['score']}")
        print("   " + snippet + "\n")
