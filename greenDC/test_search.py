from engine.search import SearchEngine

# Each query lists words we expect to see in the titles of the top results.
# This is a quick sanity check, not a proper evaluation with relevance judgements.
SAMPLE_QUERIES = [
    ("liquid cooling in data centres", ["liquid", "cooling", "immersion"]),
    ("carbon aware workload scheduling", ["carbon", "scheduling"]),
    ("data centre water consumption", ["water"]),
    ("GPU energy consumption deep learning", ["gpu", "energy"]),
    ("renewable energy for data centres", ["renewable", "solar", "wind", "energy"]),
    ("power usage effectiveness PUE", ["power", "pue", "efficiency"]),
]

BOOLEAN_QUERIES = [
    "cooling AND water",
    "liquid OR immersion",
    "cooling NOT nuclear",
]

TOP_K = 5


def title_matches(title, keywords):
    title = title.lower()
    return any(keyword in title for keyword in keywords)


def show_results(response):
    print(f"\n[{response['model']}] {response['query']}  "
          f"({response['total_hits']} hits, {response['time_ms']} ms)")
    for rank, result in enumerate(response["results"], 1):
        print(f"  {rank}. {result['doc_id']:<6} {result['score']:<8} {result['title'][:80]}")


if __name__ == "__main__":
    engine = SearchEngine()
    print("Documents in index:", engine.index.N)

    print("\n========== Ranked retrieval ==========")
    summary = []
    for query, keywords in SAMPLE_QUERIES:
        for model in ["tfidf", "bm25", "lm", "lm_jm"]:
            response = engine.search(query, model, TOP_K)
            show_results(response)

            on_topic = sum(title_matches(r["title"], keywords) for r in response["results"])
            summary.append((query, model, on_topic))

    print("\n========== Boolean retrieval ==========")
    for query in BOOLEAN_QUERIES:
        show_results(engine.search(query, "boolean", TOP_K))

    # The two Boolean checks below follow directly from set logic,
    # so they should always hold.
    both = set(r["doc_id"] for r in engine.search("cooling AND water", "boolean", 1000)["results"])
    cooling = set(r["doc_id"] for r in engine.search("cooling", "boolean", 1000)["results"])
    without = set(r["doc_id"] for r in engine.search("cooling NOT water", "boolean", 1000)["results"])
    print("\nBoolean checks:")
    print("  'cooling AND water' is a subset of 'cooling':", both <= cooling)
    print("  AND + NOT results add up to 'cooling':", both | without == cooling and not both & without)

    print("\n========== Edge cases ==========")
    for query in ["the of and", "zzzqqq", ""]:
        response = engine.search(query, "bm25", TOP_K)
        print(f"  {query!r:15} -> {response['total_hits']} hits")

    print("\n========== Summary: on-topic titles in top", TOP_K, "==========")
    for query, model, on_topic in summary:
        print(f"  {model:<6} {on_topic}/{TOP_K}  {query}")
