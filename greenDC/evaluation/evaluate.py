import json
import math
import os
import sys

# Run from the greenDC folder: python evaluation/evaluate.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.search import SearchEngine

QRELS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qrels.json")
MODELS = ["tfidf", "bm25", "lm", "combined"]


# relevant is a dict like {"GD44": 2, "GD167": 1}
# 2 = highly relevant, 1 = partly relevant, missing = not relevant

def precision_at_k(ranked, relevant, k):
    found = 0
    for doc_id in ranked[:k]:
        if doc_id in relevant:
            found = found + 1
    return found / k


def recall_at_k(ranked, relevant, k):
    found = 0
    for doc_id in ranked[:k]:
        if doc_id in relevant:
            found = found + 1
    return found / len(relevant)


def average_precision(ranked, relevant):
    # Average of the precision at each position where a relevant paper is found
    found = 0
    total = 0
    for position in range(len(ranked)):
        if ranked[position] in relevant:
            found = found + 1
            total = total + found / (position + 1)
    return total / len(relevant)


def ndcg_at_k(ranked, relevant, k):
    # Highly relevant papers count more, and papers lower down count less
    dcg = 0
    for position in range(min(k, len(ranked))):
        grade = relevant.get(ranked[position], 0)
        dcg = dcg + (2 ** grade - 1) / math.log2(position + 2)

    # The best possible DCG: all relevant papers at the top, best ones first
    best_grades = sorted(relevant.values(), reverse=True)
    ideal = 0
    for position in range(min(k, len(best_grades))):
        ideal = ideal + (2 ** best_grades[position] - 1) / math.log2(position + 2)

    return dcg / ideal


def get_ranking(engine, model_name, query):
    scores = engine.models[model_name].score(query)
    ranked = sorted(scores, key=scores.get, reverse=True)

    doc_ids = []
    for doc in ranked:
        doc_ids.append(engine.index.documents[doc]["doc_id"])
    return doc_ids


with open(QRELS_FILE, "r", encoding="utf-8") as file:
    qrels = json.load(file)

engine = SearchEngine()
print("Papers:", engine.index.N, "  Test queries:", len(qrels))

# The Ollama-based models are only there once build_embeddings.py has been run
if "semantic" in engine.models:
    MODELS = MODELS + ["semantic", "hybrid"]

# Elasticsearch is only there if ELASTICSEARCH_BASE_URL is set in .env
if "elastic" in engine.models:
    MODELS = ["elastic"] + MODELS + ["hybrid_es"]

totals = {}
per_query_ndcg = {}

for model in MODELS:
    totals[model] = {"P@5": 0, "P@10": 0, "R@10": 0, "MAP": 0, "nDCG@10": 0}
    per_query_ndcg[model] = {}

    for query_id in qrels:
        query = qrels[query_id]["query"]
        relevant = qrels[query_id]["relevant"]
        ranked = get_ranking(engine, model, query)

        ndcg = ndcg_at_k(ranked, relevant, 10)
        totals[model]["P@5"] += precision_at_k(ranked, relevant, 5)
        totals[model]["P@10"] += precision_at_k(ranked, relevant, 10)
        totals[model]["R@10"] += recall_at_k(ranked, relevant, 10)
        totals[model]["MAP"] += average_precision(ranked, relevant)
        totals[model]["nDCG@10"] += ndcg
        per_query_ndcg[model][query_id] = ndcg

# Average over all queries
print("\nModel       P@5    P@10   R@10   MAP    nDCG@10")
for model in MODELS:
    line = model.ljust(10)
    for metric in ["P@5", "P@10", "R@10", "MAP", "nDCG@10"]:
        average = totals[model][metric] / len(qrels)
        line = line + "  " + format(average, ".3f")
    print(line)

# Average nDCG@10 for each category of queries
categories = []
for query_id in qrels:
    if qrels[query_id]["category"] not in categories:
        categories.append(qrels[query_id]["category"])

print("\nnDCG@10 per category")
header = "Category".ljust(52)
for model in MODELS:
    header = header + model.ljust(10)
print(header)
for category in categories:
    line = category.ljust(52)
    for model in MODELS:
        total = 0
        count = 0
        for query_id in qrels:
            if qrels[query_id]["category"] == category:
                total = total + per_query_ndcg[model][query_id]
                count = count + 1
        line = line + format(total / count, ".3f").ljust(10)
    print(line)

# nDCG@10 for every query, to see where each model does well or badly
print("\nnDCG@10 per query")
header = "Query  "
for model in MODELS:
    header = header + model.ljust(10)
print(header + "Query text")
for query_id in qrels:
    line = query_id.ljust(7)
    for model in MODELS:
        line = line + format(per_query_ndcg[model][query_id], ".3f").ljust(10)
    print(line + qrels[query_id]["query"])
