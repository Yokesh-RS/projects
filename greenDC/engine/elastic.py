import json
import os

import requests
from dotenv import load_dotenv

# Read the Elasticsearch connection details from the .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

ELASTIC_URL = os.getenv("ELASTICSEARCH_BASE_URL", "").rstrip("/")
ELASTIC_USERNAME = os.getenv("ELASTICSEARCH_USERNAME", "")
ELASTIC_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD", "")
# The shared cluster only lets each user create indexes that start with
# their own username, so the index is called "<username>-greendc"
ELASTIC_INDEX = ELASTIC_USERNAME + "-greendc"


def is_configured():
    return ELASTIC_URL != ""


def send(method, path, body=None):
    response = requests.request(method, ELASTIC_URL + path, json=body,
                                auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD), timeout=60)
    response.raise_for_status()
    return response.json()


def send_bulk(lines):
    # The bulk API takes plain text with one JSON object per line
    response = requests.post(ELASTIC_URL + "/_bulk?refresh=true", data=lines.encode("utf-8"),
                             headers={"Content-Type": "application/x-ndjson"},
                             auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD), timeout=60)
    response.raise_for_status()
    return response.json()


def create_index():
    # Delete the old index if there is one, so we always start fresh
    try:
        send("DELETE", "/" + ELASTIC_INDEX)
    except requests.HTTPError:
        pass

    # The "english" analyzer lowercases, removes stop words and stems,
    # like our own preprocessing does
    send("PUT", "/" + ELASTIC_INDEX, {
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "title": {"type": "text", "analyzer": "english"},
                "text": {"type": "text", "analyzer": "english"}
            }
        }
    })


def add_papers(papers):
    # The bulk API takes two lines per paper: an action line and the paper itself
    lines = ""
    for paper in papers:
        lines += json.dumps({"index": {"_index": ELASTIC_INDEX, "_id": paper["doc_id"]}}) + "\n"
        lines += json.dumps(paper) + "\n"

    result = send_bulk(lines)
    if result["errors"]:
        print("Some papers could not be added:", result["items"][:2])


def count_papers():
    return send("GET", "/" + ELASTIC_INDEX + "/_count")["count"]


def search(query, size=200):
    # Elasticsearch ranks with BM25 by default. Title matches count double.
    result = send("POST", "/" + ELASTIC_INDEX + "/_search", {
        "size": size,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title^2", "text"]
            }
        }
    })

    scores = {}
    for hit in result["hits"]["hits"]:
        scores[hit["_id"]] = hit["_score"]
    return scores
