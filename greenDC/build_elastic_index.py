from engine import elastic
from engine.index import Index
from engine.search import CORPUS_FILE

BATCH_SIZE = 20

if not elastic.is_configured():
    print("Add ELASTICSEARCH_BASE_URL, ELASTICSEARCH_USERNAME and ELASTICSEARCH_PASSWORD to .env first.")
    exit()

# Use the same cleaned text as our own index, so the comparison is fair
index = Index(CORPUS_FILE)

print("Creating index", elastic.ELASTIC_INDEX, "at", elastic.ELASTIC_URL)
elastic.create_index()

papers = []
for doc in range(index.N):
    papers.append({
        "doc_id": index.documents[doc]["doc_id"],
        "title": index.documents[doc]["title"],
        "text": index.texts[doc]
    })

for i in range(0, len(papers), BATCH_SIZE):
    elastic.add_papers(papers[i:i + BATCH_SIZE])
    print("  added", min(i + BATCH_SIZE, len(papers)), "/", len(papers))

print("Papers in Elasticsearch:", elastic.count_papers())
