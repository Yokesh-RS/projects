import json
import time

import numpy as np

from engine.index import Index
from engine.llm import EMBEDDING_MODEL, embed
from engine.search import CORPUS_FILE, PASSAGES_FILE, VECTORS_FILE, split_passages

BATCH_SIZE = 32

start = time.time()
index = Index(CORPUS_FILE)

# Split every paper into passages of about 150 words
passages = []
for doc in range(index.N):
    paper = index.documents[doc]
    for text in split_passages(index.texts[doc], paper):
        passages.append({"doc_id": paper["doc_id"], "text": text})

print("Papers:", index.N)
print("Passages:", len(passages))
print("Embedding with", EMBEDDING_MODEL, "...")

vectors = []
for i in range(0, len(passages), BATCH_SIZE):
    batch = passages[i:i + BATCH_SIZE]
    texts = []
    for passage in batch:
        texts.append(passage["text"])
    vectors.extend(embed(texts))

    if (i // BATCH_SIZE) % 20 == 0:
        print("  ", i + len(batch), "/", len(passages))

# Save the vectors (one row per passage) and the passages in the same order
np.save(VECTORS_FILE, np.array(vectors, dtype=np.float32))
with open(PASSAGES_FILE, "w", encoding="utf-8") as file:
    json.dump(passages, file, ensure_ascii=False)

print("Done in", round(time.time() - start), "seconds")
print("Saved", VECTORS_FILE, "and", PASSAGES_FILE)
