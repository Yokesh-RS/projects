import json
import os

CORPUS_FILE = "data/corpus.jsonl"
CATALOGUE_FILE = "data/documents_catalogue.json"
PDF_FOLDER = "data/pdfs"

problems = []

# Load the corpus
documents = []
with open(CORPUS_FILE, "r", encoding="utf-8") as file:
    for line in file:
        documents.append(json.loads(line))

print("Documents in corpus:", len(documents))

# Check each document
seen_ids = set()
seen_openalex_ids = set()

for document in documents:
    doc_id = document.get("document_id")

    if doc_id in seen_ids:
        problems.append(doc_id + ": duplicate document id")
    seen_ids.add(doc_id)

    openalex_id = document.get("openalex_id")
    if openalex_id in seen_openalex_ids:
        problems.append(doc_id + ": duplicate OpenAlex paper")
    seen_openalex_ids.add(openalex_id)

    if not document.get("title"):
        problems.append(doc_id + ": missing title")

    if len(document.get("full_text") or "") < 1000:
        problems.append(doc_id + ": text is too short")

    if not os.path.exists(os.path.join(PDF_FOLDER, doc_id + ".pdf")):
        problems.append(doc_id + ": PDF file is missing")

# Check that the catalogue has the same documents as the corpus
with open(CATALOGUE_FILE, "r", encoding="utf-8") as file:
    catalogue = json.load(file)

print("Documents in catalogue:", len(catalogue))

for doc_id in seen_ids:
    if doc_id not in catalogue:
        problems.append(doc_id + ": missing from the catalogue")

for doc_id in catalogue:
    if doc_id not in seen_ids:
        problems.append(doc_id + ": in the catalogue but not in the corpus")

# Check for PDFs that don't belong to any document
for file_name in os.listdir(PDF_FOLDER):
    if file_name.endswith(".pdf") and file_name[:-4] not in seen_ids:
        problems.append(file_name + ": PDF does not belong to any document")

# Report
print()
if problems:
    print("Problems found:", len(problems))
    for problem in problems:
        print(" -", problem)
    print("\nFix the problems above, then run build_catalog.py again.")
else:
    print("No problems found. Corpus, PDFs and catalogue match.")
