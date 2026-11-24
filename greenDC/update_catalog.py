
import json
import os

CORPUS_PATH = "data/corpus.jsonl"
CATALOGUE_PATH = "data/documents_catalogue.json"


def update_catalogue():

    valid_documents = []

    # Step 1: Read the original corpus
    with open(CORPUS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            document = json.loads(line)

            pdf_path = document.get("document_path")

            # Step 2: Keep only documents with existing PDFs
            if pdf_path and os.path.exists(pdf_path):
                valid_documents.append(document)
            else:
                print("Removing missing document:", document.get("document_id"))

    # Step 3: Update corpus.jsonl
    with open(CORPUS_PATH, "w", encoding="utf-8") as file:
        for document in valid_documents:
            file.write(json.dumps(document, ensure_ascii=False) + "\n")

    # Step 4: Regenerate catalogue
    catalogue = {}

    for index, document in enumerate(valid_documents, start=1):

        doc_id = f"GD{index}"

        catalogue[doc_id] = {
            "metadata": {
                "name": document.get("title"),
                "authors": document.get("authors", []),
                "year": document.get("year"),
                "license": document.get("license"),
                "url": document.get("url"),
                "doi": document.get("doi"),
                "source": document.get("source"),
                "document_type": document.get("document_type"),
                "topics": document.get("topics", []),
                "document_path": document.get("document_path")
            },
            "full_text": document.get("full_text", "")
        }

    with open(CATALOGUE_PATH, "w", encoding="utf-8") as file:
        json.dump(catalogue, file, indent=4, ensure_ascii=False)

    print("\nCatalogue synchronisation completed!")
    print("Total valid documents:", len(catalogue))


if __name__ == "__main__":
    update_catalogue()
