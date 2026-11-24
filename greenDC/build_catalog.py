
import json
import os

INPUT_FILE = "data/corpus.jsonl"
OUTPUT_FILE = "data/documents_catalogue.json"
PDF_FOLDER = "data/pdfs"


def identify_topics(title, text):

    content = (title + " " + text[:5000]).lower()

    topic_keywords = {
        "Energy Efficiency": [
            "energy efficiency", "power usage",
            "energy consumption", "server efficiency"
        ],
        "Cooling Technologies": [
            "cooling", "liquid cooling", "air cooling",
            "thermal management"
        ],
        "Water Consumption": [
            "water consumption", "water usage",
            "water efficiency", "water footprint"
        ],
        "Carbon Emissions": [
            "carbon emissions", "carbon footprint",
            "greenhouse gas", "carbon dioxide"
        ],
        "Sustainable AI Infrastructure": [
            "artificial intelligence", "machine learning",
            "gpu", "large language model", "ai infrastructure"
        ],
        "Renewable Energy": [
            "renewable energy", "solar energy",
            "wind energy", "clean energy"
        ]
    }

    matched_topics = []

    for topic, keywords in topic_keywords.items():
        for keyword in keywords:
            if keyword in content:
                matched_topics.append(topic)
                break

    if not matched_topics:
        matched_topics.append("General Data Centre Sustainability")

    return matched_topics


def build_catalogue():

    catalogue = {}

    with open(INPUT_FILE, "r", encoding="utf-8") as file:

        for line in file:

            document = json.loads(line)

            doc_id = document.get("document_id")
            title = document.get("title") or "Unknown title"
            full_text = document.get("full_text") or ""

            pdf_path = os.path.join(PDF_FOLDER, doc_id + ".pdf")

            # Store the complete document as JSON
            catalogue[doc_id] = {
                "metadata": {
                    "name": title,
                    "authors": document.get("authors", []),
                    "year": document.get("year"),
                    "license": document.get("license"),
                    "url": document.get("pdf_url"),
                    "doi": document.get("doi"),
                    "source": document.get("source"),
                    "document_type": document.get("document_type"),
                    "topics": identify_topics(title, full_text),
                    "document_path": (
                        pdf_path if os.path.exists(pdf_path) else None
                    )
                },
                "full_text": full_text
            }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            catalogue,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("JSON catalogue created successfully!")
    print("Total documents:", len(catalogue))
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    build_catalogue()
