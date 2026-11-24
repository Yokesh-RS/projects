
import requests
import fitz
import json
import os
import time
from dotenv import load_dotenv
load_dotenv()

from tqdm import tqdm


PDF_FOLDER = os.getenv("PDF_FOLDER", "data/pdfs")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "data/corpus.jsonl")
BASE_URL = os.getenv("OPENALEX_BASE_URL", "https://api.openalex.org/works")

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs("data", exist_ok=True)

# Research topics
search_queries = [
    "data center energy efficiency",
    "data centre cooling technologies",
    "data center water consumption",
    "data center carbon emissions",
    "AI data center energy consumption",
    "data center renewable energy"
]


collected_ids = set()

#check duplicate papers
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
        for line in file:
            document = json.loads(line)
            collected_ids.add(document["document_id"])

def download_and_extract(pdf_url, document_id):
    try:
        response = requests.get(
            pdf_url,
            timeout=30,
            headers={"User-Agent": "GreenDCResearchProject/1.0"}
        )

        response.raise_for_status()

        # Check response is PDF
        if not response.content.startswith(b"%PDF"):
            return None

        pdf_path = os.path.join(PDF_FOLDER, document_id + ".pdf")

        with open(pdf_path, "wb") as file:
            file.write(response.content)

        pdf = fitz.open(pdf_path)

        text = ""

        for page in pdf:
            text += page.get_text() + "\n"

        pdf.close()

        # Reject empty or poor documents
        if len(text.strip()) < 1000:
            os.remove(pdf_path)
            return None

        return text

    except Exception as error:
        print("Download failed:", error)
        return None


def collect_documents(target=150):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as output:
        for query in search_queries:
            if len(collected_ids) >= target:
                break
            print("\nSearching:", query)
            cursor = "*"
            while cursor and len(collected_ids) < target:
                params = {
                    "search": query,
                    "filter": "open_access.is_oa:true,has_abstract:true",
                    "per_page": 100,
                    "cursor": cursor
                }
                try:
                    response = requests.get(
                        BASE_URL,
                        params=params,
                        timeout=30
                    )
                    response.raise_for_status()
                    data = response.json()
                except Exception as error:
                    print("API request failed:", error)
                    break
                for paper in tqdm(data["results"]):
                    if len(collected_ids) >= target:
                        break
                    paper_id = paper["id"]
                    if paper_id in collected_ids:
                        continue

                    location = paper.get("best_oa_location") or {}
                    pdf_url = location.get("pdf_url")
                    if not pdf_url:
                        continue
                    document_id = "GD" + str(len(collected_ids) + 1)

                    text = download_and_extract(
                        pdf_url,
                        document_id
                    )

                    if not text:
                        continue

                    document = {
                        "document_id": document_id,
                        "title": paper.get("title"),
                        "doi": paper.get("doi"),
                        "year": paper.get("publication_year"),
                        "authors": [
                            author["author"]["display_name"]
                            for author in paper.get("authorships", [])
                        ],
                        "source": "OpenAlex",
                        "document_type": paper.get("type"),
                        "pdf_url": pdf_url,
                        "license": location.get("license"),
                        "full_text": text
                    }

                    output.write(json.dumps(document) + "\n")
                    output.flush()

                    collected_ids.add(paper_id)

                    print(
                        "Collected:",
                        len(collected_ids),
                        document["title"]
                    )

                cursor = data.get("meta", {}).get("next_cursor")

                time.sleep(0.2)

    print("\nCollection completed.")
    print("Total full-text documents:", len(collected_ids))


if __name__ == "__main__":
    collect_documents(target=150)
