
import requests
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

download_folder = Path("data/new_pdfs")
download_folder.mkdir(parents=True, exist_ok=True)

metadata_file = Path("data/new_documents.json")

search_queries = [
    "data centre energy efficiency and power consumption",
    "data center liquid cooling and thermal management",
    "data centre water consumption and water efficiency",
    "carbon aware computing and workload scheduling",
    "GPU cluster energy consumption and sustainable AI",
    "renewable energy integration in data centres",
    "embodied carbon and data centre hardware lifecycle",
    "energy aware scheduling of deep learning workloads"
]

headers = {
    "User-Agent": "GreenDC-Evidence-Navigator/1.0"
}

new_documents = []
downloaded_ids = set()

for query in search_queries:

    print("\nSearching:", query)

    api_url = "https://api.openalex.org/works"

    params = {
        "search": query,
        "filter": "is_oa:true",
        "per_page": 25,
        "sort": "relevance_score:desc"
    }

    response = requests.get(
        api_url,
        params=params,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()
    results = response.json().get("results", [])

    for paper in results:

        paper_id = paper.get("id")

        if not paper_id or paper_id in downloaded_ids:
            continue

        downloaded_ids.add(paper_id)

        title = paper.get("title") or "Untitled"
        doi = paper.get("doi")

        # Find a direct PDF URL from an open-access location.
        pdf_url = None

        locations = paper.get("locations") or []

        for location in locations:

            if location and location.get("pdf_url"):
                pdf_url = location["pdf_url"]
                break

        if not pdf_url:
            print("No direct PDF:", title)
            continue

        safe_title = re.sub(
            r"[^a-zA-Z0-9]+",
            "_",
            title
        )[:100].strip("_")

        pdf_path = download_folder / f"{safe_title}.pdf"

        try:

            pdf_response = requests.get(
                pdf_url,
                headers=headers,
                timeout=60
            )

            pdf_response.raise_for_status()

            # Check that the response is actually a PDF.
            if not pdf_response.content.startswith(b"%PDF"):
                print("Not a valid PDF:", title)
                continue

            with open(pdf_path, "wb") as file:
                file.write(pdf_response.content)

            authors = []

            for authorship in paper.get("authorships", []):
                author = authorship.get("author", {})
                authors.append(author.get("display_name", ""))

            document = {
                "title": title,
                "authors": authors,
                "year": paper.get("publication_year"),
                "doi": doi,
                "openalex_id": paper_id,
                "source_url": paper.get("id"),
                "pdf_url": pdf_url,
                "document_path": str(pdf_path),
                "download_status": "downloaded"
            }

            new_documents.append(document)

            print("Downloaded:", title)

        except requests.RequestException as error:
            print("Download failed:", title, error)

    time.sleep(1)

with open(metadata_file, "w", encoding="utf-8") as file:
    json.dump(
        new_documents,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\nSearch completed.")
print("PDFs downloaded:", len(new_documents))
print("Metadata saved to:", metadata_file)
