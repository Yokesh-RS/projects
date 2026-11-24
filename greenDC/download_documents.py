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
API_KEY = os.getenv("OPENALEX_API_KEY")

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs("data", exist_ok=True)

# Research topics

search_queries = [

    "green data centres",
    "sustainable data centres",
    "environmentally sustainable data centres",
    "data centre sustainability challenges",
    "sustainable data centre design and architecture",
    "green computing and sustainable IT infrastructure",
    "environmental impact of hyperscale data centres",
    "life cycle assessment of data centres",


    "data centre energy consumption trends",
    "data centre energy efficiency technologies",
    "data centre power usage effectiveness PUE",
    "data centre energy optimisation strategies",
    "energy efficient server infrastructure",
    "data centre workload scheduling for energy efficiency",
    "AI workload energy optimisation",
    "GPU energy consumption in AI data centres",
    "energy demand of large language model training",
    "energy consumption of AI model inference",
    "data centre electricity demand and grid capacity",
    "data centre peak electricity demand",
    "energy efficient computing architectures",


    "data centre water consumption",
    "data centre water footprint",
    "water usage effectiveness WUE in data centres",
    "data centre cooling water consumption",
    "environmental impact of data centre water usage",
    "data centres and regional water scarcity",
    "groundwater depletion caused by data centre operations",
    "water efficient data centre cooling technologies",
    "alternatives to water intensive data centre cooling",
    "AI data centre water footprint",


    "data centre cooling technologies",
    "sustainable data centre cooling systems",
    "liquid cooling in data centres",
    "direct to chip liquid cooling",
    "immersion cooling for high performance computing",
    "free air cooling in data centres",
    "evaporative cooling environmental impact",
    "data centre thermal management",
    "waste heat recovery from data centres",
    "data centre heat reuse for district heating",
    "low energy cooling systems for hyperscale data centres",


    "data centre carbon emissions",
    "carbon footprint of data centres",
    "scope 1 scope 2 and scope 3 emissions data centres",
    "embodied carbon of data centre infrastructure",
    "lifecycle carbon emissions of servers",
    "carbon emissions from AI computing",
    "carbon aware data centre workload scheduling",
    "carbon intensity aware computing",
    "data centre decarbonisation strategies",
    "environmental impact of cloud computing",
    "carbon footprint of large language models",


    "renewable energy for data centres",
    "data centres powered by solar energy",
    "data centres powered by wind energy",
    "data centre renewable energy integration",
    "24/7 carbon free energy data centres",
    "power purchase agreements data centres",
    "data centre battery energy storage",
    "microgrids for sustainable data centres",
    "data centres and electricity grid sustainability",
    "nuclear energy for data centres",
    "small modular reactors data centre energy",
    "data centre demand response and grid flexibility",


    "environmental justice impacts of data centres",
    "data centre development and local communities",
    "data centres and competition for water resources",
    "data centre electricity demand and local communities",
    "environmental impacts of hyperscale data centre expansion",
    "resource consumption of cloud computing companies",
    "environmental costs of artificial intelligence infrastructure",
    "data centre construction and land use impacts",
    "data centre development in water stressed regions",
    "social and environmental impacts of AI infrastructure",
    "data centre resource use and environmental inequality",
    "corporate sustainability claims data centres",
    "data centre greenwashing and environmental reporting",
    "hyperscaler environmental sustainability commitments",


    "data centre electronic waste management",
    "server hardware lifecycle sustainability",
    "recycling and reuse of data centre equipment",
    "environmental impact of semiconductor manufacturing",
    "embodied energy of data centre hardware",
    "rare earth minerals used in computing infrastructure",
    "sustainable server manufacturing",
    "circular economy for data centre infrastructure",
    "GPU lifecycle environmental impact",
    "responsible disposal of data centre hardware",


    "AI powered data centre energy optimisation",
    "machine learning for data centre cooling optimisation",
    "digital twins for sustainable data centres",
    "reinforcement learning for data centre energy management",
    "advanced liquid cooling for AI data centres",
    "high temperature liquid cooling technologies",
    "immersion cooling for AI GPU clusters",
    "photonic computing energy efficiency",
    "neuromorphic computing energy efficiency",
    "low power AI accelerator architectures",
    "carbon aware cloud computing",
    "green AI and sustainable machine learning",
    "energy efficient AI inference infrastructure",
    "next generation sustainable data centre technologies",


    "Google data centre sustainability energy water",
    "Microsoft data centre environmental impact water",
    "Amazon AWS data centre sustainability",
    "Meta data centre energy consumption",
    "hyperscale cloud provider water consumption",
    "cloud computing companies carbon emissions",
    "data centre sustainability reporting comparison",
    "corporate data centre renewable energy commitments",
    "environmental impact of hyperscale cloud infrastructure",


    "data centre energy efficiency regulations",
    "European Union data centre energy reporting requirements",
    "data centre sustainability standards",
    "data centre energy efficiency code of conduct",
    "data centre environmental performance indicators",
    "data centre sustainability disclosure requirements",
    "data centre water reporting regulations",
    "sustainable digital infrastructure policy",
    "data centre environmental impact assessment",

    "how to make data centres more sustainable",
    "best practices for sustainable data centre operations",
    "strategies to reduce data centre energy consumption",
    "strategies to reduce data centre water consumption",
    "reducing carbon emissions in data centre operations",
    "sustainable data centre site selection",
    "data centre infrastructure optimisation",
    "server virtualisation and energy savings",
    "data centre capacity planning for sustainability",
    "multi objective optimisation of data centre sustainability",
    "trade offs between energy water and carbon in data centres"
]



collected_paper_ids = set()
collected_document_ids = set()
existing_count = 0

# Track duplicate papers by OpenAlex ID or DOI.
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError:
                continue

            existing_count += 1
            collected_document_ids.add(document.get("document_id"))

            paper_id = document.get("openalex_id") or document.get("doi")
            if paper_id:
                collected_paper_ids.add(paper_id)


def next_document_id():
    index = existing_count + 1
    while f"GD{index}" in collected_document_ids:
        index += 1
    return f"GD{index}", index


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
    global existing_count

    if existing_count >= target:
        print(f"Corpus already contains {existing_count} documents. Nothing to download.")
        return

    with open(OUTPUT_FILE, "a", encoding="utf-8") as output:
        for query in search_queries:
            if existing_count >= target:
                break

            print("\nSearching:", query)
            cursor = "*"

            while cursor and existing_count < target:
                params = {
                    "search": query,
                    "filter": "open_access.is_oa:true,has_abstract:true",
                    "per_page": 100,
                    "cursor": cursor
                }

                # OpenAlex accepts the API key as an api_key query parameter.
                if API_KEY:
                    params["api_key"] = API_KEY

                try:
                    response = requests.get(
                        BASE_URL,
                        params=params,
                        timeout=30,
                        headers={"User-Agent": "GreenDCResearchProject/1.0"}
                    )
                    response.raise_for_status()
                    data = response.json()
                except Exception as error:
                    print("API request failed:", error)
                    break

                for paper in tqdm(data["results"]):
                    if existing_count >= target:
                        break

                    paper_id = paper["id"]
                    paper_doi = paper.get("doi")

                    if paper_id in collected_paper_ids:
                        continue
                    if paper_doi and paper_doi in collected_paper_ids:
                        continue

                    location = paper.get("best_oa_location") or {}
                    pdf_url = location.get("pdf_url")
                    if not pdf_url:
                        continue

                    document_id, next_index = next_document_id()

                    text = download_and_extract(pdf_url, document_id)
                    if not text:
                        continue

                    document = {
                        "document_id": document_id,
                        "openalex_id": paper_id,
                        "title": paper.get("title"),
                        "doi": paper_doi,
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

                    output.write(json.dumps(document, ensure_ascii=False) + "\n")
                    output.flush()

                    collected_paper_ids.add(paper_id)
                    if paper_doi:
                        collected_paper_ids.add(paper_doi)
                    collected_document_ids.add(document_id)
                    existing_count += 1

                    print(
                        "Collected:",
                        existing_count,
                        document["title"]
                    )

                cursor = data.get("meta", {}).get("next_cursor")
                time.sleep(0.2)

    print("\nCollection completed.")
    print("Total full-text documents:", existing_count)


if __name__ == "__main__":
    collect_documents(target=150)
