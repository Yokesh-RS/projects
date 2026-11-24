import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from engine.models import MODELS
from engine.search import BASE_DIR, SearchEngine, highlight

app = FastAPI(title="GreenDC Search")
engine = SearchEngine()

PDF_FOLDER = os.path.join(BASE_DIR, "data", "pdfs")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")


@app.get("/")
def home():
    return FileResponse(os.path.join(STATIC_FOLDER, "index.html"))


@app.get("/api/stats")
def stats():
    years = [doc["year"] for doc in engine.index.documents if doc["year"]]
    return {
        "documents": engine.index.N,
        "terms": len(engine.index.vocabulary),
        "year_min": min(years),
        "year_max": max(years),
        "models": list(MODELS)
    }


@app.get("/api/search")
def search(
    q: str,
    model: str = "lm",
    k: int = Query(10, ge=1, le=50),
    offset: int = Query(0, ge=0),
    topic: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None
):
    if model not in MODELS:
        raise HTTPException(400, f"Unknown model '{model}'")

    response = engine.search(q, model, k, offset, topic, year_from, year_to)

    query_terms = set(response["query_terms"])
    for result in response["results"]:
        result["title_html"] = highlight(result["title"], query_terms)
        result["snippet_html"] = highlight(result["snippet"], query_terms)
        result["has_pdf"] = os.path.exists(os.path.join(PDF_FOLDER, result["doc_id"] + ".pdf"))
        del result["snippet"]

    return response


@app.get("/pdf/{doc_id}")
def pdf(doc_id: str):
    # doc_id goes into a file path, so only allow the GD<number> form.
    if not doc_id.startswith("GD") or not doc_id[2:].isdigit():
        raise HTTPException(404)

    path = os.path.join(PDF_FOLDER, doc_id + ".pdf")
    if not os.path.exists(path):
        raise HTTPException(404)
    return FileResponse(path, media_type="application/pdf")
