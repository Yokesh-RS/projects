import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from engine.search import BASE_DIR, SearchEngine

PDF_FOLDER = os.path.join(BASE_DIR, "data", "pdfs")
HOME_PAGE = os.path.join(BASE_DIR, "static", "index.html")

app = FastAPI()
engine = SearchEngine()


@app.get("/")
def home():
    return FileResponse(HOME_PAGE)


@app.get("/api/search")
def search(q: str, model: str = "", offset: int = 0):
    # Without a model name the engine uses its best model
    if model == "":
        model = engine.default_model
    if model not in engine.models:
        raise HTTPException(400, "Unknown model")
    return engine.search(q, model, 10, offset)


@app.get("/api/answer")
def answer(q: str):
    # The answer comes from the local Ollama model. If Ollama is not running,
    # return an empty answer so the page can show the abstract instead.
    try:
        return engine.answer(q)
    except Exception as error:
        print("Could not generate an answer:", error)
        return {"answer": "", "sources": []}


@app.get("/pdf/{doc_id}")
def pdf(doc_id: str):
    # Only allow ids like GD12, so nobody can ask for other files on the computer
    if not doc_id.startswith("GD") or not doc_id[2:].isdigit():
        raise HTTPException(404)

    path = os.path.join(PDF_FOLDER, doc_id + ".pdf")
    if not os.path.exists(path):
        raise HTTPException(404)
    return FileResponse(path)
