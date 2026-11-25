import os

import requests
from dotenv import load_dotenv

# Read the model names and settings from the .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

OLLAMA_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))
LLM_KEEP_ALIVE = os.getenv("LLM_KEEP_ALIVE", "5m")

# Qwen3 embeddings work best when the query says what it is looking for
QUERY_INSTRUCTION = "Instruct: Given a search query, retrieve passages from research papers that answer the query\nQuery: "


def embed(texts):
    response = requests.post(OLLAMA_URL + "/api/embed", json={
        "model": EMBEDDING_MODEL,
        "input": texts,
        "keep_alive": LLM_KEEP_ALIVE
    }, timeout=300)
    response.raise_for_status()
    return response.json()["embeddings"]


def embed_query(query):
    return embed([QUERY_INSTRUCTION + query])[0]


def generate(prompt):
    # qwen3 always "thinks" first. With think=True Ollama returns the thinking
    # separately, so "response" only has the final answer.
    response = requests.post(OLLAMA_URL + "/api/generate", json={
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": True,
        "keep_alive": LLM_KEEP_ALIVE,
        "options": {"temperature": LLM_TEMPERATURE, "num_ctx": LLM_NUM_CTX}
    }, timeout=300)
    response.raise_for_status()
    return response.json()["response"].strip()
