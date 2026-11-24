import json
from dotenv import load_dotenv
load_dotenv()

import os



documents = []

with open(os.getenv("OUTPUT_FILE"), "r", encoding="utf-8") as file:
    for line in file:
        documents.append(json.loads(line))

print("Total documents:", len(documents))

print("\nFirst document:")
print("Title:", documents[0]["title"])
print("Year:", documents[0]["year"])
print("Text length:", len(documents[0]["full_text"]))
print("Source:", documents[0]["source"])


