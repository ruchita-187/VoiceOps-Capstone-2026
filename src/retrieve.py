import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

# Load the saved knowledge-base index and section data
index = faiss.read_index(str(OUTPUT_DIR / "kb.index"))

with open(OUTPUT_DIR / "kb_chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

# Use the same embedding model used to build the index
model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query, top_k=5):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk = chunks[idx]

        results.append({
            "score": float(score),
            "source_file": chunk["source_file"],
            "section_id": chunk["section_id"],
            "title": chunk["title"],
            "text": chunk["text"]
        })

    return results


if __name__ == "__main__":
    question = input("Enter a customer question: ")

    results = retrieve(question)

    print("\nRetrieved knowledge-base sections:\n")

    for result in results:
        print("-" * 60)
        print(f"Source: {result['source_file']}")
        print(f"Section: {result['section_id']}")
        print(f"Title: {result['title']}")
        print(f"Score: {result['score']:.4f}")
        print(result["text"])