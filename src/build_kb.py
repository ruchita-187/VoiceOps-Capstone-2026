
import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "kb"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

def read_frontmatter(content):
    metadata = {}

    if not content.startswith("---"):
        return metadata

    parts = content.split("---", 2)

    if len(parts) < 3:
        return metadata

    for line in parts[1].strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip('"').strip("'")

    return metadata


def split_into_sections(content):
    # Split at headings containing section IDs, e.g. [KB-00.1]
    pattern = r"(?=^## .*?\[[^\]]+\])"
    sections = re.split(pattern, content, flags=re.MULTILINE)

    chunks = []

    for section in sections:
        section = section.strip()

        if not section or section.startswith("---"):
            continue

        match = re.search(r"\[([^\]]+)\]", section)

        if match:
            section_id = match.group(1)
        else:
            section_id = "INTRO"

        chunks.append({
            "section_id": section_id,
            "text": section
        })

    return chunks


all_chunks = []

for file_path in sorted(KB_DIR.glob("*.md")):
    content = file_path.read_text(encoding="utf-8")
    metadata = read_frontmatter(content)

    for section in split_into_sections(content):
        all_chunks.append({
            "source_file": file_path.name,
            "section_id": section["section_id"],
            "doc_id": metadata.get("doc_id", ""),
            "title": metadata.get("title", ""),
            "doc_type": metadata.get("doc_type", ""),
            "status": metadata.get("status", ""),
            "effective_from": metadata.get("effective_from", ""),
            "text": section["text"]
        })

if not all_chunks:
    raise ValueError("No knowledge-base sections were found. Check the kb folder.")

print(f"Found {len(all_chunks)} document sections.")

# Create embeddings
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk["text"] for chunk in all_chunks]

print("Creating embeddings...")
embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

embeddings = np.asarray(embeddings, dtype="float32")

# Create FAISS similarity index
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Save index and its matching section information
faiss.write_index(index, str(OUTPUT_DIR / "kb.index"))

with open(OUTPUT_DIR / "kb_chunks.json", "w", encoding="utf-8") as file:
    json.dump(all_chunks, file, indent=2, ensure_ascii=False)

print("\nKnowledge-base index created successfully.")
print(f"Sections indexed: {len(all_chunks)}")
print(f"Index saved to: {OUTPUT_DIR / 'kb.index'}")
print(f"Section data saved to: {OUTPUT_DIR / 'kb_chunks.json'}")
