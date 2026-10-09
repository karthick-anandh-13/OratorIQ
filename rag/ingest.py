"""Build a local TF-IDF index. Run from project root: python -m rag.ingest"""
import json, re
from pathlib import Path
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = Path(__file__).resolve().parent
KNOWLEDGE = BASE / "knowledge"
STORAGE = BASE / "storage"
CHUNKS = STORAGE / "chunks.json"
INDEX = STORAGE / "tfidf_index.joblib"

def make_chunks(text, filename):
    title = filename.removesuffix(".md").replace("_", " ").title()
    sections, heading, lines = [], title, []
    for line in text.replace("\r\n", "\n").splitlines():
        if line.lstrip().startswith("#"):
            body = "\n".join(lines).strip()
            if body:
                sections.append((heading, body))
            heading, lines = re.sub(r"^#+\s*", "", line).strip() or title, []
        else:
            lines.append(line)
    body = "\n".join(lines).strip()
    if body:
        sections.append((heading, body))
    result = []
    for section, body in sections:
        paras = re.split(r"\n\s*\n", body)
        piece = ""
        for para in paras:
            para = para.strip()
            if not para: continue
            candidate = (piece + "\n\n" + para).strip() if piece else para
            if len(candidate) > 1000 and piece:
                result.append({"id": f"{filename}::{len(result)+1}", "source": filename,
                               "title": title, "section": section, "content": piece})
                piece = para
            else: piece = candidate
        if piece:
            result.append({"id": f"{filename}::{len(result)+1}", "source": filename,
                           "title": title, "section": section, "content": piece})
    return result

def main():
    STORAGE.mkdir(parents=True, exist_ok=True)
    docs = sorted(KNOWLEDGE.glob("*.md"))
    if not docs:
        raise SystemExit(f"No .md knowledge files found in {KNOWLEDGE}")
    chunks = []
    for path in docs:
        chunks.extend(make_chunks(path.read_text(encoding="utf-8"), path.name))
    texts = [f"{c['title']} {c['section']} {c['source']} {c['content']}" for c in chunks]
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode",
        ngram_range=(1,2), sublinear_tf=True, max_features=30000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]+\b")
    matrix = vectorizer.fit_transform(texts)
    CHUNKS.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    dump({"vectorizer": vectorizer, "matrix": matrix, "chunk_count": len(chunks)}, INDEX)
    print(f"Knowledge files: {len(docs)} | Chunks: {len(chunks)} | Vocabulary: {len(vectorizer.vocabulary_)}")
    print(f"Created: {CHUNKS}")
    print(f"Created: {INDEX}")

if __name__ == "__main__": main()
