"""Search the local OratorIQ knowledge index."""
import json
from pathlib import Path
from joblib import load

BASE = Path(__file__).resolve().parent
CHUNKS = BASE / "storage" / "chunks.json"
INDEX = BASE / "storage" / "tfidf_index.joblib"

def retrieve(query: str, top_k: int = 4) -> list[dict]:
    query = (query or "").strip()
    if not query or not CHUNKS.exists() or not INDEX.exists():
        return []
    index = load(INDEX)
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    qvec = index["vectorizer"].transform([query])
    scores = (index["matrix"] @ qvec.T).toarray().ravel()
    ranked = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    output = []
    for i in ranked:
        if scores[i] <= 0: continue
        item = dict(chunks[i])
        item["score"] = round(float(scores[i]), 4)
        output.append(item)
        if len(output) >= max(1, int(top_k)): break
    return output

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=4)
    a = p.parse_args()
    results = retrieve(a.query, a.top_k)
    if not results:
        print("No matches. Run: python -m rag.ingest")
    for n, item in enumerate(results, 1):
        print(f"\n{n}. {item['title']} / {item['section']} (score={item['score']})")
        print(f"Source: {item['source']}\n{item['content']}")
