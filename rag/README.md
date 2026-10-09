# OratorIQ RAG module

1. Copy the `rag` folder into `D:\OratorIQ\rag`.
2. Activate the project's `.venv311`.
3. Install dependencies if needed: `python -m pip install scikit-learn joblib`
4. From `D:\OratorIQ`, run: `python -m rag.ingest`
5. Test retrieval: `python -m rag.retrieve "fast pacing speaking rate natural pauses"`
6. Test coaching: `python -m rag.coach artifacts\results\inference_hybrid_result.json`

Re-run ingestion whenever Markdown knowledge files change. This initial version uses local TF-IDF retrieval and template-based coaching; no external LLM/API is required. The generated index files are not included because ingestion creates them.
