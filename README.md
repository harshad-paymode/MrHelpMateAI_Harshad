# MrHelpMate

**Why this matters**
- Helps users understand complex insurance documents with clear, grounded answers.
-Reduces misinterpretation by highlighting relevant terms and conditions instead of guessing.
- Encourages transparency by surfacing citations directly from the policy text.

**Stack / Tech**
- LangChain (RAG + LCEL pipelines)
- ChromaDB (vector store)
- BM25 + vector retrieval + reranking
- Streamlit (UI)
- LLMs (generation + embeddings)
- Python, logging, dataclasses, guardrails

**How it works (high level)**
1. Documents are chunked and indexed offline.
2. Queries retrieve relevant chunks (BM25 + vector).
3. Results are reranked and validated.
4. Guardrails (moderation + fallback + small-talk routing) decide how to respond.
5. The LLM answers ONLY from the policy — or says it’s unavailable.

**Strengths**
- Moderation for unsafe queries
- Smart fallback instead of hallucinations
- Small-talk routing
- Threshold-based relevance filtering
- Clear citations where possible

**Limitations / Future Work**
- No deep evaluation pipeline yet (RAGAS / DeepEval planned)
- Limited conversational memory
- Performance tuning and benchmarking ongoing

**Run locally**
```bash
pip install -r requirements.txt
python -m ingestion.build_index      # one-time
python -m streamlit run app.py
