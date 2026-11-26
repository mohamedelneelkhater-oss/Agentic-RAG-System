# Agentic & Corrective RAG System for Course Q&A

An end-to-end **Agentic & Corrective Retrieval-Augmented Generation (RAG)** question-answering system designed to run efficiently on CPU across multi-format course materials (`PDF`, `DOCX`, `CSV`, and `TXT`). The system combines semantic retrieval with **ChromaDB**, relevance grading, query rewriting, safe refusal on unanswerable queries, multi-step decomposition, lightweight helper tools, benchmark evaluation, and an interactive **Gradio** web application.

---

## Key Features

* **Agentic Routing & Strategy Planning:** Dynamically chooses between direct corrective RAG, multi-step decomposition for comparative/compound questions, summarization, dataset analytics, or safe calculation before querying evidence.
* **Corrective RAG Loop:** Retrieves top-`k` chunks, grades relevance against a similarity threshold (`0.20`) and an LLM grader (`yes`/`no`), rewrites queries when no verified chunks pass (up to `max_rewrites=2`), and safely refuses (`UNANSWERED`) when evidence is insufficient.
* **Multi-Format Course Ingestion:** Indexes `PDF`, `DOCX`, `CSV`, and `TXT` files into a persistent **ChromaDB** vector store (`cosine` similarity) using `sentence-transformers/all-MiniLM-L6-v2` embeddings on CPU.
* **Multi-Model CPU Benchmarking:** Compares `standard` retrieval against `agentic_corrective` across lightweight models (`flan-t5-small`, `flan-t5-base`, and `qwen2.5-0.5b-instruct`) on a 15-question evaluation suite.
* **Interactive Gradio UI & Docker Ready:** Includes a 3-tab Gradio interface for live Q&A, dataset inspection & document upload, and live test-set benchmarking, plus local runner and Dockerfile support.

---

## Architecture Overview

1. **Ingestion & Chunking:** Loads course documents across `Machine Learning`, `Python Programming`, and `SQL and Databases` (6 source files, 29 chunks, max chunk length `500` chars with `1`-sentence overlap), storing metadata (`course`, `file`, `type`, `page`, `path`).
2. **Vector Indexing:** Encodes chunks with `sentence-transformers/all-MiniLM-L6-v2` (`384` dimensions, normalized on `cpu`) and persists them in `./chroma_db`.
3. **Agent & Corrective Execution:**
   * **Tools:** Safe AST-based `calculator_tool`, catalog/dataset `data_analysis_tool`, `summarizer_tool`, and `multi_step_rag` decomposition.
   * **Grading & Rewrite:** Filters candidate chunks with a strict LLM grader (`do_sample=False`), rewrites keywords on zero verified chunks, and returns cited sources and execution traces.
4. **Evaluation & Selection:** Scores models on a 15-question test set (`test_set.csv`) using keyword recall for answerable questions, refusal accuracy for unanswerable questions, context hit rate, latency, and a weighted score (`0.80 * quality + 0.20 * speed_score`).

---

## Benchmark Results (15-Question Test Set)

### Standard vs. Agentic Corrective RAG

| Model | Mode | Quality | Context Hit | Avg Latency | Avg Rewrites | Refusal Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `flan-t5-base` | `agentic_corrective` | `0.867` | `0.846` | `3.102s` | `0.600` | `1.000` |
| `flan-t5-base` | `standard` | `0.800` | `0.923` | `1.516s` | `0.000` | `0.500` |
| `flan-t5-small` | `agentic_corrective` | `0.733` | `1.000` | `1.152s` | `0.400` | `1.000` |
| `flan-t5-small` | `standard` | `0.600` | `0.923` | `0.889s` | `0.000` | `0.500` |
| `qwen2.5-0.5b-instruct` | `agentic_corrective` | `0.867` | `1.000` | `12.694s` | `0.267` | `0.500` |
| `qwen2.5-0.5b-instruct` | `standard` | `0.733` | `0.923` | `5.089s` | `0.000` | `0.500` |

### Leaderboard (`0.80 * quality + 0.20 * speed_score`)

| Rank | Model | Quality | Refusal Rate | Avg Latency | Speed Score | Final Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | `flan-t5-small` | `0.733` | `1.000` | `1.152s` | `1.000` | **`0.787`** |
| **2** | `flan-t5-base` | `0.867` | `1.000` | `3.102s` | `0.372` | **`0.768`** |
| **3** | `qwen2.5-0.5b-instruct` | `0.867` | `0.500` | `12.694s` | `0.091` | **`0.711`** |

---

## Project Structure

```text
.
├── gradio_app/
│   ├── app.py              # 3-tab Gradio application (port 7860)
│   └── best_model.json     # Benchmark summary and selected model config
├── run_local.py            # Initializes ChromaDB if empty and starts Gradio
├── Dockerfile              # Container image based on python:3.11-slim (port 7860)
├── requirements.txt        # Python dependencies
└── .gitignore              # Ignores local DB, saved model weights, and large artifacts
