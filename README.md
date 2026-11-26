# Agentic & Corrective RAG System for Course Q&A

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/mohamedelneel)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mohamedelneelkhater-oss/Agentic-RAG-System)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-F97316?style=for-the-badge&logo=gradio&logoColor=white)](https://www.gradio.app/)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-4F46E5?style=for-the-badge)](https://www.trychroma.com/)

An end-to-end **Agentic & Corrective Retrieval-Augmented Generation (RAG)** question-answering system engineered to run reliably on CPU across multi-format educational materials (`PDF`, `DOCX`, `CSV`, and `TXT`). The system pairs semantic vector retrieval in **ChromaDB** with relevance grading, query rewriting, safe abstention (`UNANSWERED`) on unanswerable queries, multi-step question decomposition, lightweight helper tools, benchmark evaluation, and an interactive **Gradio** web interface.

> **Kaggle Notebook:** Explore and reproduce the full training, evaluation, and benchmark pipeline on Kaggle: **[Agentic & Corrective RAG Notebook](https://www.kaggle.com/code/mohamedelneel)** *(update with your exact notebook slug if needed)*.

---

## Table of Contents

- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Benchmark Results](#benchmark-results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Gradio Interface Overview](#gradio-interface-overview)

---

## Key Features

- **Agentic Strategy Routing:** Plans execution dynamically before retrieval—choosing between Corrective RAG, multi-step decomposition for comparative/compound questions, summarization, dataset analytics, or safe AST-based calculation.
- **Corrective RAG Loop:** Retrieves top-`k` chunks, filters candidates by cosine similarity threshold (`0.20`) and a deterministic LLM relevance grader (`yes`/`no`, `do_sample=False`), rewrites queries when no verified chunks pass (up to `max_rewrites=2`), and safely refuses (`UNANSWERED`) when grounding is missing.
- **Multi-Format Course Ingestion:** Parses `PDF`, `DOCX`, `CSV`, and `TXT` files across `Machine Learning`, `Python Programming`, and `SQL and Databases` (6 source files, 29 chunks, max chunk size `500` chars with `1`-sentence overlap) with rich metadata (`course`, `file`,
