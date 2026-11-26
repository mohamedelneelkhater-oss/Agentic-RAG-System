# Agentic & Corrective RAG System

An end-to-end **Agentic and Corrective Retrieval-Augmented Generation (RAG)** system with document ingestion, evaluation datasets, and an interactive Gradio web interface.

## Project Structure
- `ingestion.py` — Document loading, chunking, and ingestion pipeline
- `agentic_rag.py` — Agentic routing and corrective RAG workflow
- `gradio_app/` — Interactive web interface for querying the system
- `course_materials/` — Source reference materials
- `test_set.csv` & `evaluation_details.csv` — Evaluation dataset and detailed results

## Setup & Usage
1. Install dependencies:
   `pip install -r requirements.txt`
2. Run the application locally:
   `python run_local.py`
