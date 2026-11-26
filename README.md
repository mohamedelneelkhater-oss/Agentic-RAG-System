Agentic & Corrective RAG System
An end-to-end Agentic and Corrective Retrieval-Augmented Generation (RAG) system for course Q&A, combining intelligent agent routing, corrective retrieval grading and query refinement, multi-format document ingestion, model benchmarking, and an interactive Gradio web interface.

Kaggle Notebook & Dataset
Kaggle Notebook: https://www.kaggle.com/code/mohamedelneel/agentic-corrective-rag-system

Kaggle Dataset (Knowledge Base): https://www.kaggle.com/datasets/mohamedelneel/course-qa-knowledge-base

Recommended Kaggle Settings: Accelerator = None (CPU) and Internet = On.

Project Structure
ingestion.py — Multi-format document loading (PDF, DOCX, CSV, TXT), text cleaning, sentence-aware chunking, and ChromaDB vector indexing (all-MiniLM-L6-v2)

agentic_rag.py — Planning & routing agent, tool suite (calculator, data analysis, summarizer, knowledge base search), corrective RAG grading/refinement loop, and safe refusal handling

gradio_app/ — Interactive Gradio web application for querying, inspecting chunks, and switching models

course_materials/ — Multi-course reference materials across multiple formats

test_set.csv & evaluation_details.csv — Evaluation questions and benchmark results

run_local.py — Local entry point that bootstraps the vector store and launches the app

requirements.txt — Python dependencies

Dockerfile — Container definition for running the application (EXPOSE 7860)

Setup & Usage
Install dependencies:pip install -r requirements.txt

Run the application locally:python run_local.py
