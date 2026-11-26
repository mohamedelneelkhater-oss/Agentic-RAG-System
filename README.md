import shutil
import subprocess
from pathlib import Path
from kaggle_secrets import UserSecretsClient

GITHUB_USERNAME = "mohamedelneelkhater-oss"
REPO_NAME = "Agentic-RAG-System"
BRANCH = "main"

KAGGLE_NOTEBOOK_URL = "https://www.kaggle.com/code/<your-kaggle-username>/<your-notebook-slug>"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/<your-kaggle-username>/<your-dataset-slug>"

readme_lines = [
    "# Agentic & Corrective RAG System",
    "",
    "An end-to-end **Agentic and Corrective Retrieval-Augmented Generation (RAG)** system for course Q&A, combining intelligent agent routing, corrective retrieval grading and query refinement, multi-format document ingestion, model benchmarking, and an interactive Gradio web interface.",
    "",
    "## Kaggle Notebook & Dataset",
    f"- **Kaggle Notebook:** {KAGGLE_NOTEBOOK_URL}",
    f"- **Kaggle Dataset (Knowledge Base):** {KAGGLE_DATASET_URL}",
    "- **Recommended Kaggle Settings:** `Accelerator = None (CPU)` and `Internet = On`.",
    "",
    "## Project Structure",
    "- `ingestion.py` — Multi-format document loading (PDF, DOCX, CSV, TXT), text cleaning, sentence-aware chunking, and ChromaDB vector indexing (`all-MiniLM-L6-v2`)",
    "- `agentic_rag.py` — Planning & routing agent, tool suite (calculator, data analysis, summarizer, knowledge base search), corrective RAG grading/refinement loop, and safe refusal handling",
    "- `gradio_app/` — Interactive Gradio web application for querying, inspecting chunks, and switching models",
    "- `course_materials/` — Multi-course reference materials across multiple formats",
    "- `test_set.csv` & `evaluation_details.csv` — Evaluation questions and benchmark results",
    "- `run_local.py` — Local entry point that bootstraps the vector store and launches the app",
    "- `requirements.txt` — Python dependencies",
    "- `Dockerfile` — Container definition for running the application (`EXPOSE 7860`)",
    "",
    "## Setup & Usage",
    "1. Install dependencies:",
    "   `pip install -r requirements.txt`",
    "2. Run the application locally:",
    "   `python run_local.py`",
]

working_dir = Path("/kaggle/working")
readme_path = working_dir / "README.md"
readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

sync_dir = working_dir / "github_sync"
sync_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(readme_path, sync_dir / "README.md")

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
repo_url = f"https://{GITHUB_USERNAME}:{token}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"

subprocess.run(["git", "remote", "set-url", "origin", repo_url], cwd=sync_dir, check=True)
subprocess.run(["git", "add", "README.md"], cwd=sync_dir, check=True)
subprocess.run(["git", "commit", "-m", "Add README.md to repository"], cwd=sync_dir, check=True)
subprocess.run(["git", "push", "origin", BRANCH], cwd=sync_dir, check=True)

print("README.md pushed to GitHub successfully!")
print("\n" + readme_path.read_text(encoding="utf-8"))
