Agentic & Corrective RAG System for Course Q&A

An end-to-end Agentic & Corrective Retrieval-Augmented Generation (RAG) question-answering system engineered to run reliably on CPU across multi-format course materials (PDF, DOCX, CSV, and TXT). The system combines semantic vector search in ChromaDB with relevance grading, iterative query rewriting, safe refusal (UNANSWERED) on out-of-scope queries, multi-step sub-question decomposition, lightweight helper tools, benchmark evaluation, and an interactive Gradio web interface.

Kaggle Notebook: Explore, inspect, and reproduce the full pipeline on Kaggle: Agentic & Corrective RAG Notebook on Kaggle (update the slug if your published notebook uses a specific URL path).

Table of Contents

Key Capabilities

System Architecture & Workflow

Data Ingestion, Chunking & Indexing

Agentic Routing & Corrective RAG Loop

Benchmark Evaluation & Results

Project Structure

Getting Started (Local & Docker)

Gradio Application Guide

Key Engineering Trade-Offs

Key Capabilities

Agentic Strategy Routing: Dynamically selects an execution path before retrieval—routing queries to Corrective RAG, multi-step decomposition for comparative/compound questions, content summarization, dataset/catalog analytics, or safe AST-based calculation.

Corrective RAG Loop: Retrieves top-$k$ candidate chunks, filters against a cosine similarity floor (0.20) and a deterministic LLM relevance grader (yes/no, do_sample=False), rewrites queries when no verified chunks pass (up to max_rewrites=2), and safely abstains (UNANSWERED) when grounding is missing.

Multi-Format Course Ingestion: Parses PDF, DOCX, CSV, and TXT files across Machine Learning, Python Programming, and SQL and Databases (6 source files, 29 chunks in the baseline catalog) with structured metadata (course, file, type, page, path).

CPU-First Vector Indexing: Encodes chunks using sentence-transformers/all-MiniLM-L6-v2 (384 dimensions, normalized embeddings on cpu) stored persistently in ./chroma_db with cosine similarity.

Standard vs. Agentic Benchmarking: Compares standard single-pass retrieval against agentic_corrective across lightweight models (flan-t5-small, flan-t5-base, and qwen2.5-0.5b-instruct) on a 15-question evaluation suite.

Interactive Gradio UI & Containerization: Includes a 3-tab Gradio interface on port 7860, automated local initialization (run_local.py), and a Dockerfile based on python:3.11-slim.

System Architecture & Workflow

                        +------------------------+
                        |   User Question / UI   |
                        +-----------+------------+
                                    |
                                    v
                     +------------------------------+
                     | Agentic Strategy Planner     |
                     | (Route / Decompose / Tools)  |
                     +--------------+---------------+
                                    |
          +-------------------------+-------------------------+
          |                                                   |
          v                                                   v
+--------------------+                              +--------------------+
|   Helper Tools     |                              |   Corrective RAG   |
| - calculator_tool  |                              | 1. ChromaDB Search |
| - data_analysis    |                              | 2. LLM Grader      |
| - summarizer_tool  |                              | 3. Query Rewrite   |
+---------+----------+                              | 4. Safe Refusal    |
          |                                         +---------+----------+
          |                                                   |
          +-------------------------+-------------------------+
                                    |
                                    v
                     +------------------------------+
                     | Grounded Answer + Sources    |
                     | + Execution Trace            |
                     +------------------------------+


Data Ingestion, Chunking & Indexing

Multi-Format Extraction:

PDF (pypdf): Extracts text page-by-page while retaining page metadata.

DOCX (python-docx): Splits structured content across headings/sections.

CSV (pandas): Loads tabular records such as sql_glossary.csv for lookup and dataset inspection.

TXT: Reads plain-text course syllabi and notes.

Bounded Chunking: Splits extracted text into chunks up to 500 characters with a 1-sentence overlap (overlap=1), averaging ~235 characters across the 29 baseline chunks.

Vector Persistence: Generates normalized 384-dimensional vectors with sentence-transformers/all-MiniLM-L6-v2 on cpu and stores them persistently in ./chroma_db using cosine similarity.

Agentic Routing & Corrective RAG Loop

When agentic=True, the planner inspects the user query and chooses among:

calculator_tool: Evaluates mathematical expressions safely via Python's ast module without raw eval.

data_analysis_tool: Queries catalog and dataset statistics (courses, files, formats, chunk counts, and average lengths).

summarizer_tool: Retrieves and grades relevant chunks before generating a concise 2–4 sentence summary.

multi_step_rag: Decomposes comparative or multi-part questions (such as "What is the difference between X and Y?" or compound and queries) into focused sub-questions and retrieves/grades evidence per sub-query.

corrective_rag:

Retrieves top-$k$ chunks (k=6 by default) filtered by similarity $\ge 0.20$.

Grades each candidate chunk using a strict LLM grader (yes / no, do_sample=False).

If zero verified chunks pass, rewrites the query with focused keywords up to max_rewrites=2.

If no verified evidence is found after retries, returns a safe refusal (UNANSWERED) rather than hallucinating.

When agentic=False (standard baseline), the system performs a single retrieval pass (up to 4 chunks with similarity $\ge 0.25$) and generates an answer directly without grading, rewriting, or helper tools.

Benchmark Evaluation & Results

Models were evaluated on a 15-question test set (test_set.csv) spanning direct questions across PDF, TXT, DOCX, and CSV, ambiguous queries requiring rewrites, multi-step/tool queries, and out-of-scope unanswerable questions:

Answerable questions: Scored by expected keyword recall (0.0 if falsely refused).

Unanswerable questions: Scored 1.0 for safe refusal and 0.0 if answered/hallucinated.

Composite leaderboard score:

$$\text{final\_score} = 0.80 \times \text{quality} + 0.20 \times \text{speed\_score}$$

where $\text{speed\_score} = \frac{\min(\text{avg\_latency})}{\text{avg\_latency}}$.

Standard vs. Agentic Corrective RAG

Model

Mode

Quality

Context Hit

Avg Latency

Avg Rewrites

Refusal Rate

flan-t5-base

agentic_corrective

0.867

0.846

3.102s

0.600

1.000

flan-t5-base

standard

0.800

0.923

1.516s

0.000

0.500

flan-t5-small

agentic_corrective

0.733

1.000

1.152s

0.400

1.000

flan-t5-small

standard

0.600

0.923

0.889s

0.000

0.500

qwen2.5-0.5b-instruct

agentic_corrective

0.867

1.000

12.694s

0.267

0.500

qwen2.5-0.5b-instruct

standard

0.733

0.923

5.089s

0.000

0.500

Model Leaderboard

Rank

Model

Quality

Refusal Rate

Avg Latency

Speed Score

Final Score

1

flan-t5-small

0.733

1.000

1.152s

1.000

0.787

2

flan-t5-base

0.867

1.000

3.102s

0.372

0.768

3

qwen2.5-0.5b-instruct

0.867

0.500

12.694s

0.091

0.711

Project Structure

.
├── gradio_app/
│   ├── app.py              # 3-tab Gradio web interface (port 7860)
│   └── best_model.json     # Benchmark summary & selected model metadata
├── run_local.py            # Initializes ChromaDB if empty and starts Gradio
├── Dockerfile              # Container image based on python:3.11-slim (port 7860)
├── requirements.txt        # Python package dependencies
└── .gitignore              # Excludes local DB, saved model weights, and build artifacts


Getting Started (Local & Docker)

1. Clone the Repository

git clone https://github.com/mohamedelneelkhater-oss/Agentic-RAG-System.git
cd Agentic-RAG-System


2. Install Dependencies

pip install -r requirements.txt


3. Run Locally

python run_local.py


run_local.py automatically initializes ./chroma_db if no index exists yet and launches the Gradio server on http://0.0.0.0:7860.

4. Run with Docker

docker build -t agentic-rag-course-qa .
docker run -p 7860:7860 agentic-rag-course-qa


Gradio Application Guide

Agentic RAG Q&A Assistant: Select a model, toggle Agentic Corrective mode, filter by course, tune top_k and max_rewrites, and inspect generated answers, citations, execution traces, and verified chunks.

Original Dataset & Document Upload: Browse the file catalog, sql_glossary.csv, raw syllabus texts, and indexed chunks, or upload new PDF, DOCX, CSV, and TXT files directly into ChromaDB.

15-Question Test Set & Model Benchmark: Review model rankings and the 15-question test set, and click Run 15-Question Test Set Live to run evaluations interactively.

Key Engineering Trade-Offs

Latency vs. Answer Quality on CPU: flan-t5-small ranks first (0.787) due to fast CPU execution (1.152s average latency) and 1.000 refusal accuracy, while flan-t5-base achieves higher answer quality (0.867) with 1.000 refusal accuracy at 3.102s average latency.

Strict Grading vs. Recall on Multi-Part Queries: A strict yes/no grader paired with query rewriting prevents hallucinations on out-of-scope questions, though it can occasionally reject partially relevant chunks on comparative sub-queries.
