import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/kaggle/working")

import gradio as gr
import pandas as pd
import ingestion as ing
import agentic_rag as arag

CFG = json.load(open(ing.CONFIG_PATH, encoding="utf-8"))
BEST_MODEL = CFG["best_model"]
MODELS = list(arag.LLM_REGISTRY.keys())
UPLOAD_DIR = Path(ing.DATA_DIR) / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TEST_SET_PATH = Path(ing.BASE_DIR) / "test_set.csv"


def get_course_choices():
    cat = ing.catalog()
    courses = sorted(cat["course"].unique().tolist()) if not cat.empty else []
    return ["All"] + courses


def get_original_data_views():
    cat_df = ing.catalog().drop(columns=["path"], errors="ignore")
    csv_path = Path(ing.DATA_DIR) / "sql_databases" / "sql_glossary.csv"
    sql_df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

    txt_blocks = []
    for p in sorted(Path(ing.DATA_DIR).rglob("*.txt")):
        txt_blocks.append(f"### {p.name}\n```text\n{p.read_text(encoding='utf-8')}\n```")
    syllabi_md = "\n\n".join(txt_blocks) if txt_blocks else "No TXT syllabi found."

    chunks_df = ing.get_all_chunks_df()
    return cat_df, sql_df, syllabi_md, chunks_df


def answer_question(question, model_name, agentic_mode, course_filter, top_k, max_rewrites):
    if not question or not question.strip():
        return "Please enter a question.", "", ""

    out = arag.run_agentic_rag(
        question=question.strip(),
        model_name=model_name,
        k=int(top_k),
        course=course_filter,
        agentic=bool(agentic_mode),
        max_rewrites=int(max_rewrites),
    )

    sources_md = (
        "\n".join(f"- `{s}`" for s in out["sources"])
        if out["sources"]
        else "- *No verified sources (safe refusal)*"
    )
    meta_banner = (
        f"**Model:** `{model_name}` | **Strategy/Tool:** `{out['tool']}` | "
        f"**Retrieved:** {out['retrieved']} | **Verified:** {out['accepted']} | "
        f"**Rewrites:** {out['rewrites']} | **Latency:** {out['latency']:.2f}s"
    )
    answer_md = (
        f"### Answer\n{out['answer']}\n\n"
        f"### Verified Sources\n{sources_md}\n\n---\n{meta_banner}"
    )
    trace_md = f"### Agentic & Corrective Execution Trace\n{arag.format_trace(out)}"

    if out["hits"]:
        blocks = []
        for idx, h in enumerate(out["hits"], 1):
            m = h["meta"]
            blocks.append(
                f"**Chunk {idx}** — `{m['file']}` ({m['course']}, {m['type']}, part {m['page']}, sim={h['score']:.2f})\n"
                f"> {h['text']}"
            )
        chunks_preview = "\n\n".join(blocks)
    else:
        chunks_preview = "*No vector chunks used for this response.*"

    return answer_md, trace_md, chunks_preview


def upload_and_ingest(files, course_name):
    if not files:
        cat_df, _, _, chunks_df = get_original_data_views()
        return "No files selected.", cat_df, chunks_df, gr.update(choices=get_course_choices())

    target_course = course_name.strip() or "Uploaded Documents"
    logs = []
    for f in files:
        src = Path(f.name if hasattr(f, "name") else str(f))
        if src.suffix.lower() not in ing.SUPPORTED:
            logs.append(f"- **{src.name}**: Unsupported format (allowed: PDF, DOCX, CSV, TXT)")
            continue
        dest = UPLOAD_DIR / src.name
        shutil.copy(src, dest)
        n_chunks = ing.ingest_file(str(dest), course=target_course)
        logs.append(
            f"- **{src.name}**: Extracted, cleaned, chunked & indexed (**{n_chunks} chunks**) into `{target_course}`."
        )

    cat_df, _, _, chunks_df = get_original_data_views()
    return "\n".join(logs), cat_df, chunks_df, gr.update(choices=get_course_choices())


def run_live_test_suite(model_name, agentic_mode):
    if not TEST_SET_PATH.exists():
        return "test_set.csv not found.", pd.DataFrame()

    tdf = pd.read_csv(TEST_SET_PATH)
    rows = []
    for _, row in tdf.iterrows():
        q = row["question"]
        answerable = bool(row["answerable"])
        kws = eval(row["keywords"]) if isinstance(row["keywords"], str) else []
        out = arag.run_agentic_rag(q, model_name, agentic=bool(agentic_mode))
        ans = out["answer"]
        refused = arag.is_refusal(ans)
        if answerable:
            passed = (not refused) and (sum(k.lower() in ans.lower() for k in kws) / max(len(kws), 1) >= 0.5)
        else:
            passed = refused
        rows.append({
            "ID": row["id"],
            "Category": row["category"],
            "Question": q,
            "Tool Used": out["tool"],
            "Answer": ans,
            "Passed": "PASS" if passed else "FAIL",
            "Rewrites": out["rewrites"],
            "Latency (s)": round(out["latency"], 2),
        })

    res_df = pd.DataFrame(rows)
    acc = (res_df["Passed"] == "PASS").mean() * 100
    avg_lat = res_df["Latency (s)"].mean()
    summary = (
        f"### Live Test Set Results (`{model_name}` | `{'Agentic Corrective' if agentic_mode else 'Standard'}`)\n"
        f"- **Pass Rate:** {(res_df['Passed'] == 'PASS').sum()}/{len(res_df)} (**{acc:.1f}%**)\n"
        f"- **Average Latency:** **{avg_lat:.2f}s** per question"
    )
    return summary, res_df


def build_ui():
    cat_df, sql_df, syllabi_md, chunks_df = get_original_data_views()
    ranking_df = pd.DataFrame(CFG["ranking"])
    test_df = pd.read_csv(TEST_SET_PATH) if TEST_SET_PATH.exists() else pd.DataFrame()

    with gr.Blocks(title="Agentic & Corrective RAG System") as demo:
        gr.Markdown(
            f"# Agentic & Corrective RAG Course Assistant\n"
            f"Unified **Intelligent Agent + Corrective RAG** system running on lightweight models. "
            f"**Saved Best Model:** `{BEST_MODEL}` (selectable below)."
        )

        with gr.Tabs():
            with gr.Tab("1. Agentic RAG Q&A Assistant"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_dd = gr.Dropdown(
                            choices=MODELS,
                            value=BEST_MODEL,
                            label="Select LLM Model (Best Model Pre-selected)",
                        )
                        agentic_chk = gr.Checkbox(
                            value=True,
                            label="Enable Agentic + Corrective Mode (Planner + Tools + Grader + Rewrite)",
                        )
                        course_dd = gr.Dropdown(
                            choices=get_course_choices(),
                            value="All",
                            label="Course Filter",
                        )
                        top_k_sl = gr.Slider(3, 8, value=6, step=1, label="Top-K Retrieved Chunks")
                        rewrites_sl = gr.Slider(0, 2, value=2, step=1, label="Max Query Rewrites")

                    with gr.Column(scale=2):
                        question_tb = gr.Textbox(
                            lines=2,
                            placeholder="Ask a course question, a comparative multi-step question, dataset stats, or math...",
                            label="Student Question",
                        )
                        ask_btn = gr.Button("Ask Agentic RAG", variant="primary")

                        gr.Examples(
                            examples=[
                                ["What does the learning rate control in gradient descent?"],
                                ["how can i stop my model from memorizing the training data?"],
                                ["What is the difference between a list and a tuple in Python?"],
                                ["How many files and total chunks are in the dataset?"],
                                ["Calculate 40 / (40 + 10)"],
                                ["What is the final exam date of the SQL course?"],
                            ],
                            inputs=question_tb,
                            label="Sample Questions (Direct, Vague, Multi-Step, Tools, Unanswerable)",
                        )

                        answer_out = gr.Markdown(label="Answer & Sources")
                        with gr.Accordion("Agent Reasoning Trace & Verified Chunks", open=True):
                            trace_out = gr.Markdown()
                            chunks_out = gr.Markdown()

                ask_btn.click(
                    fn=answer_question,
                    inputs=[question_tb, model_dd, agentic_chk, course_dd, top_k_sl, rewrites_sl],
                    outputs=[answer_out, trace_out, chunks_out],
                )
                question_tb.submit(
                    fn=answer_question,
                    inputs=[question_tb, model_dd, agentic_chk, course_dd, top_k_sl, rewrites_sl],
                    outputs=[answer_out, trace_out, chunks_out],
                )

            with gr.Tab("2. Original Dataset & Document Upload"):
                gr.Markdown(
                    "### Dataset Source & Kaggle Link Info\n"
                    "- **Working Directory:** `/kaggle/working/course_materials/` (3 courses, 4 formats: PDF, DOCX, CSV, TXT)\n"
                    "- **Kaggle Dataset Export:** Save this notebook version and click **Output -> New Dataset** on `course_materials/` to publish your shareable Kaggle Dataset URL."
                )
                with gr.Row():
                    upload_files = gr.File(
                        file_count="multiple",
                        file_types=[".pdf", ".docx", ".csv", ".txt"],
                        label="Upload Additional Course Files (PDF, DOCX, CSV, TXT)",
                    )
                    with gr.Column():
                        upload_course_tb = gr.Textbox(value="Uploaded Documents", label="Target Course Name")
                        upload_btn = gr.Button("Ingest Uploaded Files into ChromaDB", variant="secondary")
                        upload_status = gr.Markdown()

                gr.Markdown("### 1. Knowledge Base File Catalog")
                catalog_df_view = gr.Dataframe(value=cat_df, interactive=False)

                gr.Markdown("### 2. Original CSV Data (`sql_glossary.csv`)")
                gr.Dataframe(value=sql_df, interactive=False)

                with gr.Accordion("3. Original TXT Syllabi (Raw Content)", open=False):
                    gr.Markdown(syllabi_md)

                with gr.Accordion("4. Processed & Indexed Chunks in ChromaDB", open=False):
                    chunks_df_view = gr.Dataframe(value=chunks_df, interactive=False)

                upload_btn.click(
                    fn=upload_and_ingest,
                    inputs=[upload_files, upload_course_tb],
                    outputs=[upload_status, catalog_df_view, chunks_df_view, course_dd],
                )

            with gr.Tab("3. 15-Question Test Set & Model Benchmark"):
                gr.Markdown(f"### Saved Model Leaderboard (Winner: `{BEST_MODEL}`)")
                gr.Dataframe(value=ranking_df, interactive=False)

                gr.Markdown("### 15-Question Evaluation Test Set (`test_set.csv`)")
                gr.Dataframe(value=test_df, interactive=False)

                with gr.Row():
                    eval_model_dd = gr.Dropdown(choices=MODELS, value=BEST_MODEL, label="Model to Evaluate")
                    eval_mode_chk = gr.Checkbox(value=True, label="Agentic Corrective Mode")
                    run_eval_btn = gr.Button("Run 15-Question Test Set Live", variant="primary")

                eval_summary_md = gr.Markdown()
                eval_results_df = gr.Dataframe(interactive=False)

                run_eval_btn.click(
                    fn=run_live_test_suite,
                    inputs=[eval_model_dd, eval_mode_chk],
                    outputs=[eval_summary_md, eval_results_df],
                )

    return demo


if __name__ == "__main__":
    ing.get_embedder()
    arag.load_llm(BEST_MODEL)
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        theme=gr.themes.Soft(),
    )
