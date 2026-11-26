import ast
import gc
import operator as op
import os
import re
import time
import pandas as pd

import ingestion as ing
from ingestion import CONFIG_PATH, SAVED_LLM_DIR

UNANSWERED = "Sorry, the course materials do not contain enough information to answer this question."
UNANSWERED_MARKERS = ("not contain enough information", "don't know", "cannot be answered from")
MIN_SCORE = 0.25         # Standard RAG threshold
GRADE_MIN_SCORE = 0.20   # Corrective RAG: pre-filter threshold before calling LLM grader
MAX_ACCEPT = 4           # Max verified chunks passed to generator

LLM_REGISTRY = {
    "flan-t5-small": {"repo": "google/flan-t5-small", "type": "seq2seq"},
    "flan-t5-base": {"repo": "google/flan-t5-base", "type": "seq2seq"},
    "qwen2.5-0.5b-instruct": {"repo": "Qwen/Qwen2.5-0.5B-Instruct", "type": "causal"},
}

ANSWER_SYSTEM = (
    "You are an intelligent course assistant. Answer the question using ONLY the verified context. "
    "Be accurate, clear, and concise in 1 to 3 complete sentences. "
    f"If the answer is not supported by the context, reply exactly: {UNANSWERED}"
)
SUMMARIZE_SYSTEM = (
    "You are an academic summarizer. Summarize the key concepts from the provided course context "
    "into 2 to 4 clear, concise sentences using only facts present in the context."
)
GRADER_SYSTEM = (
    "You are a strict relevance grader. Decide whether the context contains facts that help answer the question. "
    "Reply with only one word: yes or no."
)
REWRITE_SYSTEM = (
    "You rewrite student questions into clear, keyword-rich search queries for a course knowledge base. "
    "Reply with the rewritten query only."
)

STOPWORDS = set(
    "a an the is are was were be to of in on for and or what which who whom how why when where does do did can could "
    "should would i my me we you your it its this that these those with about from by at as into than then there "
    "stop use used using get make tell give please explain".split()
)

_LLMS = {}


def load_llm(name):
    if name in _LLMS:
        return _LLMS[name]
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    cfg = LLM_REGISTRY[name]
    path = cfg["repo"]
    marker = f"{SAVED_LLM_DIR}/model_name.txt"
    if os.path.exists(marker) and open(marker).read().strip() == name:
        path = SAVED_LLM_DIR

    tok = AutoTokenizer.from_pretrained(path)
    model_cls = AutoModelForSeq2SeqLM if cfg["type"] == "seq2seq" else AutoModelForCausalLM
    model = model_cls.from_pretrained(path, torch_dtype=torch.float32).eval()
    _LLMS[name] = (tok, model)
    return _LLMS[name]


def unload_llm(name):
    _LLMS.pop(name, None)
    gc.collect()


def chat(name, system, user, max_new_tokens=120, num_beams=1):
    import torch
    tok, model = load_llm(name)
    with torch.no_grad():
        if LLM_REGISTRY[name]["type"] == "seq2seq":
            prompt = f"{system}\n\n{user}\nAnswer:"
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1536)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False)
            return tok.decode(out[0], skip_special_tokens=True).strip()

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=1536)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------- Tools Suite
_SAFE_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg, ast.Mod: op.mod,
}


def _eval_ast(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_ast(node.operand))
    raise ValueError("Unsupported math expression")


def calculator_tool(question):
    """Extracts and safely evaluates arithmetic expressions (e.g., precision, weighted grades)."""
    cleaned = question.replace("%", "/100").replace("×", "*").replace("÷", "/")
    exprs = re.findall(r"[0-9\.\+\-\*\/\(\)\s]{3,}", cleaned)
    candidates = [e.strip() for e in exprs if any(c in e for c in "+-*/") and re.search(r"\d", e)]
    if not candidates:
        return None
    expr = max(candidates, key=len)
    try:
        val = _eval_ast(ast.parse(expr, mode="eval").body)
        formatted = f"{val:.4f}".rstrip("0").rstrip(".") if isinstance(val, float) else str(val)
        return f"Calculated result for `{expr}` = **{formatted}**."
    except Exception:
        return None


def data_analysis_tool(question):
    """Performs Pandas metadata & corpus statistics analysis on the knowledge base."""
    cat_df = ing.catalog()
    chunks_df = ing.get_all_chunks_df()
    if cat_df.empty:
        return "The knowledge base is currently empty."

    q = question.lower()
    total_files = len(cat_df)
    total_chunks = int(cat_df["chunks"].sum())
    courses = sorted(cat_df["course"].unique().tolist())
    file_types = cat_df["type"].value_counts().to_dict()
    chunks_by_course = cat_df.groupby("course")["chunks"].sum().to_dict()
    avg_len = int(chunks_df["text"].str.len().mean()) if not chunks_df.empty else 0

    parts = [
        f"The knowledge base contains **{total_files} files** across **{len(courses)} courses** "
        f"({', '.join(courses)}), indexed into **{total_chunks} total chunks** (average {avg_len} chars/chunk).",
        f"Chunks per course: {', '.join(f'{k}: {v}' for k, v in chunks_by_course.items())}.",
        f"File formats: {', '.join(f'{k} ({v})' for k, v in file_types.items())}.",
    ]
    if "longest" in q or "most chunks" in q:
        top_file = cat_df.sort_values("chunks", ascending=False).iloc[0]
        parts.append(f"The file with the most chunks is **{top_file['file']}** ({top_file['chunks']} chunks).")
    return " ".join(parts)


# ---------------------------------------------------------------- Retrieval & Grading
def retrieve(query, k=6, course=None):
    vec = ing.get_embedder().encode([query], normalize_embeddings=True).tolist()
    where = {"course": course} if course and course != "All" else None
    res = ing.get_collection().query(query_embeddings=vec, n_results=k, where=where)
    return [
        {"id": i, "text": t, "meta": m, "score": 1 - d}
        for i, t, m, d in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def grade_chunk(question, chunk_text, name):
    verdict = chat(
        name,
        GRADER_SYSTEM,
        f"Context: {chunk_text}\n\nQuestion: {question}\n\nDoes the context help answer the question? Answer yes or no.",
        max_new_tokens=3,
    )
    return verdict.strip().lower().startswith("yes")


def grade_hits(question, hits, name, max_accept=MAX_ACCEPT):
    accepted, rows = [], []
    for h in sorted(hits, key=lambda x: -x["score"]):
        if len(accepted) >= max_accept:
            verdict = "skipped (enough verified chunks)"
        elif h["score"] < GRADE_MIN_SCORE:
            verdict = "no (similarity too low)"
        elif grade_chunk(question, h["text"], name):
            verdict = "yes"
            accepted.append(h)
        else:
            verdict = "no"
        rows.append(f"[{verdict}] {h['meta']['file']} (part {h['meta']['page']}, sim={h['score']:.2f})")
    return accepted, rows


def keyword_query(question):
    tokens = re.findall(r"[a-zA-Z0-9%']+", question.lower())
    return " ".join(t for t in tokens if t not in STOPWORDS)


def rewrite_query(question, name, tried):
    user = f"Question: {question}\nAlready tried (use different keywords): {'; '.join(tried)}\nRewritten query:"
    new = chat(name, REWRITE_SYSTEM, user, max_new_tokens=40).strip().strip('"')
    lowered = [t.lower() for t in tried]
    if len(new) < 6 or new.lower() in lowered:
        new = keyword_query(question)
        if new.lower() in lowered:
            new = f"{new} definition explanation"
    return new


def decompose_multi_step(question):
    """Decomposes multi-part or comparative questions into focused sub-queries."""
    q = question.strip()
    m = re.search(r"(?:difference between|compare)\s+(.+?)\s+and\s+(.+?)(?:\?|$| in )", q, flags=re.I)
    if m:
        t1, t2 = m.group(1).strip(), m.group(2).strip()
        return [f"What is {t1}?", f"What is {t2}?"]

    if " and " in q.lower() and len(q.split()) >= 9:
        parts = [p.strip(" ?.") for p in re.split(r"\s+and\s+", q, maxsplit=1, flags=re.I) if len(p.strip()) > 10]
        if len(parts) == 2:
            return parts
    return [q]


# ---------------------------------------------------------------- Answer Synthesis
def is_refusal(answer):
    a = answer.lower().replace("\u2019", "'")
    return any(m in a for m in UNANSWERED_MARKERS)


def source_labels(hits):
    labels, seen = [], set()
    for h in hits:
        m = h["meta"]
        label = f"{m['file']} (course: {m['course']}, {m['type']}, part {m['page']})"
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def clean_answer(text):
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return re.sub(r"\s+([.,;:!?])", r"\1", text).strip()


def is_bad_answer(text):
    return len(re.findall(r"[A-Za-z0-9]", text)) < 3


def extractive_fallback(question, hits):
    q_words = set(keyword_query(question).split())
    best, best_score = "", -1
    for h in hits:
        for sent in re.split(r"(?<=[.!?])\s+", h["text"]):
            score = len(q_words & set(keyword_query(sent).split()))
            if score > best_score:
                best, best_score = sent.strip(), score
    return best


def generate_answer(question, hits, model_name, is_summary=False):
    context = "\n\n".join(h["text"] for h in hits)[:3200]
    sys_prompt = SUMMARIZE_SYSTEM if is_summary else ANSWER_SYSTEM
    user = f"Context:\n{context}\n\nQuestion: {question}"
    answer = clean_answer(chat(model_name, sys_prompt, user, max_new_tokens=140, num_beams=2))
    if is_bad_answer(answer):
        retry_sys = sys_prompt + " Write a complete, informative sentence."
        answer = clean_answer(chat(model_name, retry_sys, user, max_new_tokens=140, num_beams=4))
    if is_bad_answer(answer):
        answer = extractive_fallback(question, hits)
    return answer


# ---------------------------------------------------------------- Agent Planner
def plan_agent_strategy(question):
    """Determines the optimal tool/strategy before retrieving."""
    q = question.lower().strip()

    if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", q) and any(
        w in q for w in ("calculate", "compute", "what is", "solve", "evaluate", "=")
    ):
        return "calculator"

    if any(
        phrase in q
        for phrase in (
            "how many files", "how many courses", "how many chunks",
            "dataset statistics", "knowledge base stats", "analyze the dataset",
            "file types", "most chunks", "list all courses", "data summary",
        )
    ):
        return "data_analysis"

    if any(w in q for w in ("summarize", "summary of", "overview of", "key takeaways")):
        return "summarize"

    sub_queries = decompose_multi_step(question)
    if len(sub_queries) > 1:
        return "multi_step_rag"

    return "corrective_rag"


# ---------------------------------------------------------------- Main Execution Pipeline
def run_agentic_rag(question, model_name, k=6, course=None, agentic=True, max_rewrites=2):
    start = time.time()
    trace = []
    accepted = []
    rewrites = 0
    retrieved_total = 0
    tool_used = "Standard RAG"

    # ---------------- BASELINE MODE: Standard RAG
    if not agentic:
        hits = retrieve(question, k=min(k, 4), course=course)
        retrieved_total = len(hits)
        trace.append({"step": "1. Standard Retrieve", "detail": f'Query: "{question}" -> {len(hits)} chunks (no grading)'})
        accepted = hits if hits and hits[0]["score"] >= MIN_SCORE else []
        if not accepted:
            trace.append({"step": "2. Decision", "detail": "Top similarity below threshold -> return unanswered."})
            answer = UNANSWERED
        else:
            trace.append({"step": "2. Generate", "detail": f"Generating answer from {len(accepted)} ungraded chunks."})
            answer = generate_answer(question, accepted, model_name) or UNANSWERED
            if is_refusal(answer):
                answer, accepted = UNANSWERED, []

        return {
            "answer": answer,
            "sources": source_labels(accepted),
            "hits": accepted,
            "trace": trace,
            "tool": tool_used,
            "rewrites": 0,
            "retrieved": retrieved_total,
            "accepted": len(accepted),
            "latency": time.time() - start,
            "agentic": False,
        }

    # ---------------- AGENTIC + CORRECTIVE RAG MODE
    strategy = plan_agent_strategy(question)
    trace.append({"step": "1. Agent Planner", "detail": f"Selected strategy: **{strategy}**"})

    if strategy == "calculator":
        calc_out = calculator_tool(question)
        if calc_out:
            tool_used = "Calculator Tool"
            trace.append({"step": "2. Calculator Tool", "detail": calc_out})
            return {
                "answer": calc_out,
                "sources": ["Built-in Python Calculator Tool"],
                "hits": [],
                "trace": trace,
                "tool": tool_used,
                "rewrites": 0,
                "retrieved": 0,
                "accepted": 0,
                "latency": time.time() - start,
                "agentic": True,
            }
        strategy = "corrective_rag"

    if strategy == "data_analysis":
        tool_used = "Data Analysis Tool (Pandas)"
        stats_answer = data_analysis_tool(question)
        trace.append({"step": "2. Data Analysis Tool", "detail": "Queried ChromaDB metadata & Pandas catalog."})
        return {
            "answer": stats_answer,
            "sources": ["Knowledge Base Catalog (ChromaDB Metadata)"],
            "hits": [],
            "trace": trace,
            "tool": tool_used,
            "rewrites": 0,
            "retrieved": 0,
            "accepted": 0,
            "latency": time.time() - start,
            "agentic": True,
        }

    if strategy == "summarize":
        tool_used = "Summarization Tool + Corrective RAG"
        hits = retrieve(question, k=k, course=course)
        retrieved_total = len(hits)
        good, rows = grade_hits(question, hits, model_name, max_accept=4)
        trace.append({"step": "2. Retrieve & Grade for Summary", "detail": "\n".join(rows)})
        if not good:
            answer = UNANSWERED
        else:
            accepted = good
            answer = generate_answer(question, accepted, model_name, is_summary=True) or UNANSWERED
        return {
            "answer": answer,
            "sources": source_labels(accepted),
            "hits": accepted,
            "trace": trace,
            "tool": tool_used,
            "rewrites": 0,
            "retrieved": retrieved_total,
            "accepted": len(accepted),
            "latency": time.time() - start,
            "agentic": True,
        }

    if strategy == "multi_step_rag":
        tool_used = "Multi-Step Planner + Corrective RAG"
        sub_queries = decompose_multi_step(question)
        trace.append({"step": "2. Sub-Query Decomposition", "detail": f"Split into {len(sub_queries)} sub-queries: {sub_queries}"})
        seen_ids = set()
        for idx, sq in enumerate(sub_queries, 1):
            hits = retrieve(sq, k=max(4, k // 2 + 1), course=course)
            retrieved_total += len(hits)
            fresh = [h for h in hits if h["id"] not in seen_ids]
            seen_ids.update(h["id"] for h in hits)
            good, rows = grade_hits(sq, fresh, model_name, max_accept=2)
            if not good and max_rewrites > 0:
                rq = rewrite_query(sq, model_name, [sq])
                rewrites += 1
                hits2 = retrieve(rq, k=4, course=course)
                retrieved_total += len(hits2)
                fresh2 = [h for h in hits2 if h["id"] not in seen_ids]
                seen_ids.update(h["id"] for h in hits2)
                good, rows2 = grade_hits(sq, fresh2, model_name, max_accept=2)
                rows += [f"[Rewritten: {rq}]"] + rows2
            accepted.extend(good)
            trace.append({"step": f"3.{idx} Sub-Query Retrieval & Grade", "detail": "\n".join(rows)})

        accepted = sorted(accepted, key=lambda h: -h["score"])[:MAX_ACCEPT]
    else:
        tool_used = "Corrective RAG (Grader + Query Rewriter)"
        seen, tried = set(), [question]
        query = question
        for attempt in range(max_rewrites + 1):
            hits = retrieve(query, k=k, course=course)
            retrieved_total += len(hits)
            trace.append({"step": f"2.{attempt + 1} Retrieve", "detail": f'Query: "{query}" -> {len(hits)} chunks'})
            fresh = [h for h in hits if h["id"] not in seen]
            seen.update(h["id"] for h in hits)
            good, rows = grade_hits(question, fresh, model_name, max_accept=3)
            accepted.extend(good)
            trace.append({
                "step": f"3.{attempt + 1} Grade Chunks",
                "detail": "\n".join(rows) + f"\n=> {len(good)} verified of {len(fresh)} new chunks",
            })
            if accepted or attempt == max_rewrites:
                break
            query = rewrite_query(question, model_name, tried)
            tried.append(query)
            rewrites += 1
            trace.append({"step": f"4.{attempt + 1} Query Refinement", "detail": f'Weak context -> rewritten query: "{query}"'})

        accepted = sorted(accepted, key=lambda h: -h["score"])[:MAX_ACCEPT]

    if not accepted:
        trace.append({"step": "Final Decision", "detail": "No verified chunks found -> safe refusal to prevent hallucination."})
        answer = UNANSWERED
    else:
        trace.append({"step": "Final Decision", "detail": f"Synthesizing grounded answer from {len(accepted)} verified chunk(s)."})
        answer = generate_answer(question, accepted, model_name) or UNANSWERED
        if is_refusal(answer):
            answer, accepted = UNANSWERED, []

    return {
        "answer": answer,
        "sources": source_labels(accepted),
        "hits": accepted,
        "trace": trace,
        "tool": tool_used,
        "rewrites": rewrites,
        "retrieved": retrieved_total,
        "accepted": len(accepted),
        "latency": time.time() - start,
        "agentic": True,
    }


def format_trace(out):
    return "\n".join(f"- **{t['step']}**: {t['detail']}" for t in out["trace"])
