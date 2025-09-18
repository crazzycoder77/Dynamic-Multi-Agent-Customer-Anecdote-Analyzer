# ------------------------
# 0. Check and install required modules
# ------------------------
import importlib
import subprocess
import sys
import os
import pickle
import argparse
import re
import json

print("⚡ Checking required modules...")

required_modules = [
    "pandas",
    "faiss",
    "numpy",
    "tqdm",
    "sentence_transformers",
    "langchain_groq",
    "langgraph"
]

missing = []
for module in required_modules:
    try:
        importlib.import_module(module)
    except ImportError:
        missing.append(module)

if missing:
    print(f"⚡ Installing missing modules: {missing}")
    for module in missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", module])
else:
    print("✅ All required modules are installed.")

# ------------------------
# 1. Imports
# ------------------------
import pandas as pd
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# ------------------------
# 2. CLI Arguments
# ------------------------
parser = argparse.ArgumentParser(description="Dynamic Multi-Agent RAG with ChatGroq + Supervisor")
parser.add_argument("--api_key", required=True, help="Groq API Key")
parser.add_argument("--csv", required=False, help="Path to reviews CSV (needed if rebuilding)")
parser.add_argument("--rebuild", action="store_true", help="Force rebuild embeddings/index")
args = parser.parse_args()

os.environ["GROQ_API_KEY"] = args.api_key

# ------------------------
# 3. Embeddings + FAISS
# ------------------------
print("📦 Initializing embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
INDEX_FILE = "faiss_index.bin"
META_FILE = "review_meta.pkl"

def build_or_load_index(new_csv=None, force_rebuild=False):
    global texts, asins
    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE) and not force_rebuild:
        print("📦 Loading existing FAISS index and metadata...")
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "rb") as f:
            meta = pickle.load(f)
        texts, asins = meta["texts"], meta["asins"]
        print("✅ FAISS index loaded.")
        return index, texts, asins

    if new_csv is None:
        raise ValueError("CSV required to rebuild embeddings/index.")

    print(f"📦 Building new FAISS index from CSV: {new_csv}...")
    df = pd.read_csv(new_csv, encoding="utf-8", on_bad_lines='skip')
    df = df.dropna(subset=["ASIN", "Review Text"])
    texts = df["Review Text"].astype(str).tolist()
    asins = df["ASIN"].tolist()

    print("⚡ Generating embeddings...")
    embeddings = embed_model.encode(texts, batch_size=64, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump({"texts": texts, "asins": asins}, f)
    print("✅ FAISS index built and saved.")
    return index, texts, asins

# Load or build index
if args.rebuild:
    csv_path = args.csv or input("Enter CSV path to rebuild index: ").strip()
    index, texts, asins = build_or_load_index(new_csv=csv_path, force_rebuild=True)
else:
    try:
        index, texts, asins = build_or_load_index(force_rebuild=False)
    except FileNotFoundError:
        csv_path = args.csv or input("Enter CSV path to build index: ").strip()
        index, texts, asins = build_or_load_index(new_csv=csv_path, force_rebuild=True)

# ------------------------
# 4. Initialize LLM
# ------------------------
print("🤖 Initializing LLM...")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
print("✅ LLM initialized.")

# ------------------------
# 5. Caching for LLM checks
# ------------------------
relevance_cache = {}       # For aggregation node
classification_cache = {}  # For classification node

# ------------------------
# 6. Nodes
# ------------------------
def query_expansion_node(state):
    print("🔍 Running Query Expansion Node...")
    raw_query = state.get("query", "").strip()
    if not raw_query:
        raise ValueError("Query missing.")
    prompt = f"""
You are a query reformulation agent. Expand the user query into synonyms, numeric/unit info,
and different ways customers might express it.
Original query: "{raw_query}"
Expanded query:
"""
    result = llm.invoke(prompt)
    expanded = result if isinstance(result, str) else getattr(result, "content", str(result))
    state["expanded_query"] = expanded.strip().replace("\n", " ")
    print(f"✅ Query expanded: {state['expanded_query']}")
    return state

def dynamic_retrieval_node(state, batch_size=50, max_iter=20):
    print("🔎 Running Retrieval Node...")
    expanded_query = state.get("expanded_query")
    if not expanded_query:
        raise ValueError("Expanded query missing.")

    q_emb = embed_model.encode([expanded_query]).astype("float32")
    faiss.normalize_L2(q_emb)

    retrieved = []
    start_idx = 0
    iteration = 0
    total_reviews = len(asins)

    while iteration < max_iter and start_idx < total_reviews:
        current_batch_size = min(batch_size, total_reviews - start_idx)
        scores, idxs = index.search(q_emb, start_idx + current_batch_size)
        batch_idxs = idxs[0][start_idx:start_idx+current_batch_size]
        batch_scores = scores[0][start_idx:start_idx+current_batch_size]

        batch = [{"idx": int(i), "asin": asins[int(i)], "review_text": texts[int(i)], "score": float(s)}
                 for i, s in zip(batch_idxs, batch_scores)]
        retrieved.extend(batch)

        last_review_idx = batch[-1]["idx"]
        if last_review_idx in relevance_cache:
            relevant = relevance_cache[last_review_idx]
        else:
            relevance_prompt = f"""
For the query "{expanded_query}", is the following review still contextually relevant?
Review: "{batch[-1]['review_text']}"
Answer yes or no.
"""
            result = llm.invoke(relevance_prompt)
            answer = result.lower() if isinstance(result, str) else str(getattr(result, "content", result)).lower()
            relevant = "yes" in answer
            relevance_cache[last_review_idx] = relevant

        if not relevant:
            break

        start_idx += current_batch_size
        iteration += 1

    state["retrieved"] = retrieved
    print(f"✅ Retrieved {len(retrieved)} reviews.")
    return state

def dynamic_classification_node(state, batch_size=20):
    print("📝 Running Classification Node...")
    query = state.get("query", "")
    retrieved = state.get("retrieved", [])
    classified = []

    idx = 0
    while idx < len(retrieved):
        batch = retrieved[idx:idx+batch_size]
        reviews_block = "\n".join([f"{i+1}. [{r['asin']}] {r['review_text']}" for i, r in enumerate(batch)])

        cache_key = f"{query}_{idx}"
        if cache_key in classification_cache:
            batch_classified = classification_cache[cache_key]
        else:
            prompt = f"""
Classify these reviews for the query "{query}".
Return JSON array with fields:
- idx: <1-based index>
- asin
- match: true/false
- sentiment: positive/negative/neutral
- reason: one sentence

Reviews:
{reviews_block}
"""
            result = llm.invoke(prompt)
            text = result if isinstance(result, str) else getattr(result, "content", str(result))
            match_json = re.search(r"(\[.*\])", text, flags=re.DOTALL)
            batch_classified = []
            if match_json:
                try:
                    batch_classified = json.loads(match_json.group(1))
                except:
                    batch_classified = []
            classification_cache[cache_key] = batch_classified

        for obj in batch_classified:
            try:
                local_idx = int(obj.get("idx", 1)) - 1
                source = batch[local_idx]
                classified.append({
                    "idx": source["idx"],
                    "asin": source["asin"],
                    "match": bool(obj.get("match")),
                    "sentiment": obj.get("sentiment"),
                    "reason": obj.get("reason", "")
                })
            except:
                asin = obj.get("asin")
                if asin:
                    source_idx = next((r["idx"] for r in batch if r["asin"] == asin), None)
                    classified.append({
                        "idx": source_idx,
                        "asin": asin,
                        "match": bool(obj.get("match")),
                        "sentiment": obj.get("sentiment"),
                        "reason": obj.get("reason", "")
                    })

        if idx + batch_size < len(retrieved):
            last_review = batch[-1]['review_text']
            prompt_stop = f"""
For the query "{query}", should we continue classifying additional reviews beyond this one?
Last review: "{last_review}"
Answer yes or no.
"""
            answer = llm.invoke(prompt_stop)
            ans_text = answer.lower() if isinstance(answer, str) else str(getattr(answer, "content", answer)).lower()
            if "no" in ans_text:
                break

        idx += batch_size

    state["classified"] = classified
    print(f"✅ Classified {len(classified)} reviews.")
    return state

def aggregation_node(state, max_reviews=200):
    print("🧩 Running Aggregation Node...")
    classified = state.get("classified", [])
    retrieved = state.get("retrieved", [])
    idx_to_text = {r["idx"]: r["review_text"] for r in retrieved}

    matched = [r for r in classified if r.get("match")]
    if not matched:
        state["aggregated"] = {
            "count": 0,
            "asin_counts": {},
            "examples": [],
            "insights": "No matching reviews found for the query."
        }
        return state

    asin_counts = {}
    for r in matched:
        asin_counts[r["asin"]] = asin_counts.get(r["asin"], 0) + 1

    examples_with_text = []
    for r in matched:
        if len(examples_with_text) >= max_reviews:
            break
        review_text = idx_to_text.get(r["idx"], "").strip()
        if not review_text:
            continue

        last_idx = r["idx"]
        if last_idx in relevance_cache:
            relevant = relevance_cache[last_idx]
        else:
            relevance_prompt = f"""
You are a context-checking agent.

Query: "{state.get('query', '')}"

Review: "{review_text}"

Question: Is this review contextually relevant to the query? 
Answer "true" or "false".
"""
            relevance_result = llm.invoke(relevance_prompt)
            relevance_text = relevance_result if isinstance(relevance_result, str) else str(getattr(relevance_result, "content", relevance_result))
            relevant = "true" in relevance_text.lower()
            relevance_cache[last_idx] = relevant

        if relevant:
            examples_with_text.append({
                "asin": r["asin"],
                "sentiment": r.get("sentiment"),
                "reason": r.get("reason"),
                "review_text": review_text
            })

    reviews_text = "\n".join([f"- [{r['asin']}] {r['review_text']}" for r in examples_with_text])
    query = state.get("query", "")

    insight_prompt = f"""
You are an expert customer experience analyst.

Query: "{query}"

Relevant Reviews:
{reviews_text}

Tasks:
1. Summarize key issues or pain points customers are expressing.
2. Identify patterns or trends across ASINs.

Output in JSON format:
{{
  "summary": "...",
  "patterns": "... "
}}
"""
    result = llm.invoke(insight_prompt)
    text = result if isinstance(result, str) else str(getattr(result, "content", str(result)))

    match_json = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    insights = {}
    if match_json:
        try:
            insights = json.loads(match_json.group(1))
        except:
            insights = {"summary": text.strip(), "patterns": ""}

    state["aggregated"] = {
        "count": len(matched),
        "asin_counts": asin_counts,
        "examples": examples_with_text,
        "insights": insights
    }
    print(f"✅ Aggregated insights from {len(matched)} matching reviews.")
    return state

# ------------------------
# 6b. Supervisor Node
# ------------------------
MAX_SUPERVISOR_LOOPS = 3
def supervisor_node(state):
    print("🛠️ Running Supervisor Node...")
    if "conversation_memory" not in state:
        state["conversation_memory"] = []
    if "supervisor_loop_count" not in state:
        state["supervisor_loop_count"] = 0

    aggregated = state.get("aggregated", {})
    query = state.get("query", "").strip()
    issues = []

    if not aggregated or aggregated.get("count", 0) == 0:
        issues.append("No relevant matching reviews found.")
    elif len(aggregated.get("examples", [])) == 0:
        issues.append("No example reviews to show.")

    state["next_node"] = "output"
    suggested_query = None

    if issues and state["supervisor_loop_count"] < MAX_SUPERVISOR_LOOPS:
        prompt = f"""
You are an intelligent supervisor analyzing customer reviews.

Original Query: "{query}"
Issues Detected: {issues}
Aggregated Insights: {aggregated.get('insights', {})}

Tasks:
1. Suggest an improved or expanded query that could better capture relevant reviews.
2. Provide a short reason for the suggested improvement.

Output JSON with fields:
{{
  "improved_query": "...",
  "reason": "..."
}}
"""
        result = llm.invoke(prompt)
        text = result if isinstance(result, str) else str(getattr(result, "content", str(result)))
        match_json = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if match_json:
            try:
                data = json.loads(match_json.group(1))
                suggested_query = data.get("improved_query")
            except:
                suggested_query = None

        if suggested_query:
            state["query"] = suggested_query
            state["next_node"] = "query_expansion"
            state["supervisor_loop_count"] += 1

    state["conversation_memory"].append({
        "query": query,
        "aggregated": aggregated,
        "issues": issues,
        "suggested_query": suggested_query
    })

    state["supervisor_review"] = {
        "query": query,
        "issues": issues,
        "conversation_history": state["conversation_memory"][-5:],
        "action_suggested": "Improved query generated and loop back" if suggested_query else "OK",
        "loop_count": state["supervisor_loop_count"]
    }
    print(f"✅ Supervisor Node done. Issues: {issues if issues else 'None'}")
    return state

# ------------------------
# 7. Output Node (CLI)
# ------------------------
def output_node(state):
    print("✅ Generating Output...")
    aggregated = state.get("aggregated", {})
    supervisor = state.get("supervisor_review", {})

    print("\n=== Query Results ===\n")
    print(f"Total Matching Reviews: {aggregated.get('count', 0)}\n")

    print("ASIN Counts:")
    for asin, count in aggregated.get("asin_counts", {}).items():
        print(f"  - {asin}: {count}")

    print("\nExample Reviews:")
    for ex in aggregated.get("examples", []):
        print(f"\nASIN: {ex['asin']}")
        print(f"Sentiment: {ex['sentiment']}")
        print(f"Reason: {ex['reason']}")
        print(f"Review Text: {ex['review_text']}")  # full review

    print("\nInsights Summary:")
    insights = aggregated.get("insights", {})
    print(f"  - Summary: {insights.get('summary', '')}")
    print(f"  - Patterns: {insights.get('patterns', '')}")

    print("\n=== Supervisor Review ===\n")
    print(f"Query: {supervisor.get('query','')}")

    issues = supervisor.get("issues", [])
    if issues:
        print("Issues identified:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No major issues identified by supervisor.")

    print(f"Action Suggested: {supervisor.get('action_suggested','')}")
    print("-" * 60)
    return END

# ------------------------
# 8. Build LangGraph
# ------------------------
print("🛠️ Building LangGraph workflow...")
graph = StateGraph(dict)
graph.add_node("query_expansion", query_expansion_node)
graph.add_node("retrieval", dynamic_retrieval_node)
graph.add_node("classification", dynamic_classification_node)
graph.add_node("aggregation", aggregation_node)
graph.add_node("supervisor", supervisor_node)
graph.add_node("output", output_node)

graph.set_entry_point("query_expansion")
graph.add_edge("query_expansion", "retrieval")
graph.add_edge("retrieval", "classification")
graph.add_edge("classification", "aggregation")
graph.add_edge("aggregation", "supervisor")
graph.add_edge("supervisor", "output")
graph.add_edge("output", END)

compiled = graph.compile()
print("✅ LangGraph compiled successfully.")

# ------------------------
# 9. CLI Loop
# ------------------------
print("\n🤖 Dynamic Multi-Agent Customer Anecdote Analyzer")
print("Type 'exit' to quit.\n")

while True:
    q = input("Query> ").strip()
    if q.lower() in ["exit", "quit"]:
        break
    state = {"query": q}
    compiled.invoke(state)
    print("-" * 60)
