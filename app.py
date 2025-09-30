# ai_multi_tool_assistant_task5_v2_fixed.py

import os
import io
import re
import numpy as np
import streamlit as st
import wikipedia

# Embeddings / FAISS
import faiss
from sentence_transformers import SentenceTransformer

# Gemini
import google.generativeai as genai

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="Smart AI Multi-Tool Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Smart AI Assistant: Multi-Tool Answer Generator")

st.markdown("""
This assistant combines three tools:  
- GlobalMart RAG: internal product knowledge (FAISS + SentenceTransformers)  
- Calculator: math expressions  
- Wikipedia: general knowledge  

Use the sidebar to provide your Google Gemini API key and upload GlobalMart docs (TXT).  
You can ask multi-part questions separated by AND or ;.
""")

# ----------------------------
# Sidebar: API key + Upload Docs
# ----------------------------
st.sidebar.header("Configuration")

gemini_api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")
use_demo_docs = st.sidebar.checkbox("Use sample GlobalMart docs (demo)", value=True)

uploaded_files = st.sidebar.file_uploader(
    "Upload GlobalMart text files (.txt) — optional",
    type=["txt"],
    accept_multiple_files=True
)

if not gemini_api_key:
    st.sidebar.warning("Add your Gemini API key to enable LLM planner/synthesizer.")
else:
    try:
        genai.configure(api_key=gemini_api_key)
    except Exception as e:
        st.sidebar.error(f"Failed to configure Gemini key: {e}")

# ----------------------------
# RAG / FAISS Utilities
# ----------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384  # dimension for this model

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

def build_faiss_index(text_list):
    embedder = load_embedder()
    embeddings = embedder.encode(text_list, convert_to_numpy=True, show_progress_bar=False)
    vectors = np.array(embeddings).astype("float32")
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    id_to_doc = {i: text_list[i] for i in range(len(text_list))}
    return index, id_to_doc

def rag_search(index, id2doc, query, k=3):
    embedder = load_embedder()
    qvec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    if index.ntotal == 0:
        return []
    D, I = index.search(qvec, k)
    results = [id2doc.get(int(idx), "") for idx in I[0] if idx != -1]
    return results

# ----------------------------
# Load / Build Knowledge Base
# ----------------------------
SAMPLE_DOCS = [
    "GlobalMart: We offer 10% discount vouchers on orders above $100. Discounts vary seasonally.",
    "GlobalMart returns policy: Items can be returned within 14 days. Electronics: 7 days.",
    "GlobalMart loyalty program: Earn points on each purchase; redeem for vouchers or free shipping."
]

texts = []
if uploaded_files:
    for file in uploaded_files:
        try:
            raw = file.read()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            parts = [p.strip() for p in re.split(r'\n{2,}|\r\n{2,}', raw) if p.strip()]
            if not parts:
                parts = [raw.strip()]
            texts.extend(parts)
        except Exception as e:
            st.sidebar.error(f"Could not read {file.name}: {e}")
elif use_demo_docs:
    texts = SAMPLE_DOCS.copy()

if texts:
    try:
        rag_index, id_to_doc = build_faiss_index(texts)
    except Exception as e:
        st.error(f"Error building RAG index: {e}")
        rag_index, id_to_doc = faiss.IndexFlatL2(EMBED_DIM), {}
else:
    rag_index, id_to_doc = faiss.IndexFlatL2(EMBED_DIM), {}

# ----------------------------
# Tools
# ----------------------------
def globalmart_rag(query, top_k=3):
    if rag_index.ntotal == 0:
        return "GlobalMart knowledge base is empty. Upload docs or enable sample docs."
    results = rag_search(rag_index, id_to_doc, query, k=top_k)
    if not results:
        return "No relevant GlobalMart documents found."
    return "\n\n---\n\n".join(results)

def safe_calc(expr):
    try:
        expr_clean = re.sub(r'[^0-9\.\+\-\*\/\(\)\s]', '', expr)
        if not expr_clean.strip():
            return "Invalid or empty expression."
        result = eval(expr_clean, {"builtins": None}, {})
        return str(result)
    except Exception as e:
        return f"Calculator Error: {e}"

def wiki_search(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except Exception as e:
        return f"Wikipedia Error: {e}"

# ----------------------------
# Gemini LLM Helpers
# ----------------------------
def init_gemini():
    try:
        # ✅ FIXED: correct latest supported model
        return genai.GenerativeModel("models/gemini-2.5-pro")
    except Exception as e:
        st.error(f"Gemini init error: {e}")
        return None

@st.cache_resource
def get_gemini():
    return init_gemini()

gemini = get_gemini() if gemini_api_key else None
if gemini_api_key and not gemini:
    st.stop()

def run_planner(query):
    prompt = f"""
You are an agent router. Split the user's query into parts if needed. Tools:
- GlobalMart RAG System
- Calculator
- Wikipedia Search

User query: "{query}"

Output ONLY in this format:
Part 1: Tool: <Tool Name> Input: <Input>
Part 2: Tool: <Tool Name> Input: <Input>
(Only include necessary parts)
"""
    try:
        resp = gemini.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        st.error(f"Planner error: {e}")
        return None

def run_synth(user_query, tool_outputs):
    parts_text = ""
    for tn, ti, tr in tool_outputs:
        parts_text += f"{tn} (Input: {ti}): {tr}\n\n"
    prompt = f"""
User asked: "{user_query}"

Tool outputs: {parts_text}

Generate a clear, concise, friendly final answer.
"""
    try:
        resp = gemini.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        st.error(f"Synthesizer error: {e}")
        return "Could not generate final answer."

# ----------------------------
# Agent Execution
# ----------------------------
def execute_agent(query):
    plan = run_planner(query)
    if not plan:
        return None, None

    parts = re.findall(
        r"Part\s*\d+:\s*Tool:\s*(.+?)\s*Input:\s*(.+?)(?=(?:\nPart\s*\d+:|$))",
        plan, flags=re.DOTALL | re.IGNORECASE
    )
    if not parts:
        st.error("Planner output could not be parsed:\n" + plan)
        return None, None

    outputs = []
    for tool_name, tool_input in parts:
        tool_name = tool_name.strip()
        tool_input = tool_input.strip().strip('"').strip("'")
        if "globalmart" in tool_name.lower():
            res = globalmart_rag(tool_input)
        elif "calculator" in tool_name.lower():
            res = safe_calc(tool_input)
        elif "wikipedia" in tool_name.lower():
            res = wiki_search(tool_input)
        else:
            res = f"Unknown tool: {tool_name}"
        outputs.append((tool_name, tool_input, res))

    final_ans = run_synth(query, outputs)
    return outputs, final_ans

# ----------------------------
# Streamlit UI
# ----------------------------
st.markdown("### Ask your question below 👇")
user_query = st.text_input("Your Question:")

if st.button("Ask AI Assistant"):
    if not gemini_api_key:
        st.error("Please set your Gemini API key in the sidebar.")
    elif not user_query.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Thinking..."):
            tool_outputs, final_answer = execute_agent(user_query)

        if tool_outputs is None:
            st.error("Agent failed to produce an answer.")
        else:
            st.success("Tool Execution Complete!")
            st.markdown("#### Tool Outputs:")
            for i, (tn, ti, tr) in enumerate(tool_outputs, start=1):
                st.markdown(f"Part {i}: {tn} (Input: {ti})")
                st.write(tr)
            st.markdown("#### Final Synthesized Answer:")
            st.info(final_answer)
