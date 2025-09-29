import streamlit as st
import google.generativeai as genai
import os

# ============ 1. SIDEBAR CONTROLS ============
st.sidebar.header("⚙️ Settings")

api_key = st.sidebar.text_input("🔑 Gemini API Key", type="password")
model_name = st.sidebar.selectbox(
    "📦 Select Model",
    [
        "models/gemini-2.5-flash",
        "models/gemini-flash-latest",
    ]
)
temperature = st.sidebar.slider("🌡️ Temperature", 0.0, 1.0, 0.7, 0.1)
max_tokens = st.sidebar.slider("📝 Max Tokens", 100, 2048, 500, 50)

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = []

if st.sidebar.button("🆕 New Chat"):
    st.session_state.messages = []

# ============ 2. INITIALIZE ============
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 RAG-Powered Conversational AI")

if not api_key:
    st.warning("Please enter your Gemini API key in the sidebar to start.")
    st.stop()

# Configure Gemini
genai.configure(api_key=api_key)
llm = genai.GenerativeModel(model_name)

# ============ 3. KNOWLEDGE BASE & SEARCH FUNCTION ============
def load_knowledge_base(file_path="knowledge_base.txt"):
    if not os.path.exists(file_path):
        return ["Knowledge base not found."]
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
    return chunks

knowledge_chunks = load_knowledge_base()

def semantic_search(query, top_k=3):
    query_words = query.lower().split()
    scored_chunks = []
    for chunk in knowledge_chunks:
        score = sum(chunk.lower().count(word) for word in query_words)
        scored_chunks.append((score, chunk))
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    top_chunks = [chunk for score, chunk in scored_chunks if score > 0]
    return top_chunks[:top_k] if top_chunks else ["No relevant information found in knowledge base."]

# ============ 4. RAG PIPELINE FUNCTION ============
def rag_pipeline(question):
    try:
        context_docs = semantic_search(question)
        context = "\n".join(context_docs)
        prompt = f"""
You are a friendly AI assistant. 
Answer in **2–3 short sentences**, human-like, not robotic.
Don’t just repeat the question — give a natural confirmation and clarity.

Context:
{context}

Question: {question}

Answer:
"""
        response = llm.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )

        if response and response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                return candidate.content.parts[0].text.strip()

        return "⚠️ Sorry, I couldn't generate a valid answer this time."

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ============ 5. MAIN CHAT UI ============
user_input = st.text_input("💬 Ask your question:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        answer = rag_pipeline(user_input)
    st.session_state.messages.append({"role": "assistant", "content": answer})

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**🧑 You:** {msg['content']}")
    else:
        st.markdown(f"**🤖 AI:** {msg['content']}")
