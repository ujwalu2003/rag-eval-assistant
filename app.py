import os
import sys
import subprocess
import streamlit as st

if not os.path.exists("db"):
    with st.spinner("First start: building the search index (a few minutes)..."):
        subprocess.run([sys.executable, "ingest.py"], check=True)

from rag import answer

K = 8            # change after k8_strict confirms the best setting
RERANK = False

st.set_page_config(page_title="Paper Q&A", page_icon="📄")
st.title("📄 Ask the AI Research Papers")
st.caption("Answers come only from: Transformer, BERT, GPT-3, LoRA, RAG, Llama 2")

question = st.text_input("Ask a question")

if question:
    with st.spinner("Searching the papers..."):
        try:
            text, sources = answer(question, k=K, rerank=RERANK)
        except SystemExit:
            st.error("The free API limit was reached. Please try again later.")
            st.stop()
        except Exception as e:
            st.error("Something went wrong. Please try again.")
            st.stop()

    st.subheader("Answer")
    st.write(text)

    if "Not found" not in text:
        st.subheader("Sources")
        seen = set()
        for s in sources:
            name = s["source"].replace("data\\", "").replace("data/", "")
            name = name.replace(".pdf.pdf", "").replace(".pdf", "")
            key = (name, s["page"])
            if key not in seen:
                seen.add(key)
                st.write(f"- **{name}**, page {s['page']}")