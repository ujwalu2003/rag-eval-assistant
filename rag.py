import os
import time
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder

MODEL = "openai/gpt-oss-20b"

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
reranker = None
collection = chromadb.PersistentClient("db").get_collection("docs")
client = Groq(api_key=os.environ["GROQ_API_KEY"])

def ask_llm(prompt, retries=3):
    api_failed = False
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0 if attempt == 0 else 0.3,
            )
        except Exception as e:
            msg = str(e)
            print("  API error:", msg[:250])
            if "per day" in msg or "TPD" in msg:
                raise SystemExit("Daily limit reached. Stop here.")
            api_failed = True
            time.sleep(20)
            continue
        api_failed = False
        text = r.choices[0].message.content or ""
        if text.strip():
            return text
        print("  Empty reply, finish_reason =", r.choices[0].finish_reason, "- retrying")
    if api_failed:
        raise RuntimeError("API failed after retries")
    return ""

def answer(question, k=4, rerank=False, fetch=15):
    n = fetch if rerank else k
    results = collection.query(
        query_embeddings=[model.encode(question).tolist()],
        n_results=n,
    )
    chunks = results["documents"][0]
    sources = results["metadatas"][0]

    if rerank:
        global reranker
        if reranker is None:
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        scores = reranker.predict([(question, c) for c in chunks])

    context = "\n\n".join(chunks)
    prompt = (
        "You are a question-answering assistant. Use ONLY the context below. "
        "Do NOT use outside knowledge, even if you know the answer. "
        "If the context does not directly answer the question, "
        "reply exactly: Not found in documents.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    return ask_llm(prompt), sources

if __name__ == "__main__":
    q = input("Ask a question: ")
    text, sources = answer(q, k=4, rerank=True)
    print("\nAnswer:", text)
    if "Not found" not in text:
        print("\nSources:")
        for s in sources:
            print("-", s["source"], "page", s["page"])