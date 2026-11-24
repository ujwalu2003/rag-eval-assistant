import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
collection = chromadb.PersistentClient("db").get_collection("docs")

question = input("Ask a question: ")
results = collection.query(
    query_embeddings=[model.encode(question).tolist()],
    n_results=3,
)

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print("\n--- From", meta["source"], "page", meta["page"])
    print(doc)