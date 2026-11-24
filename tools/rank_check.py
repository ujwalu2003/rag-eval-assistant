import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
col = chromadb.PersistentClient("db").get_collection("docs")

q = input("Question: ")
phrase = input("Phrase the right chunk must contain: ")
res = col.query(query_embeddings=[model.encode(q).tolist()], n_results=50)

for rank, doc in enumerate(res["documents"][0], 1):
    if phrase.lower() in doc.replace("\n", " ").lower():
        print("First match at rank", rank)
        break
else:
    print("Not in the top 50")