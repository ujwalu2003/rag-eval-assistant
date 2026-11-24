import chromadb

col = chromadb.PersistentClient("db").get_collection("docs")
data = col.get(include=["documents", "metadatas"])

word = input("Search for text: ")
for doc, m in zip(data["documents"], data["metadatas"]):
    if word.lower() in doc.lower():
        print("\n--", m["source"], "page", m["page"])
        print(doc[:300])