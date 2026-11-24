import chromadb
from collections import Counter

col = chromadb.PersistentClient("db").get_collection("docs")
data = col.get(include=["documents", "metadatas"])

print(Counter(m["source"] for m in data["metadatas"]))

found = False
for doc, m in zip(data["documents"], data["metadatas"]):
    if "gpt3" in m["source"] and "2048" in doc:
        print("Found 2048 in", m["source"], "page", m["page"])
        found = True
        break
if not found:
    print("2048 not found in any gpt3 chunk")