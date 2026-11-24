import glob
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Model that turns text into numbers
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Database stored in a folder called "db"
client = chromadb.PersistentClient("db")
collection = client.get_or_create_collection("docs")

# Cut text into overlapping pieces
def chunk(text, size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

# Read every PDF, page by page
for file in glob.glob("data/*.pdf"):
    print("Reading", file)
    reader = PdfReader(file)
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pieces = chunk(text)
        if not pieces:
            continue
        collection.upsert(
            ids=[f"{file}-{page_num}-{i}" for i in range(len(pieces))],
            documents=pieces,
            embeddings=model.encode(pieces).tolist(),
            metadatas=[{"source": file, "page": page_num + 1}] * len(pieces),
        )

print("Total chunks stored:", collection.count())