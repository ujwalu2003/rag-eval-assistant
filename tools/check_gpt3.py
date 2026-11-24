from pypdf import PdfReader

r = PdfReader("data/gpt3.pdf")
print("Encrypted:", r.is_encrypted)
print("Pages:", len(r.pages))
for i in [0, 1, 10]:
    if i < len(r.pages):
        t = r.pages[i].extract_text() or ""
        print(f"Page {i+1}: {len(t)} characters")