import pickle
from PyPDF2 import PdfReader

# PDF file name
pdf_path = "Recipe-Book.pdf"   # ivide ninte PDF name kodukku

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"

# Better chunking with overlap
chunk_size = 2000
overlap = 300

chunks = []

for i in range(0, len(text), chunk_size - overlap):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)

# Save chunks
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print(f"✅ Created chunks.pkl with {len(chunks)} chunks")