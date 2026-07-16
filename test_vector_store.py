from pdf_loader import PDFLoader
from text_splitter import TextSplitter
from vector_store import VectorStore

loader = PDFLoader()
documents = loader.load_pdfs()

print("Documents:", len(documents))

splitter = TextSplitter()
chunks = splitter.split_documents(documents)

print("Chunks:", len(chunks))

vector = VectorStore()

print("Creating vector store...")

vector.create_vector_store(chunks)

print("Done!")