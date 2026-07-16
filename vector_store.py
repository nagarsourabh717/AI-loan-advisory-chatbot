from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks):
    print("Step 1: Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Step 2: Embedding model loaded")

    print("Step 3: Creating FAISS database...")

    vector_db = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    print("Step 4: Saving vector store...")

    vector_db.save_local("vectorstore")

    print("✅ Vector Store Created Successfully")


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )