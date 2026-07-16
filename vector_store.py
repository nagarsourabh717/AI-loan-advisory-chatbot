from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"

def create_vector_store(chunks):
    print("Step 1: Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME
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
        model_name=MODEL_NAME
    )

    return FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )