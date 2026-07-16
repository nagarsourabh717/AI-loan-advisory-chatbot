from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


def create_vector_store(chunks):

    print("Loading Gemini Embeddings...")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    print("Creating FAISS Database...")

    vector_db = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    vector_db.save_local("vectorstore")

    print("Vector Store Created Successfully")


def load_vector_store():

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    return FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )