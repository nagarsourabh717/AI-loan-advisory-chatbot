import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


def ask_question(question):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "❌ GROQ_API_KEY not found.", []

    if not os.path.exists("vectorstore"):
        return "❌ Please process a PDF first.", []

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)

    vector_db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = vector_db.similarity_search(question, k=3)

    context = "\n\n".join(doc.page_content for doc in docs)

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )

    prompt = f"""
You are an AI Loan Advisory Assistant.

Answer ONLY from the loan policy below.

If the answer is not found, reply:

"I couldn't find this information in the provided loan policy."

Loan Policy:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    sources = [doc.page_content for doc in docs]

    return response.content, sources