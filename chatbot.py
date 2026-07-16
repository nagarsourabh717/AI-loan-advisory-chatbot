import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from vector_store import load_vector_store

load_dotenv()


def ask_question(question):

    try:

        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            return "❌ GOOGLE_API_KEY not found.", []

        if not os.path.exists("vectorstore"):
            return "❌ Please process a PDF first.", []

        vector_db = load_vector_store()

        docs = vector_db.similarity_search(question, k=3)

        context = "\n\n".join(doc.page_content for doc in docs)

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
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

    except Exception as e:
        print("Chatbot Error:", e)
        return f"❌ {str(e)}", []