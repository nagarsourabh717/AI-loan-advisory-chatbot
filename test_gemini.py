import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

print("API Key:", api_key)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

response = llm.invoke("Hello")
print(response.content)