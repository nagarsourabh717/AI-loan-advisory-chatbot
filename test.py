import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
print("Key Loaded:", api_key is not None)

emb = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=api_key
)

result = emb.embed_query("Hello World")

print(result[:5])