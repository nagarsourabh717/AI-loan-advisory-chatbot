import os
import streamlit as st

from chatbot import ask_question
from pdf_loader import read_pdf
from text_splitter import split_text
from vector_store import create_vector_store


st.set_page_config(
    page_title="AI Loan Advisory Chatbot",
    page_icon="🏦",
    layout="wide"
)

# ===========================
# Session State
# ===========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = "No PDF Uploaded"

# ===========================
# Sidebar
# ===========================

with st.sidebar:

    st.title("🏦 AI Loan Advisor")

    # ✅ Multiple PDF Upload
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    # ✅ Process Button
    if st.button("Process PDF"):

        if uploaded_files:

            with st.spinner("Processing PDFs..."):

                all_text = ""

                # ✅ Read all PDFs
                for file in uploaded_files:
                    text = read_pdf(file)
                    all_text += text

                # ✅ Check empty text
                if len(all_text.strip()) == 0:
                    st.error("No text found inside PDFs.")
                else:
                    # ✅ Create chunks
                    chunks = split_text(all_text)

                    # ✅ Create vector DB
                    create_vector_store(chunks)

                    # ✅ Update session
                    st.session_state.current_pdf = f"{len(uploaded_files)} PDFs Uploaded"
                    st.session_state.messages = []

                    st.success("All PDFs Processed Successfully ✅")

        else:
            st.warning("Please upload at least one PDF")

    st.divider()

    st.write("### Current PDF")
    st.info(st.session_state.current_pdf)

    st.divider()

    if os.path.exists("vectorstore"):
        st.success("Vector Database Ready ✅")
    else:
        st.warning("Vector Database Not Found")

    st.divider()

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ===========================
# Main Page
# ===========================

st.title("🏦 AI Loan Advisory Chatbot")

st.write(
    "Upload one or more loan policy PDFs and ask any loan-related question."
)

# ===========================
# Chat History
# ===========================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ===========================
# User Input
# ===========================

question = st.chat_input("Ask your question...")

if question:

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):

            try:
                answer, sources = ask_question(question)

                st.markdown(answer)

                if sources:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(sources, start=1):
                            st.write(f"### Source {i}")
                            st.write(source)

            except Exception as e:
                answer = f"❌ Error: {str(e)}"
                st.error(answer)

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })