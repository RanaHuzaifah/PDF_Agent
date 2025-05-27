import streamlit as st
from utils import extract_text_from_pdf, chunk_and_embed
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

st.title("Ask Questions About Your PDF")

pdf_file = st.file_uploader("Upload your PDF", type="pdf")

if pdf_file:
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())
    st.success("PDF uploaded successfully!")

    with st.spinner("Extracting text and building knowledge base..."):
        text = extract_text_from_pdf("temp.pdf")
        vectorstore = chunk_and_embed(text)

    question = st.text_input("Ask a question about the PDF:")
    if question:
        docs = vectorstore.similarity_search(question)
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        answer = chain.run(input_documents=docs, question=question)
        st.write("**Answer:**", answer)