import os
from dotenv import load_dotenv
# Importing necessary libraries and modules
import streamlit as st  # Streamlit library for creating the user interface
from langchain.embeddings.openai import OpenAIEmbeddings  # Module for embeddings using OpenAI language models
import tempfile  # Module for handling temporary files
import time  # Module for time-related operations
from langchain import OpenAI  # Classes from Langchain library
from langchain.text_splitter import CharacterTextSplitter  # Class for splitting text into smaller chunks
from langchain.document_loaders import PyPDFLoader  # Class for loading and splitting PDF documents
from langchain.chains.summarize import load_summarize_chain  # Function for loading summarization chain
from langchain.docstore.document import Document  # Class representing a document

# Initializing OpenAI language model

dotenv_path = "openai.env"
load_dotenv(dotenv_path)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError(f"Unable to retrieve OPENAI_API_KEY from {dotenv_path}")
else:
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
    


# Initializing text splitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)

# Streamlit application title and author
st.title("📄PDF Summarizer")

# File uploader to upload PDF files
pdf_file = st.file_uploader("Choose a PDF file", type="pdf")

# If a PDF file is uploaded
if pdf_file is not None:
    # Temporary file creation to store the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(pdf_file.read())
        pdf_path = tmp_file.name
        # Loading and splitting PDF pages
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()

        # User input for page selection
        page_selection = st.radio("Page selection", ["Single page", "Page range", "Overall Summary"])

        # If single page selection is chosen
        if page_selection == "Single page":
            page_number = st.number_input("Enter page number", min_value=1, max_value=len(pages), value=1, step=1)
            view = pages[page_number - 1]
            texts = text_splitter.split_text(view.page_content)
            docs = [Document(page_content=t) for t in texts]
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            summaries = chain.run(docs)

            st.subheader("Summary")
            st.write(summaries)

        # If page range selection is chosen
        elif page_selection == "Page range":
            start_page = st.number_input("Enter start page", min_value=1, max_value=len(pages), value=1, step=1)
            end_page = st.number_input("Enter end page", min_value=start_page, max_value=len(pages), value=start_page,
                                       step=1)

            texts = []
            for page_number in range(start_page, end_page + 1):
                view = pages[page_number - 1]
                page_texts = text_splitter.split_text(view.page_content)
                texts.extend(page_texts)
            docs = [Document(page_content=t) for t in texts]
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            summaries = chain.run(docs)
            st.subheader("Summary")
            st.write(summaries)

        # If overall summary selection is chosen
        elif page_selection == "Overall Summary":
            combined_content = ''.join([p.page_content for p in pages])  # Concatenating entire page content
            texts = text_splitter.split_text(combined_content)
            docs = [Document(page_content=t) for t in texts]
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            summaries = chain.run(docs)
            st.subheader("Summary")
            st.write(summaries)

else:
    time.sleep(30)
    st.warning("No PDF file uploaded", "🚨")  # Warning if no PDF file is uploaded
