# 1) DOWNLOAD LM STUDIO: https://lmstudio.ai/
# 2) GO TO THE DISCOVER SECTION (LEFT MENU) AND INSTALL THE MODEL QWEN-3-4B-INSTRUCT
# 3) GO TO THE DEVELOPER SECTION, LOAD THE MODEL (TOP OF PAGE) AND START THE SERVER (TOP LEFT BUTTON)

import requests
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
import os

# --- CONFIGURATION --------------------------
LMSTUDIO_ENDPOINT = "http://xxx.xxx.x.xxx:xxxx"  # Your LM Studio server URL, you can find this on the right side of the LM Studio Developer tab
API_PATH = "/v1/completions"
MODEL_NAME = "qwen-3-4b-instruct"                # Installed model name
# ---------------------------------------------

# Function to send a prompt to the local LLM and get a response
def ask_llm(prompt: str,
            max_tokens: int = 256,
            temperature: float = 0.2) -> str:
    url = LMSTUDIO_ENDPOINT.rstrip("/") + API_PATH
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["text"]



# ----- RAG (Retrieval-Augmented Generation) -----
# Function to build or load a Chroma vector database from a PDF
def get_vectorstore(pdf_paths, persist_dir="kb_chroma"):
    """
    Ingests one or more PDFs into a persistent Chroma vector database.
    If persist_dir exists, it loads the index; otherwise, it builds it from scratch.
    """
    embedding = GPT4AllEmbeddings(device="cpu")  # CPU-based embedding model

    if os.path.isdir(persist_dir):
        # 1) Load existing Chroma index
        vectordb = Chroma(
            persist_directory=persist_dir,
            collection_name="kb_pdf",
            embedding_function=embedding
        )
    else:
        # 2) Build a new vectorstore from PDF
        loader = PyPDFLoader(pdf_paths)                    # Load PDF document
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(         # Split into overlapping chunks
            chunk_size=500, chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)
        vectordb = Chroma.from_documents(                  # Create and persist vectorstore
            documents=chunks,
            embedding=embedding,
            persist_directory=persist_dir,
            collection_name="kb_pdf"
        )
    return vectordb

# Function to generate a rich prompt for the LLM based on documents + metrics
def build_rag_prompt(metrics: dict,
                     vectordb: Chroma,
                     port_val: float,
                     base: str,
                     k: int = 4) -> str:
    """
    Constructs an LLM prompt that includes:
      • RAG-retrieved documents
      • Risk metrics dictionary
      • Portfolio total value and currency
    """
    # Create a query based on the metrics dictionary
    query = ("Use the following documents to interpret these risk metrics:\n"
             + str(metrics))
    hits = vectordb.similarity_search(query, k=k)          # Retrieve top-k relevant chunks
    context = "\n\n".join(doc.page_content for doc in hits)

    # Final prompt passed to the LLM
    prompt = f"""
    You are a Senior Financial Risk Analyst. Use the following supporting
    documentation (do NOT quote verbatim; just internalize the ideas):

    {context}

    Here are the portfolio risk metrics:
    {metrics}

    The portfolio value is {port_val:,.2f} {base}.

    For each metric give:
    1. A brief definition.
    2. Ideal use cases.
    3. Pros (≥3 bullets).
    4. Cons (≥3 bullets).
    5. Best‑practice hints for risk management.

    Explain in plain language suitable for a non‑expert client.
    """
    return prompt.strip()
