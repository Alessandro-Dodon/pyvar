"""
LLM Integration Module
----------------------

Provides functionality to:
1. Query a local LM Studio model for interpretation of VaR/ES metrics.
2. Build or load a persistent Chroma vectorstore from PDF documents for RAG.
3. Construct a plain-text RAG prompt incorporating portfolio metrics and backtest summary.

This module serves as the core of the LLM-driven interpretation step in the
risk management pipeline. It allows you to embed domain knowledge from PDFs,
formulate clear prompts, and retrieve analysis from your local LLM.

This script is used by pyvar_llm_report.py, which can be found 
in the examples directory.

Authors
-------
Niccolò Lecce, Alessandro Dodon, Marco Gasparetti

Created
-------
May 2025

Contents
--------
- ask_llm: Sends a prompt to the locally running LM Studio LLM and retrieves the response.
- get_vectorstore: Builds or loads a persistent Chroma vector store from PDF documents.
- build_rag_prompt: Constructs a clear, plain-text prompt for RAG analysis.
"""


#----------------------------------------------------------
# Packages
#----------------------------------------------------------
import os
import re
import requests
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings


# ---------------- CONFIGURATION ----------------
LMSTUDIO_ENDPOINT = "http://xxx.xxx.x.xxx:xxxx"  # Your LM Studio server URL
API_PATH            = "/v1/completions"
MODEL_NAME          = "qwen-3-4b-instruct"    # Installed model name
# -----------------------------------------------


#----------------------------------------------------------
# Ask the LLM Function
#----------------------------------------------------------
def ask_llm(prompt,
            max_tokens = 256,
            temperature = 0.2):
    """
    Main
    ----
    Sends a prompt to the locally running LM Studio LLM and retrieves the response.

    Parameters
    ----------
    prompt : str
        The text prompt to send to the LLM.
    max_tokens : int, optional
        Maximum number of tokens to generate (default is 256).
    temperature : float, optional
        Sampling temperature controlling response randomness (default is 0.2).

    Returns
    -------
    str
        The text output returned by the LLM.
    """
    url = LMSTUDIO_ENDPOINT.rstrip("/") + API_PATH
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    resp = requests.post(url, json=payload, timeout=400)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["text"]


#----------------------------------------------------------
# Vectorstore Function
#----------------------------------------------------------
def get_vectorstore(pdf_paths, persist_dir: str = "kb_chroma"):
    """
    Main
    ----
    Builds or loads a persistent Chroma vector store from PDF documents.

    If a persisted directory exists, it loads the existing collection. Otherwise,
    it reads the PDF(s), splits the text into chunks, embeds them, and persists
    a new Chroma collection.

    Parameters
    ----------
    pdf_paths : str or List[str]
        Path (or list of paths) to the PDF document(s) to ingest.
    persist_dir : str, optional
        Directory where the Chroma vector store will be saved or loaded from
        (default is "kb_chroma").

    Returns
    -------
    Chroma
        A Chroma vector store instance, either loaded from disk or newly created.
    """
    embedding = GPT4AllEmbeddings(device="cpu")
    if os.path.isdir(persist_dir):
        vectordb = Chroma(
            persist_directory=persist_dir,
            collection_name="kb_pdf",
            embedding_function=embedding
        )
    else:
        loader = PyPDFLoader(pdf_paths)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=persist_dir,
            collection_name="kb_pdf"
        )
    return vectordb


#----------------------------------------------------------
# RAG Function
#----------------------------------------------------------
def build_rag_prompt(
    *,
    summary_text,
    combined,
    vectordb,
    portfolio_value,
    base,
    confidence_level,
    k = 2):
    """
    Main
    ----
    Constructs a clear, plain-text prompt without markdown.

    Parameters
    ----------
    summary_text : str
        Prebuilt lines like "Asset-Normal VaR has a value of XX EUR, backtest showed..."
    combined : dict
        {"VaR & ES Metrics": {...}, "Backtest Summary": {...}}
    vectordb : Chroma
    portfolio_value : float
    base : str
    k : int

    Returns
    -------
    str
        complete prompt to send to the LLM
    """
    # Get context from KB
    hits = vectordb.similarity_search("VaR formulas", k=k)
    raw_context = "\n".join(doc.page_content for doc in hits)
    context_lines = [
        line for line in raw_context.splitlines()
        if not re.match(r'^\s*\d+(?:[\s,]+\d+)*\s*$', line)
    ]
    context = "\n".join(context_lines)

    # Assemble prompt
    prompt_sections = [
        f'''You are a financial analyst.

    Use the following background knowledge:

    {context}

    Answer the following questions using only the data below:

    {summary_text}

    The VaR values were calculated at a {confidence_level:.0%} confidence level.  
    The total portfolio value is {portfolio_value:,.2f} {base}.

    Explain using non technical language:
    
    - Which model has the highest VaR?
    - Which model has the lowest VaR?
    - Which model had the most violations?
    - Which models were accepted and why? 

    Rules:
    - Only use the information shown above.  
    - Do not include calculations 
    - Be short and clear.'''
    ]
    return "\n".join(prompt_sections)
