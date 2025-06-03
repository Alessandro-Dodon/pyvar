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

Usage
-----
1. Install dependencies:
   pip install requests langchain-chroma langchain-community[gpt4all] 

2. Configure your LM Studio endpoint and model at the top of this file.

3. Call:
   - `get_vectorstore(pdf_paths, persist_dir)` to load or build the knowledge base.
   - `build_rag_prompt(...)` to assemble your prompt.
   - `ask_llm(prompt, max_tokens, temperature)` to get the model's response.

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
    alfa = 0.05,
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
    # get context from KB
    hits = vectordb.similarity_search("VaR formulas", k=k)
    raw_context = "\n".join(doc.page_content for doc in hits)
    context_lines = [
        line for line in raw_context.splitlines()
        if not re.match(r'^\s*\d+(?:[\s,]+\d+)*\s*$', line)
    ]
    context = "\n".join(context_lines)

    # metrics list
    metrics = combined.get("VaR & ES Metrics", {})
    met_lines = "\n".join(f"- {name}: {val:.2f} {base}" for name, val in metrics.items())

    # # assemble prompt
    # prompt_sections = [
    #     f'''You are a senior financial analyst. 
    # Comment these Value at Risk metrics:
    # {summary_text}
    # The VaR was computed at a {confidence_level:.0%} confidence level.
    # The portfolio value is {portfolio_value:,.2f} {base}.
    # Remember that the model id accepted if the model model passes both coverage and independence tests.
    
    # Violation Metrics:
    # Number of Violations N, Violation Rate = N/T

    # - Provide a short, non-technical report in no more than four bullet points.  
    # - Do NOT show calculations or p-values, only conclusions.  
    # - Stop as soon as you have four bullets:

    # 1. Best model: name (VaR value, violations, decision)  
    # 2. Worst model name (VaR value, violations, decision)  
    # 3. Overall performance one summary sentence  
    # 4. Portfolio impact one sentence on what this means for the user  
    # - Do NOT repeat yourself. If you finish early, stop immediately.''']
    
    # return "\n".join(prompt_sections)


    prompt_sections = [
        f'''You are a senior financial analyst.  
    Comment on the following Value at Risk (VaR) metrics and their backtest outcomes:

    {summary_text}

    The VaR figures were computed at a {confidence_level:.0%} confidence level.  
    The total portfolio value is {portfolio_value:,.2f} {base}.  
    A model is considered acceptable only if it passes both the coverage and independence tests.

    Violation metrics:
    - Number of violations
    - Violation rate

    Write a short, non-technical report using exactly four bullet points:  

    1. Best model — give the name, VaR value, number of violations, and whether it was accepted or rejected.  
    2. Worst model — same format.  
    3. Overall performance — one clear sentence about the set of models tested.  
    4. Portfolio impact — one sentence on what this means for the user.  

    Do NOT:
    - Include p-values, calculations, or statistics  
    - Repeat any information or generate more than four bullets  
    - Invent data or speculate beyond what is shown  

    Keep it professional, precise, and only use the data above.'''
    ]

    return "\n".join(prompt_sections)
