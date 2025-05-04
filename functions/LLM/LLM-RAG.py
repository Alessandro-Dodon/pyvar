# 1) DOWNLOAD LM STUDIO: https://lmstudio.ai/
# 2) GO TO DISCOVER SECTION (ON THE LEFT) AND INSTALL THE MODEL QWEN-3-4B-INSTRUCT (SHOULD WORK WITH EVERY PC)
# 3) GO TO DEVELOPER SECTION (ON THE LEFT) LOAD THE MODEL (ON THE TOP OF THE PAGE) AND START THE SERVER (BUTTON ON THE TOP LEFT CORNER)

import requests
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter    import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
import os


# --- CONFIGURATION --------------------------
LMSTUDIO_ENDPOINT = "http://xxx.xxx.x.xxx:xxxx"    # CAN BE FOUND ON THE RIGHT SIDE OF THE PAGE
API_PATH          = "/v1/completions"
MODEL_NAME        = "qwen-3-4b-instruct"           # NAME OF THE MODEL INSTALLED IN LM STUDIO
# ---------------------------------------------

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



test_prompt = ("Explain the meaning of life")
print("=== Prompt ===\n", test_prompt, "\n")
try:
    answer = ask_llm(test_prompt)
    print("=== Risposta Qwen-3-4B-Instruct ===\n", answer)
except Exception as e:
    print("Errore nella chiamata:", e)




# ----- RAG (Retrieval-Augmented Generation) -----
def get_vectorstore(pdf_paths, persist_dir="kb_chroma"):
    """
    Ingests uno o più PDF in un database Chroma persistente.
    Se persist_dir esiste, lo riapre; altrimenti lo crea.
    """
    # embedding model (CPU)
    embedding = GPT4AllEmbeddings(device="cpu")

    if os.path.isdir(persist_dir):
        # 1) Carica l'indice già esistente
        vectordb = Chroma(
            persist_directory=persist_dir,
            collection_name="kb_pdf",
            embedding_function=embedding
        )
    else:
        # 2) Costruisci da zero
        #    carica PDF
        loader = PyPDFLoader(pdf_paths)
        docs   = loader.load()
        #    split in chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)
        #    costruisci e salva su disco
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=persist_dir,
            collection_name="kb_pdf"
        )
    return vectordb


def build_rag_prompt(metrics: dict,
                     vectordb: Chroma,
                     port_val: float,
                     base: str,
                     k: int = 4) -> str:
    """
    Costruisce il prompt per l’LLM includendo:
      • documenti RAG
      • metriche
      • valore portafoglio totale e valuta
    """
    query = ("Use the following documents to interpret these risk metrics:\n"
             + str(metrics))
    hits  = vectordb.similarity_search(query, k=k)
    context = "\n\n".join(doc.page_content for doc in hits)

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
