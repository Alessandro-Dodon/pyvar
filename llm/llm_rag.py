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


def ask_llm(prompt: str,
            max_tokens: int = 256,
            temperature: float = 0.2) -> str:
    """
    Sends a prompt to the locally running LM Studio LLM and retrieves the response.
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


def get_vectorstore(pdf_paths, persist_dir: str = "kb_chroma") -> Chroma:
    """
    Builds or loads a persistent Chroma vector DB from PDF documents.
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


def build_rag_prompt(
    *,
    summary_text: str,
    combined: dict,
    vectordb: Chroma,
    portfolio_value: float,
    base: str,
    confidence_level,
    k: int = 2
) -> str:
    """
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
        Il prompt completo da inviare all’LLM.
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

    # assemble prompt
    prompt_sections = [
        f'''You are a senior financial analyst. 
Comment these Value at Risk metrics:
{summary_text}
The VaR was computed at a {confidence_level:.0%} confidence level.
The portfolio value is {portfolio_value:,.2f} {base}.
Remember that: Interpretation of p-Values
• Kupiec p-value: if > alfa, exception rate is consistent with model.
• Christoffersen p-value: if > alfa, exceptions are serially independent.
• Joint p-value: if > alfa, model passes both coverage and independence tests.
Violation Metrics
Number of Violations N, Violation Rate = N/T

— Provide a **short**, non-technical report (avoid jargon, no markdown).
— Don't show reasoning or calculations, only the conclusions.
— Limit to **4 bullets**:
   1. Best model: name (value, violations, p-value)
   2. Worst model: name (value, violations, p-value)
   3. Overall performance summary (1 sentence)
   4. Portfolio impact (1 sentence)
— Do NOT repeat yourself. If you finish early, stop immediately.''']
    
    return "\n".join(prompt_sections)


'''Do not use Markdown. Use plain numbered sections. Be client-facing, concise, and data-driven. Use “risk management” at least once.

0. Metrics & Backtest Summary  
Here are the actual numbers—integrate them verbatim into your text:  
{summary_text}

1. Context Summary  
In 2–3 sentences, explain why we compute these VaR metrics and how they fit into our risk management framework. Reference the key formula (“VaR = z·σ”) in one phrase.

2. Metric Definitions  
Below is the list of all metrics with their values (including ES), as of today. For each, do exactly:
  a) One-sentence definition  
  b) 2–3 bullets: when to use it (client-friendly)  
  c) 2–3 bullets: real-world pros  
  d) 2–3 bullets: real-world cons  
  e) One sentence interpreting *its* backtest results or “Not backtested” (use the numbers in section 0)

Metrics list:  
{metrics_list}

3. Portfolio Summary  
One crisp sentence on how portfolio size scales VaR and ES.

4. Best-Practice Recommendations  
Five actionable, non-technical tips, each starting with “To improve your risk management, you should…”

5. Executive Summary  
2–3 sentences at C-suite level: highlight the single biggest takeaway from the actual data and the next step.

6. References  
List any sources you used, or say “No external references used.”'''