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
    combined: dict,
    vectordb: Chroma,
    portfolio_value: float,
    base: str,
    k: int = 2
) -> str:
    """
    Constructs a clear, plain-text prompt without markdown.

    Retrieves context from PDF, cleans numeric noise, formats metrics and backtest,
    and instructs the LLM using numbered, plain headings.
    """
    # 1) Retrieve and clean context chunks
    hits = vectordb.similarity_search("VaR formulas", k=k)
    raw_context = "\n".join(doc.page_content for doc in hits)
    context_lines = [line for line in raw_context.splitlines()
                     if not re.match(r'^\s*\d+(?:[\s,]+\d+)*\s*$', line)]
    context = "\n".join(context_lines)

    # 2) Extract metrics and backtest entries
    metrics = combined.get("VaR & ES Metrics", {})
    backtest = combined.get("Backtest Summary", {})

    # 3) Format metric and backtest lines
    met_lines = "\n".join(f"- {name}: {val:.2f} {base}" for name, val in metrics.items())
    bt_lines = "\n".join(
        f"- {model}: violations={d['Violations']}, rate={d['Violation Rate']:.3f}, "
        f"kupiec_p={d['Kupiec p-value']:.3f}, chr_p={d['Christoffersen p-value']:.3f}, "
        f"joint_p={d['Joint p-value']:.3f}"
        for model, d in backtest.items()
    )

    # 4) Construct plain-text prompt with explicit handling of non-backtested metrics
    prompt_sections = [
        "Do not use markdown. Use plain numbered sections.",
        # Context Summary
        f"1. Context Summary:\n"
        f"Provide 2-3 sentences relating the theoretical formulas to the portfolio metrics.\n"
        f"Context formulas:\n{context}\n",
        # Metric Definitions
        "2. Metric Definitions:\n"
        "For each metric, follow this template exactly:\n"
        "  Metric Name\n"
        "  a) Definition (one sentence)\n"
        "  b) Ideal Use Cases (2-3 bullets)\n"
        "  c) Pros (3 bullets)\n"
        "  d) Cons (3 bullets)\n"
        "  e) Interpretation Guidance:\n"
        "     - If backtest data available: comment violations and p-values.\n"
        "     - If no backtest data: state 'Not backtested'.\n"
        f"Metrics list:\n{met_lines}\n",
        # Backtest Summary
        f"3. Backtest Summary:\n{bt_lines if bt_lines else '- No backtests performed'}\n",
        # Portfolio Summary
        "4. Portfolio Summary:\nOne sentence on how portfolio value scales VaR and ES.\n",
        # Best-Practice Recommendations
        "5. Best-Practice Recommendations:\nFive actionable, non-technical tips.\n",
        # Executive Summary
        "6. Executive Summary:\nProvide 2-3 sentences at C-suite level."
    ]

    return "\n".join(prompt_sections)
