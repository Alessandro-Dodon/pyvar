# PyVaR LLM Module

This folder contains optional tools for integrating **local large language models (LLMs)** with the PyVaR package.

---

## 🔧 Main Scripts

- `llm_rag.py` — enables a local LLM to answer questions about your results using retrieval-augmented generation (RAG).
- `pdf_reporting.py` — generates automated risk management reports from PyVaR output.

These scripts are designed to:
- Provide **automated financial reporting**
- Allow **interactive querying of your results** using a local LLM

---

## 📚 Knowledge Base

Since this system runs locally, a supporting knowledge base improves LLM output.  
We include `knowledge_base.pdf` — a short PDF with theoretical clarifications about the risk models used in PyVaR.  
It is used by the RAG pipeline to enhance accuracy and relevance.

---

## 🖥️ LLM Backend: LM Studio

These tools require **[LM Studio](https://lmstudio.ai/)**, a desktop app for running local LLMs.  
This is kept **separate from the main PyVaR package** to simplify installation for most users.

---

## 🚀 Quick Start

If you want to try out the LLM functionality together with PyVaR:

- Check the Python script in the `examples/` folder — it shows a full pipeline from risk modeling to local LLM Q&A/reporting.
- For help setting up LM Studio, see the notebook `tutorial_llm.ipynb`.
