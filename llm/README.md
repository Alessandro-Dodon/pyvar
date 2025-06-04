# pyvar LLM Module

This folder contains optional tools for integrating **local large language models (LLMs)** with the pyvar package. These tools are not included in the main Python package to keep the installation lightweight. However, integrating pyvar with LLM-based applications is straightforward.

---

## 🔧 Main Scripts

The two scripts are:
- `llm_rag.py` — enables a local LLM to answer questions about your results using retrieval-augmented generation (RAG).
- `pdf_reporting.py` — generates automated risk management reports from pyvar output.

These scripts are designed to:
- Provide **automated financial reporting**
- Allow **interactive querying of your results** using a local LLM

### Configuration

To customize the prompt used for LLM interpretation in the report, edit the questions inside the `build_rag_prompt()` function in `llm_rag.py`.

```python
# Inside build_rag_prompt(...)
prompt_sections = [
    f'''You are a financial analyst.

Answer the following questions using only the data below:

...

Which model has the highest VaR?
Which model has the lowest VaR?
Which model had the most violations?
Which models were accepted and why?

Rules:
- Only use the information shown above.
- Do not include calculations
- Be short and clear.
]

```

You can replace or reword the questions to fit your analysis. For best results with local LLMs ask focused, concrete questions, and keep prompts short and unambiguous.

---

## 📚 Knowledge Base

Since this system runs locally, a supporting knowledge base improves LLM output. We include `knowledge_base.pdf` — a short PDF with theoretical clarifications about the risk models used in pyvar. It is used by the RAG pipeline to enhance accuracy and relevance.

---

## 🖥️ LLM Backend: LM Studio

These tools require **[LM Studio](https://lmstudio.ai/)**, a desktop app for running local LLMs. This is kept separate from the main pyvar package to simplify installation for most users.

---

## 🚀 Quick Start

If you want to try out the LLM functionality together with pyvar:

- Check the Python script in the `examples/` folder — it shows a full pipeline from risk modeling to local LLM Q&A/reporting.
- For help setting up LM Studio, see the notebook `tutorial_llm.ipynb`.
