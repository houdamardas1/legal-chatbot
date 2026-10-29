# ⚖️ Legal Contract Chatbot (RAG + LLM)

An AI assistant that answers questions about legal contracts. It combines **Retrieval-Augmented Generation (RAG)** over a contract corpus with a **Llama 3 model served by Groq** to give grounded answers and to extract key contract information (client, contract type, dates, signatories, amounts).

Built as an end-of-studies internship project (PFE).

## ✨ Features

- **Question answering over contracts**: relevant passages are retrieved from a FAISS vector index and passed to the LLM, which answers only from that context and cites the source document IDs.
- **Structured information extraction**: questions about the client name, contract type, date, signatories or amount trigger a strict-JSON extraction prompt, followed by validation and clean formatting.
- **Robust output handling**: JSON is parsed out of free-form LLM output, placeholder values ("insert name", "_____", …) are filtered, and dates are normalised to ISO format.
- **Bring your own corpus**: upload a JSON file of documents in the web UI to build a custom index on the fly.
- **Bilingual answers**: answers are produced in French and can be translated to English with Meta's M2M100 model.
- **Resilient API client**: a custom LangChain `BaseLLM` wrapper for the Groq API with timeouts and exponential-backoff retries.
- **Two interfaces**: a Streamlit web app and a command-line chatbot.

## 🏗️ Architecture

```
User question
     │
     ▼
Intent routing ──► targeted field? ──► JSON extraction prompt ──► validation & formatting
     │                                         ▲
     ▼                                         │
FAISS retriever (MiniLM embeddings) ──► top-k contract chunks
     │
     ▼
Groq LLM (Llama 3) ──► grounded answer + source IDs ──► optional FR→EN translation (M2M100)
```

## 🧰 Tech stack

| Area | Tools |
|---|---|
| Language | Python 3.10+ |
| LLM | Llama 3 via the Groq API |
| Orchestration | LangChain (custom `BaseLLM`, text splitters, retrievers) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face) |
| Vector store | FAISS |
| Translation | `facebook/m2m100_418M` (Transformers) |
| UI | Streamlit |
| Data | CUAD-style contract corpus (JSON) |

## 📁 Project structure

```
.
├── app.py              # Streamlit web app (RAG, extraction, translation, JSON upload)
├── cli_chatbot.py      # Command-line chatbot that builds the FAISS index and chats in the terminal
├── chatbot_utils.py    # Reusable RAG / extraction helpers
├── GroqLLM.py          # LangChain wrapper for the Groq API with retries
├── data/               # Put your contract corpus here (not versioned)
├── requirements.txt
└── .env.example
```

## 🚀 Getting started

### 1. Clone and install

```bash
git clone https://github.com/houdamardas1/legal-chatbot.git
cd legal-chatbot
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
```

Then edit `.env` and set `GROQ_API_KEY` (a free key is available at [console.groq.com](https://console.groq.com/keys)).

### 3. Add a contract corpus

The app expects a JSON file of documents shaped like this:

```json
{
  "documents": [
    { "id": 1, "text": "This Agreement is entered into by ..." }
  ]
}
```

A good source is the [CUAD dataset](https://www.atticusprojectai.org/cuad) (Contract Understanding Atticus Dataset). Save your file as `data/train_modified.json`, or point `CUAD_JSON_PATH` in `.env` to it.

### 4. Build the index and chat in the terminal

```bash
python cli_chatbot.py
```

This builds the FAISS index in `index_cuad.faiss/` and starts an interactive session.

### 5. Launch the web app

```bash
streamlit run app.py
```

The app loads `index_cuad.faiss/` by default, or you can upload a JSON corpus from the sidebar.

## 💬 Example questions

- *Quel est le nom du client ?*
- *Quel est le type de contrat ?*
- *Qui sont les signataires ?*
- *Quel est le montant du contrat ?*
- *Que dit la clause de confidentialité ?*

## 🔒 Security

No credentials are stored in this repository. The API key is read from the `GROQ_API_KEY` environment variable (loaded from a local `.env` file that is git-ignored).

## 👩‍💻 Author

**Houda Mardas** · [GitHub](https://github.com/houdamardas1)
