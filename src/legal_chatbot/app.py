"""Streamlit chat interface. Run with: ``streamlit run app.py``."""

from __future__ import annotations

import hashlib

import streamlit as st

from legal_chatbot.config import Settings
from legal_chatbot.llm import GroqAPIError, GroqLLM
from legal_chatbot.prompts import SYSTEM_PROMPT
from legal_chatbot.rag import (
    Answer,
    ContractAssistant,
    CorpusFormatError,
    build_vectorstore,
    get_embeddings,
    load_vectorstore,
    make_retriever,
    parse_corpus,
)
from legal_chatbot.translation import SUPPORTED_LANGUAGES, translate

EXAMPLE_QUESTIONS = (
    "Quel est le nom du client ?",
    "Qui sont les signataires et quand ont-ils signé ?",
    "Que prévoit la clause de confidentialité ?",
    "Quelle est la durée du contrat et comment le résilier ?",
)


@st.cache_resource(show_spinner="Chargement du modèle d'embeddings…")
def _embeddings(model_name: str):
    return get_embeddings(Settings.from_env())


@st.cache_resource(show_spinner="Chargement de l'index…")
def _default_index(index_dir: str, model_name: str):
    settings = Settings.from_env()
    return load_vectorstore(settings.index_dir, settings, _embeddings(model_name))


@st.cache_resource(show_spinner="Indexation du corpus importé…")
def _uploaded_index(content_hash: str, _raw: bytes, model_name: str):
    import json

    settings = Settings.from_env()
    documents = parse_corpus(json.loads(_raw))
    return build_vectorstore(documents, settings, _embeddings(model_name))


def _render_answer(answer: Answer, show_debug: bool) -> None:
    st.markdown(answer.text)
    mode = "Extraction structurée" if answer.mode == "extraction" else "Question ouverte"
    st.caption(f"{mode} · {answer.latency_s:.1f} s")
    if answer.sources:
        with st.expander(f"Sources ({len(answer.sources)})"):
            for source in answer.sources:
                st.markdown(f"**{source.doc_id}** : {source.snippet}…")
    if show_debug and answer.raw_llm_output:
        with st.expander("Sortie brute du LLM"):
            st.code(answer.raw_llm_output, language="json")


def main() -> None:
    st.set_page_config(page_title="Legal Contract Chatbot", layout="wide")
    settings = Settings.from_env()

    with st.sidebar:
        st.header("Paramètres")
        uploaded = st.file_uploader("Corpus JSON personnalisé", type=["json"])
        language = st.selectbox("Langue des réponses", list(SUPPORTED_LANGUAGES))
        show_debug = st.toggle("Mode debug", value=False)
        if st.button("Nouvelle conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.caption(f"LLM : `{settings.groq_model}`")
        st.caption(f"Embeddings : `{settings.embedding_model}`")

    st.title("Chatbot juridique")
    st.write(
        "Posez une question sur vos contrats : les réponses sont fondées sur les documents indexés."
    )

    if not settings.groq_api_key:
        st.error(
            "`GROQ_API_KEY` est absente. Copiez `.env.example` vers `.env` et ajoutez votre clé."
        )
        st.stop()

    try:
        if uploaded is not None:
            raw = uploaded.getvalue()
            vectorstore = _uploaded_index(
                hashlib.sha256(raw).hexdigest(), raw, settings.embedding_model
            )
        elif not settings.index_dir.exists():
            # Checked up front so we don't load the embedding model just to find no index.
            raise FileNotFoundError(settings.index_dir)
        else:
            vectorstore = _default_index(str(settings.index_dir), settings.embedding_model)
    except FileNotFoundError:
        st.warning(
            "Aucun index trouvé. Lancez `legal-chatbot build-index` ou importez un corpus JSON "
            "dans la barre latérale."
        )
        st.stop()
    except (CorpusFormatError, ValueError) as exc:
        st.error(f"Corpus invalide : {exc}")
        st.stop()

    assistant = ContractAssistant(
        make_retriever(vectorstore, settings.top_k),
        GroqLLM.from_settings(settings, system_prompt=SYSTEM_PROMPT),
        settings.max_context_chars,
    )

    messages: list[dict] = st.session_state.setdefault("messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_answer(message["answer"], show_debug)
            else:
                st.markdown(message["content"])

    if not messages:
        cols = st.columns(len(EXAMPLE_QUESTIONS))
        for col, example in zip(cols, EXAMPLE_QUESTIONS, strict=True):
            if col.button(example, use_container_width=True):
                st.session_state.pending_question = example
                st.rerun()

    question = st.chat_input("Votre question…") or st.session_state.pop("pending_question", None)
    if not question:
        return

    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyse des contrats…"):
            try:
                answer = assistant.ask(question)
            except GroqAPIError as exc:
                st.error(f"Le service LLM est indisponible : {exc}")
                return
            target = SUPPORTED_LANGUAGES[language]
            if target != "fr":
                try:
                    answer.text = translate(answer.text, "fr", target, settings.translation_model)
                except Exception as exc:  # translation is best effort
                    answer.text += f"\n\n_(Traduction indisponible : {exc})_"
        _render_answer(answer, show_debug)
    messages.append({"role": "assistant", "answer": answer})


if __name__ == "__main__":
    main()
