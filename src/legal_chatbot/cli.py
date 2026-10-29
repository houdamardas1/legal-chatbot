"""Command-line interface: ``legal-chatbot build-index`` and ``legal-chatbot chat``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Settings
from .llm import GroqAPIError, GroqLLM
from .prompts import SYSTEM_PROMPT


def _build_index(args: argparse.Namespace, settings: Settings) -> int:
    from .rag import build_vectorstore, load_corpus, save_vectorstore

    corpus = Path(args.corpus or settings.corpus_path)
    index_dir = Path(args.index_dir or settings.index_dir)
    documents = load_corpus(corpus)
    print(f"Loaded {len(documents)} documents from {corpus}")
    vectorstore = build_vectorstore(documents, settings)
    save_vectorstore(vectorstore, index_dir)
    print(f"Index saved to {index_dir} ({vectorstore.index.ntotal} chunks)")
    return 0


def _chat(args: argparse.Namespace, settings: Settings) -> int:
    from .rag import ContractAssistant, load_vectorstore, make_retriever

    index_dir = Path(args.index_dir or settings.index_dir)
    vectorstore = load_vectorstore(index_dir, settings)
    llm = GroqLLM.from_settings(settings, system_prompt=SYSTEM_PROMPT)
    assistant = ContractAssistant(
        make_retriever(vectorstore, settings.top_k), llm, settings.max_context_chars
    )

    print("Chatbot juridique prêt. Tapez votre question ('exit' pour quitter).\n")
    while True:
        try:
            question = input("Vous : ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit", "bye"}:
            break
        if not question:
            continue
        try:
            answer = assistant.ask(question)
        except GroqAPIError as exc:
            print(f"Erreur : {exc}\n")
            continue
        print(f"\nAssistant ({answer.mode}, {answer.latency_s:.1f}s) :\n{answer.text}")
        if answer.sources:
            print("Sources : " + ", ".join(s.doc_id for s in answer.sources))
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="legal-chatbot", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-index", help="embed a JSON corpus into a FAISS index")
    build.add_argument("--corpus", help="path to the JSON corpus (default: $CUAD_JSON_PATH)")
    build.add_argument("--index-dir", help="output directory (default: $INDEX_DIR)")
    build.set_defaults(func=_build_index)

    chat = sub.add_parser("chat", help="ask questions in the terminal")
    chat.add_argument("--index-dir", help="FAISS index directory (default: $INDEX_DIR)")
    chat.set_defaults(func=_chat)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args, Settings.from_env())
    except (FileNotFoundError, ValueError, GroqAPIError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
