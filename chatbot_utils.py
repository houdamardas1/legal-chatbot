import os
import json
import re
import dateparser
from dotenv import load_dotenv

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

try:
    import streamlit as st
except ImportError:
    st = None  # Usage hors Streamlit possible

from GroqLLM import GroqLLM  # Ta classe LLM personnalisée

load_dotenv()

def extraire_json_du_texte(texte: str):
    """
    Tente d'extraire un JSON strict depuis une chaîne de texte.
    Supporte JSON imbriqué et nettoie tout contenu avant/après.
    """
    try:
        # Cherche le plus grand bloc JSON possible (premier JSON bien formé)
        blocs = re.findall(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", texte, re.DOTALL)
        for bloc in blocs:
            try:
                data = json.loads(bloc)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                continue
        return None
    except Exception as e:
        print(f"Erreur extraction JSON : {e}")
        return None

def valeur_valide(val):
    """
    Vérifie si une valeur extraite est considérée valide et utile.
    """
    if val is None:
        return False
    if isinstance(val, str):
        val_strip = val.strip().lower()
        invalids = [
            "non précisé", "non specifié", "non spécifié", "n/a", "", 
            "insert legal name", "insert name", "_____", "date", 
            "name of signatory", "non mentionné", "confidentiel"
        ]
        return not any(invalid in val_strip for invalid in invalids)
    if isinstance(val, (list, tuple)) and len(val) > 0:
        return any(valeur_valide(x) for x in val)
    return False

def normalize_date(date_str):
    """
    Convertit une date texte en format ISO YYYY-MM-DD si possible.
    Sinon retourne la chaîne d'origine.
    """
    if not date_str or not isinstance(date_str, str):
        return date_str
    lower = date_str.lower().strip()
    if lower in ["non précisé", "non mentionné", "confidentiel"]:
        return date_str
    dt = dateparser.parse(date_str, settings={'DATE_ORDER': 'DMY'})
    return dt.strftime("%Y-%m-%d") if dt else date_str

def formater_reponse(data, question):
    """
    Formate la réponse à afficher selon la question posée.
    """
    q = question.lower()

    def format_signataires(signataires_raw):
        nettoyes = []
        for s in signataires_raw:
            if not isinstance(s, dict):
                continue
            nom = s.get("nom_complet") or s.get("nom") or s.get("name") or ""
            titre = s.get("titre") or ""
            date_sig = normalize_date(s.get("date_signature", ""))
            if not valeur_valide(nom):
                continue
            entry = nom
            if titre and valeur_valide(titre):
                entry += f" ({titre})"
            if valeur_valide(date_sig):
                entry += f", signé le {date_sig}"
            nettoyes.append(entry)
        return nettoyes

    if "nom" in q and "client" in q:
        for key in ["nom_client", "bidder", "discloser"]:
            val = data.get(key)
            if valeur_valide(val):
                return f"Nom du client : {val}"
        return "Nom du client : information non disponible."

    if "date" in q:
        val = data.get("date_contrat")
        return f"Date du contrat : {normalize_date(val)}" if valeur_valide(val) else "Date du contrat : information non disponible."

    if "montant" in q:
        val = data.get("montant")
        if valeur_valide(val):
            if val.strip().lower() == "confidentiel":
                return "Montant : confidentiel."
            return f"Montant : {val}"
        return "Montant : information non disponible."

    if "type" in q and "contrat" in q:
        val = data.get("type_contrat")
        return f"Type de contrat : {val}" if valeur_valide(val) else "Type de contrat : information non disponible."

    if "signataire" in q:
        signataires = format_signataires(data.get("signataires", []))
        return "Signataires :\n- " + "\n- ".join(signataires) if signataires else "Signataires : information non disponible."

    return "Résumé non disponible. Veuillez reformuler votre question."

def repondre_avec_context(question, retriever, llm):
    """
    Réponse classique en s’appuyant sur les passages extraits du corpus.
    """
    docs = retriever.invoke(question)[:3]
    if not docs:
        return "❌ Aucun passage trouvé."

    contexte = "\n\n".join(doc.page_content for doc in docs)[:2000]

    prompt = f"""
Tu es un assistant juridique. Réponds à la question suivante en te basant uniquement sur le texte fourni :

Question : "{question}"

Texte :
{contexte}

Réponds de manière claire, concise, et en français.
"""
    réponse = llm.invoke(prompt)
    ids = [doc.metadata.get("doc_id", "inconnu") for doc in docs]
    return f"🤖 Réponse : {réponse.strip()}\n\n📚 Documents utilisés :\n" + "\n".join(f"- ID : {id}" for id in ids)

def extraire_infos_contrat(question, retriever, llm):
    """
    Extraction stricte des informations contractuelles via prompt JSON.
    """
    docs = retriever.invoke(question)
    if not docs:
        return "❌ Aucun document pertinent trouvé."

    extrait = "\n\n".join(d.page_content for d in docs)[:2000]

    prompt = f"""
Tu es un assistant juridique expert. Lis le texte ci-dessous et réponds **uniquement** par un JSON strict contenant :

{{
  "nom_client": "...",
  "type_contrat": "...",
  "date_contrat": "...",
  "signataires": [
    {{"nom_complet": "...", "titre": "...", "date_signature": "..."}}
  ],
  "montant": "..."
}}

- Si une information est absente, écris "Non précisé".
- Ne fournis aucun texte hors du JSON.
- Exemple :
{{
  "nom_client": "ACME Corp",
  "type_contrat": "Vente de services",
  "date_contrat": "2023-01-01",
  "signataires": [
    {{"nom_complet": "Jean Dupont", "titre": "CEO", "date_signature": "2023-01-05"}}
  ],
  "montant": "15000 EUR"
}}

Texte à analyser :
{extrait}
"""
    réponse = llm.invoke(prompt)

    # Affichage debug dans Streamlit
    if st:
        st.text_area("🧪 Réponse brute du LLM (debug)", réponse, height=250)

    data = extraire_json_du_texte(réponse)
    if not data or not isinstance(data, dict):
        # fallback classique
        return repondre_avec_context(question, retriever, llm)

    return formater_reponse(data, question)

def init_system(json_data):
    """
    Initialise la base vectorielle et le retriever LangChain depuis le JSON.
    """
    CHUNK_SIZE = 3000
    CHUNK_OVERLAP = 300
    MODEL_NAME = "all-MiniLM-L6-v2"

    documents = [
        Document(page_content=doc["text"].strip(), metadata={"doc_id": f"doc_{doc['id']}"} )
        for doc in json_data if doc.get("text", "").strip()
    ]

    if not documents:
        raise ValueError("❌ Aucun document valide détecté dans le fichier JSON.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 10})


