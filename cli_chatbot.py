import os
import json
import shutil
import re
from dotenv import load_dotenv
import dateparser  # pour normalisation des dates

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import requests
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult, Generation

load_dotenv()

# --- Classe LLM personnalisée pour Groq ---
class GroqLLM(BaseLLM):
    model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    temperature: float = 0.2
    max_tokens: int = 1024
    request_timeout: int = 15

    def _generate(self, prompts, stop=None, **kwargs) -> LLMResult:
        generations = []
        for prompt in prompts:
            output = self._call(prompt)
            generations.append([Generation(text=output)])
        return LLMResult(generations=generations)

    def _call(self, prompt, stop=None, **kwargs) -> str:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ Clé API GROQ manquante dans les variables d'environnement.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=self.request_timeout
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Erreur API Groq : {e}")

        result_json = response.json()
        if "choices" not in result_json or len(result_json["choices"]) == 0:
            raise Exception(f"Réponse API Groq inattendue : {result_json}")

        return result_json["choices"][0]["message"]["content"]

    @property
    def _llm_type(self) -> str:
        return "groq-custom"


# --- Fonctions utilitaires ---

def extraire_json_du_texte(texte):
    """Extrait le premier bloc JSON valide dans un texte, même entouré de texte."""
    try:
        blocs = re.findall(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", texte, re.DOTALL)
        for bloc in blocs:
            try:
                return json.loads(bloc)
            except json.JSONDecodeError:
                continue
        return None
    except Exception as e:
        print(f"Erreur extraction JSON : {e}")
        return None

def valeur_valide(val):
    """Retourne True si val est considérée valide, False sinon."""
    if val is None:
        return False
    if isinstance(val, str):
        val_strip = val.strip().lower()
        invalids = ["non précisé", "non specifié", "non spécifié", "n/a", "", 
                    "insert legal name", "insert name", "_____", "date", "name of signatory", "non mentionné", "confidentiel"]
        if any(invalid in val_strip for invalid in invalids):
            return False
        return True
    if isinstance(val, (list, tuple)) and len(val) > 0:
        # valide si au moins un élément valide
        return any(valeur_valide(x) for x in val)
    return False

def normalize_date(date_str):
    """Normalise une date en ISO ou retourne la valeur brute si impossible."""
    if not date_str or not isinstance(date_str, str):
        return date_str
    lower = date_str.lower().strip()
    if lower in ["non précisé", "non mentionné", "confidentiel"]:
        return date_str
    dt = dateparser.parse(date_str, settings={'DATE_ORDER': 'DMY'})
    if dt:
        return dt.strftime("%Y-%m-%d")
    return date_str

def formater_reponse(data, question):
    q = question.lower()

    def format_signataires(signataires_raw):
        nettoyes = []
        for s in signataires_raw:
            if not isinstance(s, dict):
                continue
            nom = s.get("nom_complet") or s.get("nom") or s.get("name") or ""
            titre = s.get("titre") or ""
            date_sig = normalize_date(s.get("date_signature", ""))
            # Filtrage placeholders
            placeholders = ["name of signatory", "name and function of the signatory", "non précisé", "non mentionné", "confidentiel"]
            if not valeur_valide(nom) or any(ph in nom.lower() for ph in placeholders):
                continue
            entry = nom
            if titre and titre.lower() not in placeholders:
                entry += f" ({titre})"
            if date_sig and date_sig.lower() not in placeholders and date_sig != "":
                entry += f", signé le {date_sig}"
            nettoyes.append(entry)
        return nettoyes

    if "nom" in q and "client" in q:
        for key in ["nom_client", "bidder", "discloser"]:
            val = data.get(key)
            if valeur_valide(val):
                return f"Nom du client : {val}"
        return "Nom du client : information non disponible dans le document."

    if "date" in q and "signature" in q:
        signataires = format_signataires(data.get("signataires", []))
        if signataires:
            return "Dates de signature des signataires :\n- " + "\n- ".join(signataires)
        else:
            return "Dates de signature : information non disponible."

    if "date" in q:
        val = data.get("date_contrat")
        if valeur_valide(val):
            return f"Date du contrat : {normalize_date(val)}"
        else:
            return "Date du contrat : information confidentielle, merci de contacter le service juridique."

    if "montant" in q:
        val = data.get("montant")
        if valeur_valide(val):
            if val.strip().lower() == "confidentiel":
                return "Montant : confidentiel, merci de contacter le service financier."
            return f"Montant : {val}"
        else:
            return "Montant : information confidentielle, merci de contacter le service financier."

    if "type" in q and "contrat" in q:
        val = data.get("type_contrat")
        if valeur_valide(val):
            return f"Type de contrat : {val}"
        else:
            return "Type de contrat : information non disponible."

    if "signataire" in q:
        signataires = format_signataires(data.get("signataires", []))
        if signataires:
            return "Signataires :\n- " + "\n- ".join(signataires)
        else:
            return "Signataires : information non disponible."

    # Réponse par défaut (résumé)
    nom_client = None
    for key in ["nom_client", "bidder", "discloser"]:
        val = data.get(key)
        if valeur_valide(val):
            nom_client = val
            break
    type_contrat = data.get("type_contrat")
    if not valeur_valide(type_contrat):
        type_contrat = None
    date_contrat = data.get("date_contrat")
    if not valeur_valide(date_contrat):
        date_contrat = None
    signataires = format_signataires(data.get("signataires", []))
    montant = data.get("montant")
    if not valeur_valide(montant):
        montant = None

    resp = []
    resp.append(f"Nom du client : {nom_client}" if nom_client else "Nom du client : information non disponible.")
    resp.append(f"Type de contrat : {type_contrat}" if type_contrat else "Type de contrat : information non disponible.")
    resp.append(f"Date du contrat : {normalize_date(date_contrat)}" if date_contrat else "Date du contrat : information confidentielle, merci de contacter le service juridique.")
    resp.append("Signataires :\n- " + "\n- ".join(signataires) if signataires else "Signataires : information non disponible.")
    if montant:
        if montant.strip().lower() == "confidentiel":
            resp.append("Montant : confidentiel, merci de contacter le service financier.")
        else:
            resp.append(f"Montant : {montant}")
    else:
        resp.append("Montant : information confidentielle, merci de contacter le service financier.")

    return "\n".join(resp)


def repondre_avec_context(question, retriever, llm):
    docs = retriever.invoke(question)[:3]
    if not docs:
        return "❌ Aucun passage trouvé."

    contexte = "\n\n".join(doc.page_content for doc in docs)[:2000]

    prompt = f"""
Tu es un assistant juridique. Réponds précisément à la question suivante à partir du texte ci-dessous :

Question : "{question}"

Texte :
{contexte}

Réponds de manière claire, concise, et en français.
"""
    réponse = llm.invoke(prompt)
    ids = [doc.metadata.get("doc_id", "inconnu") for doc in docs]
    return f"🤖 Réponse : {réponse.strip()}\n\n📚 Documents utilisés :\n" + "\n".join(f"- ID : {id}" for id in ids)


def extraire_infos_contrat(question, retriever, llm):
    docs = retriever.invoke(question)
    docs = [d for d in docs if "isbn" not in d.page_content.lower()]
    if not docs:
        return "❌ Aucun document pertinent trouvé."

    extrait = "\n\n".join(d.page_content for d in docs)[:2000]

    prompt = f"""
Tu es un assistant juridique expert. Lis le texte ci-dessous et extrait strictement les informations demandées au format JSON en français :

{{
  "nom_client": "...",
  "type_contrat": "...",
  "date_contrat": "...",
  "signataires": [
    {{"nom_complet": "...", "titre": "...", "date_signature": "..."}},
    {{"nom_complet": "...", "titre": "...", "date_signature": "..."}}
  ],
  "montant": "..."
}}

Important :
- Ne jamais utiliser de placeholders ou de valeurs vagues comme "_____", "insert legal name", "date", "name of signatory".
- Si une information est absente ou non claire, indiquer "Non précisé".
- Ignore les noms génériques.
- Rends la réponse uniquement en JSON strict (aucun texte libre).

Texte à analyser :
{extrait}
"""
    réponse = llm.invoke(prompt)
    print("DEBUG - Réponse brute LLM :\n" + réponse + "\n--- Fin réponse brute ---")
    data = extraire_json_du_texte(réponse)
    if not data:
        return repondre_avec_context(question, retriever, llm)

    if not (valeur_valide(data.get("nom_client")) and valeur_valide(data.get("type_contrat"))):
        return repondre_avec_context(question, retriever, llm)

    return formater_reponse(data, question)


def main():
    INDEX_DIR = "index_cuad.faiss"
    JSON_PATH = os.getenv("CUAD_JSON_PATH", "data/train_modified.json")  # chemin du corpus JSON
    CHUNK_SIZE, CHUNK_OVERLAP = 3000, 300
    EMBED_MODEL = "all-MiniLM-L6-v2"

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
        print("🧹 Ancien index FAISS supprimé.")

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)["documents"]

    documents = [
        Document(page_content=doc["text"].strip(), metadata={"doc_id": f"doc_{doc['id']}"})
        for doc in data if doc.get("text", "").strip()
    ]

    if not documents:
        raise ValueError("❌ Aucun document valide trouvé.")

    print("✂️ Découpage des documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)

    print("🔍 Génération des embeddings...")
    all_texts = [doc.page_content for doc in chunks]
    all_metas = [doc.metadata for doc in chunks]

    vectorstore = FAISS.from_texts(all_texts, embeddings, metadatas=all_metas)
    vectorstore.save_local(INDEX_DIR)
    print("✅ Nouvel index FAISS sauvegardé.")

    print("⚙️ Connexion à l'API Groq...")
    llm = GroqLLM()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    questions_cibles = ["nom du client", "type de contrat", "date", "signataire", "montant"]

    print("\n🤖 Chatbot juridique prêt ! Tapez votre question (ou 'exit' pour quitter).\n")
    while True:
        question = input("👤 Vous : ").strip()
        if question.lower() in {"exit", "quit", "bye"}:
            print("👋 Fin de session.")
            break

        if any(mot in question.lower() for mot in questions_cibles):
            print("\n🔎 Extraction d'informations contractuelles...")
            print(extraire_infos_contrat(question, retriever, llm))
        else:
            print("\n🤖 Recherche contextuelle...")
            print(repondre_avec_context(question, retriever, llm))


if __name__ == "__main__":
    main()
