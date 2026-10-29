import os
import json
import re
import time
import shutil
import dateparser
import streamlit as st
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import requests
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

load_dotenv()

# --- Traduction M2M100 (cache avec st.cache_resource) ---
@st.cache_resource(show_spinner=False)
def load_translator():
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    return tokenizer, model

def traduire_texte_m2m100(text, src_lang, tgt_lang):
    tokenizer, model = load_translator()
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
    traduction = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return traduction

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
        return any(valeur_valide(x) for x in val)
    return False

def normalize_date(date_str):
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
    {{"nom_complet": "...", "titre": "...", "date_signature": "..."}} ,
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

# --- Fonction d'initialisation chargement vectorstore avec possibilité upload ---
@st.cache_resource(show_spinner=True)
def charger_vectorstore(uploaded_json):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if uploaded_json is not None:
        # Nettoyer ancien dossier temporaire
        tmp_index_dir = "temp_faiss"
        shutil.rmtree(tmp_index_dir, ignore_errors=True)

        # Chargement JSON
        contenu = json.load(uploaded_json)
        # On suppose que JSON a une liste sous "documents" avec {"text": "...", "id": ...}
        docs = []
        if isinstance(contenu, dict) and "documents" in contenu:
            for d in contenu["documents"]:
                if d.get("text"):
                    docs.append(Document(page_content=d["text"], metadata={"doc_id": str(d.get("id", "inconnu"))}))
        elif isinstance(contenu, list):
            # Cas liste directe de documents
            for d in contenu:
                if isinstance(d, dict) and d.get("text"):
                    docs.append(Document(page_content=d["text"], metadata={"doc_id": str(d.get("id", "inconnu"))}))
        else:
            st.error("Format JSON non reconnu pour l'index vectoriel.")
            return None

        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
        chunks = splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(tmp_index_dir)
        return FAISS.load_local(tmp_index_dir, embeddings, allow_dangerous_deserialization=True)
    else:
        # Chargement index par défaut
        index_dir = "index_cuad.faiss"
        if not os.path.exists(index_dir):
            st.error("Index par défaut introuvable.")
            return None
        return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

# --- Application Streamlit ---
def main():
    st.set_page_config(page_title="Chatbot Juridique Amélioré", layout="wide")

    st.sidebar.title("Options")
    uploaded = st.sidebar.file_uploader("📄 Charger un fichier JSON pour index personnalisé", type=["json"])

    # Charger le vectorstore (cache pour accélérer)
    vectorstore = charger_vectorstore(uploaded)
    if vectorstore is None:
        st.stop()

    llm = GroqLLM()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    if "history" not in st.session_state:
        st.session_state.history = []

    langue = st.sidebar.selectbox("Langue de la réponse", ["Français", "Anglais"])

    if st.sidebar.button("Réinitialiser la conversation"):
        st.session_state.history = []
        st.experimental_rerun()

    st.title("🧠 Chatbot Juridique Amélioré")

    # Affichage historique
    for i, (q, r) in enumerate(st.session_state.history):
        st.markdown(f"**👤 Vous:** {q}")
        st.markdown(f"**🤖 Chatbot:** {r}")
        st.markdown("---")

    question = st.text_input("Posez votre question ici :", key="input_question")

    questions_cibles = ["nom du client", "type de contrat", "date", "signataire", "montant"]

    if question:
        with st.spinner("Traitement en cours..."):
            start = time.time()
            question_lc = question.lower()
            if any(mot in question_lc for mot in questions_cibles):
                réponse = extraire_infos_contrat(question, retriever, llm)
            else:
                réponse = repondre_avec_context(question, retriever, llm)

            # Traduction si nécessaire
            if langue == "Anglais":
                try:
                    # M2M100 code : fr -> en
                    # codes langue M2M100: français "fr", anglais "en"
                    traduction = traduire_texte_m2m100(réponse, src_lang="fr", tgt_lang="en")
                    réponse = traduction
                except Exception as e:
                    réponse += f"\n\n(Note : erreur traduction automatique: {e})"

            # Nettoyage placeholders dans la réponse
            placeholders = ["non précisé", "non mentionné", "confidentiel", "n/a"]
            for ph in placeholders:
                réponse = re.sub(rf"\b{re.escape(ph)}\b", "[Information non disponible]", réponse, flags=re.IGNORECASE)

            st.session_state.history.append((question, réponse))
            elapsed = time.time() - start

        st.markdown(f"**Réponse (traitée en {elapsed:.2f} secondes) :**")
        st.text_area("", réponse, height=300, key=f"réponse_{len(st.session_state.history)}")

if __name__ == "__main__":
    main()
