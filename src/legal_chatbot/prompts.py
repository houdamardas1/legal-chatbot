"""Prompt templates. Kept in one place so they can be reviewed and versioned like code."""

SYSTEM_PROMPT = (
    "Tu es un assistant juridique spécialisé dans l'analyse de contrats. "
    "Tu réponds uniquement à partir des extraits fournis. "
    "Si la réponse ne figure pas dans les extraits, dis-le clairement au lieu d'inventer."
)

QA_PROMPT = """Réponds à la question en t'appuyant uniquement sur les extraits de contrats ci-dessous.
Cite les identifiants des extraits utilisés entre crochets, par exemple [doc_12].
Si les extraits ne permettent pas de répondre, réponds : "Je ne trouve pas cette information dans les documents."

Question : {question}

Extraits :
{context}

Réponse (en français, claire et concise) :"""

EXTRACTION_PROMPT = """Lis les extraits de contrat ci-dessous et extrais les informations demandées.
Réponds UNIQUEMENT par un objet JSON valide, sans texte avant ni après, avec exactement ces clés :

{{
  "nom_client": string | null,
  "type_contrat": string | null,
  "date_contrat": string | null,
  "signataires": [{{"nom_complet": string, "titre": string | null, "date_signature": string | null}}],
  "montant": string | null
}}

Règles :
- Utilise null quand une information est absente ou ambiguë. N'invente rien.
- N'utilise jamais de valeurs de modèle comme "_____", "[Name]", "insert legal name" ou "name of signatory".
- Si le montant est explicitement confidentiel, écris "confidentiel".

Extraits :
{context}

JSON :"""
