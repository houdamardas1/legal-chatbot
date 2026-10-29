import os
import requests
import time
from dotenv import load_dotenv

from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult, Generation

load_dotenv()

class GroqLLM(BaseLLM):
    model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    temperature: float = 0.2
    max_tokens: int = 1024
    request_timeout: int = 15  # Timeout en secondes pour la requête API
    max_retries: int = 3       # Nombre max de tentatives en cas d’erreur

    def _generate(self, prompts, stop=None, **kwargs) -> LLMResult:
        generations = []
        for prompt in prompts:
            output = self._call_with_retry(prompt)
            generations.append([Generation(text=output)])
        return LLMResult(generations=generations)

    def _call_with_retry(self, prompt):
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call(prompt)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 503:
                    print(f"⚠️ Erreur 503 Service Unavailable (tentative {attempt}/{self.max_retries})")
                    if attempt == self.max_retries:
                        return "⚠️ Le service Groq est temporairement indisponible. Merci de réessayer plus tard."
                    time.sleep(2 ** attempt)  # délai exponentiel : 2, 4, 8 sec
                else:
                    raise Exception(f"Erreur API Groq : {e}")
            except requests.exceptions.RequestException as e:
                # Autres erreurs réseau
                raise Exception(f"Erreur réseau API Groq : {e}")
        # Si toutes les tentatives échouent (par sécurité)
        return "⚠️ Impossible de contacter le service Groq pour le moment."

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

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=self.request_timeout
        )
        response.raise_for_status()  # Lève une HTTPError si code != 2xx

        result_json = response.json()
        if "choices" not in result_json or len(result_json["choices"]) == 0:
            raise Exception(f"Réponse API Groq inattendue : {result_json}")

        return result_json["choices"][0]["message"]["content"]

    @property
    def _llm_type(self) -> str:
        return "groq-custom"


if __name__ == "__main__":
    llm = GroqLLM()
    réponse = llm.invoke("Explique-moi la clause de confidentialité dans un contrat.")
    print("🧠 Réponse Groq LLM :\n", réponse)

