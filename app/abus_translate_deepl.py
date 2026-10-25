"""
Client minimal de l'API DeepL (offre gratuite « DeepL API Free » : 500 000 caractères par mois).

La clé se colle dans le Studio de doublage (colonne Traduction) ou dans .env (DEEPL_API_KEY=...).
Authentification par en-tête « Authorization: DeepL-Auth-Key ... » (l'ancien paramètre
auth_key n'est plus accepté par DeepL). Une clé gratuite se termine par « :fx ».
"""
import os
import time

import requests

import structlog
logger = structlog.get_logger()


_api_key = ""


def set_deepl_key(key: str):
    global _api_key
    _api_key = (key or "").strip()


def get_deepl_key() -> str:
    return _api_key or os.getenv("DEEPL_API_KEY", "").strip()


def deepl_available() -> bool:
    return bool(get_deepl_key())


class DeepLError(Exception):
    pass


def _endpoint(key: str) -> str:
    base = "https://api-free.deepl.com" if key.endswith(":fx") else "https://api.deepl.com"
    return f"{base}/v2/translate"


def _source_code(code: str) -> str:
    return code.split("-")[0].upper()


def _target_code(code: str) -> str:
    code = code.upper()
    # DeepL exige une variante pour l'anglais et le portugais cibles
    return {"EN": "EN-GB", "PT": "PT-PT", "ZH-CN": "ZH-HANS", "ZH-TW": "ZH-HANT"}.get(code, code.split("-")[0])


class DeepLTranslator:
    """Traduit une liste de phrases ; on envoie par paquets pour garder le contexte et limiter les requêtes."""

    BATCH = 40

    def __init__(self, source_code: str, target_code: str, formality: str = "prefer_less"):
        self.key = get_deepl_key()
        if not self.key:
            raise DeepLError("no DeepL API key")
        self.source = _source_code(source_code)
        self.target = _target_code(target_code)
        self.formality = formality
        self.session = requests.Session()

    def translate_many(self, texts):
        results = []
        for i in range(0, len(texts), self.BATCH):
            results.extend(self._request(texts[i:i + self.BATCH]))
        return results

    def translate(self, text: str) -> str:
        return self._request([text])[0]

    def _request(self, texts, attempts: int = 4):
        payload = {"text": texts, "source_lang": self.source, "target_lang": self.target,
                   "preserve_formatting": True}
        if self.target in ("FR", "DE", "ES", "IT", "NL", "PL", "PT-PT", "PT-BR", "JA", "RU"):
            payload["formality"] = self.formality
        headers = {"Authorization": f"DeepL-Auth-Key {self.key}"}
        last = None
        for attempt in range(attempts):
            try:
                r = self.session.post(_endpoint(self.key), json=payload, headers=headers, timeout=60)
            except requests.RequestException as e:
                last = DeepLError(f"network error: {e}")
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 200:
                return [t["text"] for t in r.json()["translations"]]
            if r.status_code in (429, 500, 502, 503, 504):
                last = DeepLError(f"HTTP {r.status_code}")
                time.sleep(3 * (attempt + 1))
                continue
            if r.status_code == 403:
                raise DeepLError("invalid DeepL API key (403)")
            if r.status_code == 456:
                raise DeepLError("monthly DeepL character quota reached (456)")
            raise DeepLError(f"HTTP {r.status_code}: {r.text[:200]}")
        raise last
