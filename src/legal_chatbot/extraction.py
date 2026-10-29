"""Structured contract-field extraction: intent detection, JSON parsing, validation, formatting.

Everything in this module is pure Python so it can be unit-tested without an LLM or a vector store.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import dateparser


class ContractField(str, Enum):
    CLIENT = "client"
    CONTRACT_TYPE = "contract_type"
    CONTRACT_DATE = "contract_date"
    SIGNATURE_DATES = "signature_dates"
    SIGNATORIES = "signatories"
    AMOUNT = "amount"


ALL_FIELDS: tuple[ContractField, ...] = tuple(ContractField)

# Keyword patterns (matched on accent-free lowercase text, with word boundaries).
_INTENT_PATTERNS: dict[ContractField, tuple[str, ...]] = {
    ContractField.CLIENT: (r"\bclient\b", r"\bcustomer\b", r"\bnom du client\b"),
    ContractField.CONTRACT_TYPE: (
        r"\btype de contrat\b",
        r"\btype du contrat\b",
        r"\bnature du contrat\b",
        r"\bcontract type\b",
        r"\btype of (the )?contract\b",
    ),
    ContractField.SIGNATORIES: (
        r"\bsignataires?\b",
        r"\bsignator(y|ies)\b",
        r"\bsigne par\b",
        r"\bsigned by\b",
        r"\bqui a signe\b",
        r"\bwho signed\b",
    ),
    ContractField.AMOUNT: (
        r"\bmontant\b",
        r"\bprix\b",
        r"\bamount\b",
        r"\bprice\b",
        r"\bcombien\b",
    ),
}
_DATE_PATTERN = re.compile(r"\b(date|dated|quand|when)\b")
_SIGNATURE_PATTERN = re.compile(r"\b(signature|signe|signed)\b")
_SUMMARY_PATTERN = re.compile(
    r"\b(resume|summary|summarize|informations? cles?|key (information|terms|facts))\b"
)

# Values an LLM (or a contract template) produces when the information is missing.
_PLACEHOLDER_VALUES = frozenset(
    {
        "",
        "n/a",
        "na",
        "none",
        "null",
        "unknown",
        "inconnu",
        "date",
        "name",
        "non precise",
        "non specifie",
        "non mentionne",
        "non disponible",
        "not specified",
        "not mentioned",
        "insert legal name",
        "insert name",
        "name of signatory",
        "name and function of the signatory",
        "...",
    }
)
_PLACEHOLDER_PATTERNS = (
    re.compile(r"^_+$"),
    re.compile(r"^\.+$"),
    re.compile(r"^\[.*\]$"),
    re.compile(r"^<.*>$"),
    re.compile(r"^insert\b"),
)
CONFIDENTIAL = "confidentiel"


def normalize_text(text: str) -> str:
    """Lowercase, strip accents and collapse whitespace (``"Signé  par"`` -> ``"signe par"``)."""
    decomposed = unicodedata.normalize("NFKD", text)
    no_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", no_accents).strip().lower()


def detect_fields(question: str) -> list[ContractField]:
    """Return the contract fields a question asks about, or ``[]`` for open questions."""
    q = normalize_text(question)
    if _SUMMARY_PATTERN.search(q):
        return list(ALL_FIELDS)

    fields = [
        f for f, patterns in _INTENT_PATTERNS.items() if any(re.search(p, q) for p in patterns)
    ]

    if _DATE_PATTERN.search(q):
        if _SIGNATURE_PATTERN.search(q):
            fields.append(ContractField.SIGNATURE_DATES)
            if ContractField.SIGNATORIES in fields:
                fields.remove(ContractField.SIGNATORIES)
        else:
            fields.append(ContractField.CONTRACT_DATE)

    order = {f: i for i, f in enumerate(ALL_FIELDS)}
    return sorted(set(fields), key=order.__getitem__)


def is_meaningful(value: Any) -> bool:
    """True when ``value`` carries real information (not empty, not a template placeholder)."""
    if value is None:
        return False
    if isinstance(value, str):
        v = normalize_text(value).strip(" .:;,")
        if v in _PLACEHOLDER_VALUES:
            return False
        return not any(p.search(v) for p in _PLACEHOLDER_PATTERNS)
    if isinstance(value, (list, tuple)):
        return any(is_meaningful(v) for v in value)
    if isinstance(value, dict):
        return any(is_meaningful(v) for v in value.values())
    return True


def normalize_date(value: str | None) -> str | None:
    """Convert a free-text date to ISO ``YYYY-MM-DD``; return the input unchanged if unparseable."""
    if not value or not isinstance(value, str) or not is_meaningful(value):
        return value
    parsed = dateparser.parse(value, settings={"DATE_ORDER": "DMY", "PREFER_DAY_OF_MONTH": "first"})
    return parsed.strftime("%Y-%m-%d") if parsed else value


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first JSON object found in an LLM response (handles code fences and prose)."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            obj, _ = decoder.raw_decode(cleaned, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


@dataclass
class Signatory:
    name: str
    title: str | None = None
    signature_date: str | None = None

    def display(self, with_date: bool = True) -> str:
        text = self.name
        if self.title:
            text += f" ({self.title})"
        if with_date and self.signature_date:
            text += f", signé le {self.signature_date}"
        return text


@dataclass
class ContractInfo:
    client: str | None = None
    contract_type: str | None = None
    contract_date: str | None = None
    amount: str | None = None
    signatories: list[Signatory] = field(default_factory=list)

    @classmethod
    def from_llm_json(cls, data: dict[str, Any]) -> ContractInfo:
        """Build a validated ``ContractInfo`` from the JSON returned by the extraction prompt."""

        def clean(value: Any) -> str | None:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                value = str(value)
            return value.strip() if isinstance(value, str) and is_meaningful(value) else None

        client = next(
            (
                clean(data.get(k))
                for k in ("nom_client", "client", "bidder", "discloser")
                if clean(data.get(k))
            ),
            None,
        )
        signatories: list[Signatory] = []
        raw_signatories = data.get("signataires") or data.get("signatories") or []
        if isinstance(raw_signatories, list):
            for raw in raw_signatories:
                if not isinstance(raw, dict):
                    continue
                name = clean(raw.get("nom_complet") or raw.get("nom") or raw.get("name"))
                if not name:
                    continue
                signatories.append(
                    Signatory(
                        name=name,
                        title=clean(raw.get("titre") or raw.get("title")),
                        signature_date=normalize_date(clean(raw.get("date_signature"))),
                    )
                )

        return cls(
            client=client,
            contract_type=clean(data.get("type_contrat") or data.get("contract_type")),
            contract_date=normalize_date(
                clean(data.get("date_contrat") or data.get("contract_date"))
            ),
            amount=clean(data.get("montant") or data.get("amount")),
            signatories=signatories,
        )

    def is_empty(self) -> bool:
        return not any(
            (self.client, self.contract_type, self.contract_date, self.amount, self.signatories)
        )


_UNAVAILABLE = "information non disponible dans les documents"


def format_answer(info: ContractInfo, fields: list[ContractField]) -> str:
    """Render the requested fields as a short French answer."""
    lines: list[str] = []
    for f in fields or ALL_FIELDS:
        if f is ContractField.CLIENT:
            lines.append(f"Nom du client : {info.client or _UNAVAILABLE}.")
        elif f is ContractField.CONTRACT_TYPE:
            lines.append(f"Type de contrat : {info.contract_type or _UNAVAILABLE}.")
        elif f is ContractField.CONTRACT_DATE:
            lines.append(f"Date du contrat : {info.contract_date or _UNAVAILABLE}.")
        elif f is ContractField.AMOUNT:
            if info.amount and normalize_text(info.amount) == CONFIDENTIAL:
                lines.append("Montant : confidentiel.")
            else:
                lines.append(f"Montant : {info.amount or _UNAVAILABLE}.")
        elif f in (ContractField.SIGNATORIES, ContractField.SIGNATURE_DATES):
            label = "Dates de signature" if f is ContractField.SIGNATURE_DATES else "Signataires"
            if f is ContractField.SIGNATURE_DATES:
                people = [s for s in info.signatories if s.signature_date]
            else:
                people = info.signatories
            if people:
                lines.append(f"{label} :\n" + "\n".join(f"- {s.display()}" for s in people))
            else:
                lines.append(f"{label} : {_UNAVAILABLE}.")
    return "\n".join(lines)
