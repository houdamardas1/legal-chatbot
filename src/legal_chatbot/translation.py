"""Optional French -> English translation of answers with Meta's M2M100."""

from __future__ import annotations

from functools import lru_cache

SUPPORTED_LANGUAGES = {"Français": "fr", "English": "en"}


@lru_cache(maxsize=2)
def _load(model_name: str):
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


def translate(
    text: str, src_lang: str, tgt_lang: str, model_name: str = "facebook/m2m100_418M"
) -> str:
    """Translate ``text`` line by line, so lists and line breaks survive the round trip."""
    if src_lang == tgt_lang or not text.strip():
        return text
    tokenizer, model = _load(model_name)
    tokenizer.src_lang = src_lang
    out_lines = []
    for line in text.splitlines():
        if not line.strip():
            out_lines.append(line)
            continue
        prefix = "- " if line.lstrip().startswith("- ") else ""
        content = line.lstrip()[2:] if prefix else line
        encoded = tokenizer(content, return_tensors="pt", truncation=True, max_length=512)
        generated = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
        out_lines.append(prefix + tokenizer.decode(generated[0], skip_special_tokens=True))
    return "\n".join(out_lines)
