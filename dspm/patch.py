# dspm/patch.py
import re
import hashlib
import tiktoken
import spacy
from dataclasses import dataclass, field
from typing import List

TOKENIZER_NAME = "cl100k_base"
MAX_PAYLOAD_CHARS = 200

SHORT_TAGS = {
    "constraint": "CON",
    "decision":   "DEC",
    "code":       "CODE",
    "equation":   "EQ",
    "entity":     "ENT",
    "structure":  "STR",
}

_enc = tiktoken.get_encoding(TOKENIZER_NAME)

try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    _nlp = None


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _fingerprint(text: str) -> str:
    return hashlib.md5(" ".join(sorted(text.lower().split())).encode()).hexdigest()[:16]


def _topic_keyword(text: str) -> str:
    if _nlp is not None:
        doc = _nlp(text[:200])
        for ent in doc.ents:
            return ent.text.lower().replace(" ", "_")
        nouns = [t.lemma_.lower() for t in doc
                 if t.pos_ in {"NOUN", "PROPN"} and len(t.text) > 2]
        if nouns:
            return nouns[0]
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return words[0] if words else "topic"


def _tag(ptype: str) -> str:
    return SHORT_TAGS.get(ptype, ptype[:3].upper())


@dataclass
class SemanticPatch:
    patch_id:     str
    turn_index:   int
    patch_type:   str
    payload:      str
    dependencies: List[str] = field(default_factory=list)
    utility:      float = 0.0
    token_cost:   int   = 0
    fingerprint:  str   = ""
    slot_key:     str   = ""
    is_delta:     bool  = False
    delta_base:   str   = ""
    causal_depth: int   = 0

    def __post_init__(self):
        self.payload     = " ".join(self.payload.split())[:MAX_PAYLOAD_CHARS]
        self.fingerprint = _fingerprint(self.payload)
        self.slot_key    = f"{self.patch_type}::{_topic_keyword(self.payload)}"
        self.recount()

    def recount(self):
        self.token_cost = count_tokens(f"[{_tag(self.patch_type)}] {self.payload}")

    def to_prompt_str(self) -> str:
        return f"[{_tag(self.patch_type)}] {self.payload}"
