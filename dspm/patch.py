# dspm/patch.py
import re
import hashlib
import tiktoken
from dataclasses import dataclass, field
from typing import List

TOKENIZER_NAME = "cl100k_base"
MAX_PAYLOAD_CHARS = 200

SHORT_TAGS = {
    "constraint": "CON",
    "decision":   "DEC",
    "slot":       "SLOT",
    "code":       "CODE",
    "equation":   "EQ",
    "entity":     "ENT",
    "structure":  "STR",
}

_enc = tiktoken.get_encoding(TOKENIZER_NAME)

# v0.6.0: spacy imported lazily for lightweight install
_spacy_loaded = False
_nlp = None

def _load_spacy():
    """Load spacy model lazily. Returns None if not available."""
    global _nlp, _spacy_loaded
    if not _spacy_loaded:
        _spacy_loaded = True
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            _nlp = None
    return _nlp



def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _fingerprint(text: str) -> str:
    """Ordered normalized text fingerprint (not bag-of-words).
    Sorted text made opposite-meaning rules collide:
      "admins can delete, guests cannot" == "guests can delete, admins cannot"
    Ordered text preserves meaning.
    """
    normalized = " ".join(text.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()[:16]



# v0.6.1: Common words skipped in topic keyword fallback
_COMMON_WORDS = {
    "max", "maximum", "min", "minimum", "must", "mustnt",
    "the", "all", "always", "never", "not", "no", "non",
    "use", "using", "set", "add", "update", "change",
    "is", "are", "was", "were", "be", "been",
    "for", "per", "each", "every", "any",
    "and", "or", "but", "if", "when", "then",
    "a", "an", "in", "on", "at", "to", "of", "by",
    "do", "does", "did", "done",
    "this", "that", "these", "those",
    "it", "its", "they", "them", "their",
    "will", "would", "should", "could", "can",
    "has", "have", "had",
}

def _topic_keyword(text: str) -> str:
    """Extract a distinguishing topic keyword from text.
    v0.6.1: Skips common words, includes numeric values.
    Fixes Baidik #5: different rules get different slot_keys.
    """
    nlp = _load_spacy()
    if nlp is not None:
        doc = nlp(text[:200])
        # v0.6.1: Check entities first, but skip common words
        for ent in doc.ents:
            ent_text = ent.text.lower().replace(" ", "_")
            if ent_text not in _COMMON_WORDS:
                return ent_text
        # v0.6.1: Collect ALL nouns, return first that's NOT common
        nouns = [t.lemma_.lower() for t in doc
                 if t.pos_ in {"NOUN", "PROPN"} and len(t.text) > 2]
        meaningful_nouns = [n for n in nouns if n not in _COMMON_WORDS]
        if meaningful_nouns:
            return meaningful_nouns[0]
        elif nouns:
            return nouns[0]

    # v0.6.1: Better fallback — skip common words, include numbers
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    meaningful = [w for w in words if w not in _COMMON_WORDS]
    numbers = re.findall(r"\d+[.%a-z]*", text.lower())
    if numbers:
        key_word = meaningful[0] if meaningful else "rule"
        return key_word + "_" + numbers[0]
    if meaningful:
        return meaningful[0]
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
