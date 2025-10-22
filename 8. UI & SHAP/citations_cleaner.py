# citations_cleaner.py — compact citation remover (strict=True default)
import re
from typing import List

# -----------------------------
# Normalization helper
# -----------------------------
def normalize_text_for_citations(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.replace("\u00ad", "")                      # remove soft hyphen
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)          # "Secu- rity" -> "Security"
    s = s.replace("–", "-").replace("—", "-")        # normalize dashes
    s = re.sub(r"\s{2,}", " ", s)
    return s

# -----------------------------
# Build regex library (Unicode-aware)
# -----------------------------
ULTRAW = r"[^\W\d_]"
WORD  = rf"{ULTRAW}[{ULTRAW}'’\-]*"
NAME  = rf"{WORD}(?:\s+{WORD})*"
CORP  = rf"(?:[A-Z]{{2,}}(?:\s*&\s*[A-Z]{{2,}})*|{WORD}(?:\s+{WORD}){{0,5}})"
AUTHOR_LIST = rf"(?:{NAME}(?:\s*(?:,|&|and)\s*{NAME})*|{CORP})(?:\s+et al\.)?"
YEAR = r"(?:19|20)\d{2}[a-z]?"
LOC = r"(?:p{1,2}\.?|pp\.?|ch\.|sec\.|§)\s*\d+(?:\s*[-–-]\s*\d+)?"
LOC_OPT = rf"(?:\s*(?:,|:)\s*{LOC})?"
SEP = r"(?:,\s*|\s+)"

ITEM = rf"{AUTHOR_LIST}\s*{SEP}{YEAR}{LOC_OPT}"
CIT_SEP = r"(?:\s*;\s*|\s*,\s*(?=[A-Z]))"
ITEMS = rf"{ITEM}(?:{CIT_SEP}{ITEM})*"
PREFIX = r"(?:e\.g\.,\s*|see(?:\s+also)?\s+|cf\.\s+)?"
APA_PARENS = rf"\(\s*{PREFIX}{ITEMS}\s*\)"

ITEM_LOC_FIRST = rf"{LOC}\s+{AUTHOR_LIST}\s*{SEP}{YEAR}"
ITEMS_LOC_FIRST = rf"{ITEM_LOC_FIRST}(?:{CIT_SEP}{ITEM_LOC_FIRST})*"
APA_PARENS_LOC_FIRST = rf"\(\s*{PREFIX}{ITEMS_LOC_FIRST}\s*\)"

# Heuristics (tight + general)
LEADING_NUM_SEMI_TIGHT = rf"\(\s*\d+\s*;\s*(?=[^)]*{AUTHOR_LIST}\s*(?:,\s*|\s+){YEAR})[^)]*\)"
COMMA_NUMBER_FOLLOWED_BY_AUTHYEAR = rf"\([^)]*,\s*\d{{1,4}}\s+(?={AUTHOR_LIST}\s*(?:,\s*|\s+){YEAR})[^)]*\)"
YEAR_SEMICOLON = rf"\([^)]*{YEAR}\s*;[^)]*\)"
LOC_ANYWHERE = rf"\([^)]*(?:p{1,2}\.?|pp\.?|ch\.|sec\.|§)\s*\d+[^)]*\)"
AS_CITED_IN = r"\([^)]*\bas cited in\b[^)]*\)"
BRACKET_NUMERIC = r"\[(?:\s*\d{1,3}(?:\s*[-–-]\s*\d{1,3})?\s*)(?:,\s*\d{1,3}(?:\s*[-–-]\s*\d{1,3})?\s*)*\]"
AUTHOR_SQNUM = r"\b([A-Z][A-Za-z'’\-]+)\s*\[\d+\]"
SUPERSCRIPT_LATEX = r"\^\{?\d{1,3}\}?"
SUPERSCRIPT_UNICODE = r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+"
PAREN_YEAR = r"\(\s*(?:19|20)\d{2}[a-z]?\s*\)"
BRACKET_YEAR = r"\[\s*(?:19|20)\d{2}[a-z]?\s*\]"
ETAL_YEAR_NOCOMMA = r"\([^)]*et al\.\s+(?:19|20)\d{2}[^)]*\)"
ETAL_YEAR_COMMA   = r"\([^)]*et al\.,\s*(?:19|20)\d{2}[^)]*\)"

# Compile substitution patterns (specific -> general)
regex_subs = [
    (re.compile(ETAL_YEAR_COMMA, re.IGNORECASE), " "),
    (re.compile(ETAL_YEAR_NOCOMMA, re.IGNORECASE), " "),
    (re.compile(AS_CITED_IN, re.IGNORECASE), " "),
    (re.compile(BRACKET_NUMERIC), " "),
    (re.compile(AUTHOR_SQNUM), r"\1"),
    (re.compile(SUPERSCRIPT_LATEX), " "),
    (re.compile(SUPERSCRIPT_UNICODE), " "),
    (re.compile(APA_PARENS_LOC_FIRST), " "),
    (re.compile(APA_PARENS), " "),
    (re.compile(LEADING_NUM_SEMI_TIGHT), " "),
    (re.compile(COMMA_NUMBER_FOLLOWED_BY_AUTHYEAR), " "),
    (re.compile(YEAR_SEMICOLON), " "),
    (re.compile(LOC_ANYWHERE), " "),
    (re.compile(PAREN_YEAR), " "),
    (re.compile(BRACKET_YEAR), " "),
]

# Strict-mode catch-alls (remove any (...) or [...] with a year)
STRICT_PARENS_YEAR  = re.compile(r"\([^)]*(?:19|20)\d{2}[^)]*\)", re.IGNORECASE)
STRICT_BRACKS_YEAR  = re.compile(r"\[[^\]]*(?:19|20)\d{2}[^\]]*\]", re.IGNORECASE)

# Cleanup helpers
_ws_collapse = re.compile(r"\s{2,}")
_space_before_punct = re.compile(r"\s+([,.;:)\]])")

def strip_citations(text: str, strict: bool = True) -> str:
    """Remove APA/IEEE-style citations; defaults to strict=True as requested."""
    if not isinstance(text, str):
        return text
    s = normalize_text_for_citations(text)
    for pat, repl in regex_subs:
        s = pat.sub(repl, s)
    if strict:
        s = STRICT_PARENS_YEAR.sub(" ", s)
        s = STRICT_BRACKS_YEAR.sub(" ", s)
    s = _ws_collapse.sub(" ", s)
    s = _space_before_punct.sub(r"\1", s)
    s = re.sub(r"\(\s*\)", " ", s)
    s = re.sub(r"\[\s*\]", " ", s)
    return s.strip()
