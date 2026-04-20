"""
=============================================================================
preprocessing.py  –  Text Preprocessing Pipeline for VSM Information Retrieval
=============================================================================
Handles:
  • Tokenization
  • Case folding (lowercasing)
  • Stop-word removal  (loaded from provided stopwords file)
  • Lemmatization      (via NLTK WordNetLemmatizer)
=============================================================================
"""

import re
import os
import nltk
from nltk.stem import WordNetLemmatizer

for _pkg in ("punkt", "wordnet", "omw-1.4", "averaged_perceptron_tagger"):
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass


# =============================================================================
#  Preprocessor
# =============================================================================
class Preprocessor:
    _FALLBACK_STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "was", "are", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall",
        "that", "this", "it", "its", "not", "no", "so", "as", "if",
        "we", "i", "you", "he", "she", "they", "our", "their", "your",
        "my", "me", "him", "her", "us", "them", "what", "which", "who",
        "how", "when", "where", "there", "here", "just", "also",
    }

    def __init__(self, stopwords_path: str | None = None):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords  = self._load_stopwords(stopwords_path)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_stopwords(self, path: str | None) -> set:
        """Load stop-words from file; fall back to built-in set on error."""
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                words = {line.strip().lower() for line in fh if line.strip()}
            print(f"[Preprocessor] Loaded {len(words)} stop-words from '{path}'")
            return words
        print("[Preprocessor] Using built-in fallback stop-word list.")
        return self._FALLBACK_STOPWORDS.copy()

    # ── Public API ────────────────────────────────────────────────────────────

    def tokenize(self, text: str) -> list[str]:
        text   = text.lower()                         
        tokens = re.findall(r"[a-z]+(?:'[a-z]+)*", text)  
        return tokens

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if t not in self.stopwords]

    def lemmatize(self, tokens: list[str]) -> list[str]:
        result = []
        for token in tokens:
            verb_form = self.lemmatizer.lemmatize(token, pos="v")
            result.append(verb_form)
        return result

    def process(self, text: str) -> list[str]:
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return tokens

    def process_query(self, query: str) -> list[str]:
        return self.process(query)