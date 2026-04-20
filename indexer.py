"""
=============================================================================
indexer.py  –  Inverted Index & TF-IDF Weight Builder for VSM
=============================================================================
Responsibilities
  • Build an inverted index:  term  →  {doc_id: raw_tf, ...}
  • Compute TF-IDF weights   (log-normalised tf  ×  idf)
  • Persist the index to disk (JSON) and reload it on subsequent runs
  • Support tf-threshold and df-threshold for feature selection

TF-IDF weighting scheme
  tf_weight  = 1 + log10(raw_tf)      if raw_tf > 0   (log-normalised tf)
  idf        = log10( N / df )
  tf_idf     = tf_weight × idf
=============================================================================
"""

import os
import json
import math
import pickle
from collections import defaultdict
from preprocessing import Preprocessor


# =============================================================================
#  Indexer
# =============================================================================
class Indexer:
    """
    Builds and manages an inverted index of TF-IDF weighted term vectors.

    Parameters
    ----------
    preprocessor : Preprocessor
        An already-configured preprocessing pipeline.
    min_tf  : int
        Ignore term occurrences below this raw-tf threshold per document.
    min_df  : int
        Ignore terms that appear in fewer than *min_df* documents.
    max_df_ratio : float
        Ignore terms that appear in more than this fraction of documents
        (very common, low-signal terms).  Default 0.95.
    """

    def __init__(
        self,
        preprocessor: Preprocessor,
        min_tf: int   = 1,
        min_df: int   = 1,
        max_df_ratio: float = 0.95,
    ):
        self.preprocessor  = preprocessor
        self.min_tf        = min_tf
        self.min_df        = min_df
        self.max_df_ratio  = max_df_ratio

        # ── Core data structures ──────────────────────────────────────────────
        # inverted_index[term] = {doc_id: raw_tf}
        self.inverted_index: dict[str, dict[int, int]]   = defaultdict(dict)

        # doc_lengths[doc_id] = list of (term, tf_idf) for building vectors
        self.doc_tfidf:      dict[int, dict[str, float]] = {}

        # document registry:  doc_id  →  file path / name
        self.doc_registry:   dict[int, str]              = {}

        # vocabulary after df-pruning
        self.vocabulary:     set[str]                    = set()

        # IDF values per term
        self.idf:            dict[str, float]            = {}

        # document L2 norms (for cosine similarity denominator)
        self.doc_norms:      dict[int, float]            = {}

        self._num_docs: int = 0

    # =========================================================================
    #  Index construction
    # =========================================================================

    def build_from_directory(self, corpus_dir: str) -> None:
        """
        Read every .txt file in *corpus_dir*, preprocess, and build the index.
        """
        files = sorted(
            f for f in os.listdir(corpus_dir) if f.lower().endswith(".txt")
        )
        if not files:
            raise FileNotFoundError(
                f"No .txt files found in directory: '{corpus_dir}'"
            )

        print(f"[Indexer] Found {len(files)} documents. Building index …")

        # ── Pass 1: compute raw term frequencies per document ─────────────────
        raw_index: dict[str, dict[int, int]] = defaultdict(dict)

        for doc_id, filename in enumerate(files):
            filepath = os.path.join(corpus_dir, filename)
            self.doc_registry[doc_id] = filename

            with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()

            tokens = self.preprocessor.process(text)

            # Count raw term frequencies
            tf_counts: dict[str, int] = defaultdict(int)
            for token in tokens:
                tf_counts[token] += 1

            for term, raw_tf in tf_counts.items():
                if raw_tf >= self.min_tf:
                    raw_index[term][doc_id] = raw_tf

        self._num_docs = len(files)

        # ── Pass 2: df-based feature selection ───────────────────────────────
        max_df_abs = int(self.max_df_ratio * self._num_docs)

        for term, postings in raw_index.items():
            df = len(postings)
            if self.min_df <= df <= max_df_abs:
                self.inverted_index[term] = dict(postings)
                self.vocabulary.add(term)

        print(
            f"[Indexer] Vocabulary size after df-pruning: "
            f"{len(self.vocabulary):,} terms"
        )

        # ── Pass 3: compute IDF & TF-IDF weights ─────────────────────────────
        self._compute_idf()
        self._compute_tfidf_vectors()
        print("[Indexer] Index build complete.")

    # ── IDF computation ───────────────────────────────────────────────────────

    def _compute_idf(self) -> None:
        """
        IDF = log10( N / df )
        Stored for every term in the pruned vocabulary.
        """
        N = self._num_docs
        for term in self.vocabulary:
            df = len(self.inverted_index[term])
            self.idf[term] = math.log10(N / df)

    # ── TF-IDF vector computation ─────────────────────────────────────────────

    def _compute_tfidf_vectors(self) -> None:
        """
        For each document build a sparse TF-IDF vector and store its L2 norm.

        tf_weight = 1 + log10(raw_tf)
        tfidf     = tf_weight × idf
        """
        for doc_id in range(self._num_docs):
            vec: dict[str, float] = {}
            for term in self.vocabulary:
                postings = self.inverted_index.get(term, {})
                raw_tf   = postings.get(doc_id, 0)
                if raw_tf > 0:
                    tf_w = 1 + math.log10(raw_tf)
                    vec[term] = tf_w * self.idf[term]

            self.doc_tfidf[doc_id] = vec
            # L2 norm for cosine similarity
            self.doc_norms[doc_id] = math.sqrt(sum(v * v for v in vec.values()))

    # =========================================================================
    #  Persistence  (save / load)
    # =========================================================================

    def save(self, index_path: str = "vsm_index.pkl") -> None:
        """Serialise the entire index to a pickle file."""
        payload = {
            "inverted_index": dict(self.inverted_index),
            "doc_tfidf":      self.doc_tfidf,
            "doc_registry":   self.doc_registry,
            "vocabulary":     self.vocabulary,
            "idf":            self.idf,
            "doc_norms":      self.doc_norms,
            "num_docs":       self._num_docs,
        }
        with open(index_path, "wb") as fh:
            pickle.dump(payload, fh)
        print(f"[Indexer] Index saved → '{index_path}'")

    def load(self, index_path: str = "vsm_index.pkl") -> bool:
        """
        Load a previously saved index.
        Returns True on success, False if the file does not exist.
        """
        if not os.path.isfile(index_path):
            return False
        with open(index_path, "rb") as fh:
            payload = pickle.load(fh)

        self.inverted_index = defaultdict(dict, payload["inverted_index"])
        self.doc_tfidf      = payload["doc_tfidf"]
        self.doc_registry   = payload["doc_registry"]
        self.vocabulary     = payload["vocabulary"]
        self.idf            = payload["idf"]
        self.doc_norms      = payload["doc_norms"]
        self._num_docs      = payload["num_docs"]
        print(
            f"[Indexer] Index loaded from '{index_path}' "
            f"({self._num_docs} docs, {len(self.vocabulary):,} terms)"
        )
        return True

    # =========================================================================
    #  Convenience accessors
    # =========================================================================

    @property
    def num_docs(self) -> int:
        return self._num_docs

    def get_doc_name(self, doc_id: int) -> str:
        return self.doc_registry.get(doc_id, f"doc_{doc_id}")

    def get_tfidf_vector(self, doc_id: int) -> dict[str, float]:
        return self.doc_tfidf.get(doc_id, {})

    def get_doc_norm(self, doc_id: int) -> float:
        return self.doc_norms.get(doc_id, 0.0)