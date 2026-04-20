"""
=============================================================================
vsm_retrieval.py  –  Vector Space Model Retrieval Engine
=============================================================================
Core algorithm
  1. Parse and preprocess the free-text query.
  2. Build a TF-IDF query vector (using the same IDF values as the corpus).
  3. Compute cosine similarity between the query vector and every
     document vector using the *inverted-index* (only iterate over
     documents that contain at least one query term – efficient).
  4. Filter results by alpha threshold.
  5. Return ranked list of (doc_id, score) pairs.

Cosine similarity
  sim(q, d) = (q · d) / (|q| × |d|)
=============================================================================
"""

import math
from collections import defaultdict
from indexer      import Indexer
from preprocessing import Preprocessor


# =============================================================================
#  VSMRetriever
# =============================================================================
class VSMRetriever:
    """
    Retrieves and ranks documents for a free-text query using cosine similarity
    over TF-IDF weighted vectors.

    Parameters
    ----------
    indexer       : Indexer        – pre-built / pre-loaded index.
    preprocessor  : Preprocessor  – same pipeline used during indexing.
    alpha         : float          – minimum cosine similarity threshold
                                     (default 0.005, as per assignment spec).
    top_k         : int            – maximum number of results to return.
    """

    def __init__(
        self,
        indexer:      Indexer,
        preprocessor: Preprocessor,
        alpha:        float = 0.005,
        top_k:        int   = 10,
    ):
        self.indexer      = indexer
        self.preprocessor = preprocessor
        self.alpha        = alpha
        self.top_k        = top_k

    # =========================================================================
    #  Query vector
    # =========================================================================

    def _build_query_vector(self, query_tokens: list[str]) -> dict[str, float]:
        """
        Build a TF-IDF query vector.

        Raw TF is computed from the query string, then log-normalised.
        IDF is borrowed from the corpus index.  Terms not in the
        vocabulary are silently ignored.
        """
        raw_tf: dict[str, int] = defaultdict(int)
        for token in query_tokens:
            if token in self.indexer.vocabulary:
                raw_tf[token] += 1

        query_vec: dict[str, float] = {}
        for term, tf in raw_tf.items():
            tf_w = 1 + math.log10(tf) if tf > 0 else 0
            query_vec[term] = tf_w * self.indexer.idf[term]

        return query_vec

    # =========================================================================
    #  Cosine similarity  (optimised via inverted index)
    # =========================================================================

    def _cosine_similarity(
        self,
        query_vec: dict[str, float],
    ) -> dict[int, float]:
        """
        Compute the dot-product accumulator via the inverted index.

        Only documents that share at least one term with the query are
        visited, making this efficient for large corpora.

        Returns  {doc_id: cosine_similarity}
        """
        # Accumulate dot-product scores
        dot_products: dict[int, float] = defaultdict(float)

        for term, q_weight in query_vec.items():
            postings = self.indexer.inverted_index.get(term, {})
            for doc_id, raw_tf in postings.items():
                # Re-compute tf-idf weight for this doc-term pair
                tf_w        = 1 + math.log10(raw_tf) if raw_tf > 0 else 0
                d_weight    = tf_w * self.indexer.idf[term]
                dot_products[doc_id] += q_weight * d_weight

        # Query L2 norm
        q_norm = math.sqrt(sum(v * v for v in query_vec.values()))
        if q_norm == 0:
            return {}

        # Normalise by both query and document norms
        scores: dict[int, float] = {}
        for doc_id, dot in dot_products.items():
            d_norm = self.indexer.get_doc_norm(doc_id)
            if d_norm > 0:
                scores[doc_id] = dot / (q_norm * d_norm)

        return scores

    # =========================================================================
    #  Public search interface
    # =========================================================================

    def search(self, query: str) -> list[dict]:
        """
        Execute a free-text query and return a ranked result list.

        Parameters
        ----------
        query : str  – raw free-text query string.

        Returns
        -------
        list of dict, each containing:
          rank        – 1-based rank
          doc_id      – internal document identifier
          doc_name    – filename of the speech
          score       – cosine similarity score
          query_terms – list of query terms found in this document
        """
        # ── Step 1: preprocess the query ─────────────────────────────────────
        query_tokens = self.preprocessor.process_query(query)

        if not query_tokens:
            return []

        # ── Step 2: build query vector ────────────────────────────────────────
        query_vec = self._build_query_vector(query_tokens)

        if not query_vec:
            print(f"[VSMRetriever] No query terms found in vocabulary.")
            return []

        # ── Step 3: cosine similarity ─────────────────────────────────────────
        scores = self._cosine_similarity(query_vec)

        # ── Step 4: alpha filtering ───────────────────────────────────────────
        filtered = {
            doc_id: score
            for doc_id, score in scores.items()
            if score >= self.alpha
        }

        # ── Step 5: rank (descending score), limit to top_k ──────────────────
        ranked = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        ranked = ranked[: self.top_k]

        # ── Step 6: build result records ──────────────────────────────────────
        results = []
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            doc_vec     = self.indexer.get_tfidf_vector(doc_id)
            query_terms = [t for t in query_vec if t in doc_vec]
            results.append(
                {
                    "rank":        rank,
                    "doc_id":      doc_id,
                    "doc_name":    self.indexer.get_doc_name(doc_id),
                    "score":       round(score, 6),
                    "query_terms": query_terms,
                }
            )

        return results

    # =========================================================================
    #  Batch evaluation (for the 10 provided queries)
    # =========================================================================

    def evaluate_queries(self, queries_path: str) -> dict[str, list[dict]]:
        """
        Read queries from *queries_path* (one query per line) and run each
        through the retrieval engine.

        Returns {query_string: [result, ...]}
        """
        if not __import__("os").path.isfile(queries_path):
            raise FileNotFoundError(f"Queries file not found: '{queries_path}'")

        with open(queries_path, "r", encoding="utf-8", errors="ignore") as fh:
            queries = [line.strip() for line in fh if line.strip()]

        all_results: dict[str, list[dict]] = {}
        for i, q in enumerate(queries, start=1):
            print(f"\n[Query {i:02d}] '{q}'")
            results = self.search(q)
            all_results[q] = results
            for r in results:
                print(
                    f"  Rank {r['rank']:>2} | Score {r['score']:.6f} | "
                    f"{r['doc_name']}  | Terms: {r['query_terms']}"
                )
            if not results:
                print("  → No documents above alpha threshold.")

        return all_results