"""
=============================================================================
main.py  –  Command-Line Interface for the VSM Information Retrieval System
=============================================================================
Usage examples
  # Build a fresh index from the corpus
  python main.py --corpus data/TrumpSpeeches \
                 --stopwords data/stopwords.txt \
                 --rebuild

  # Run all 10 evaluation queries
  python main.py --corpus data/TrumpSpeeches \
                 --stopwords data/stopwords.txt \
                 --queries data/queries.txt

  # Interactive free-text search session
  python main.py --corpus data/TrumpSpeeches \
                 --stopwords data/stopwords.txt \
                 --interactive

  # Launch the Tkinter GUI
  python main.py --corpus data/TrumpSpeeches \
                 --stopwords data/stopwords.txt \
                 --gui
=============================================================================
"""

import argparse
import os
import sys

from preprocessing import Preprocessor
from indexer        import Indexer
from vsm_retrieval  import VSMRetriever


# =============================================================================
#  Bootstrap helpers
# =============================================================================

INDEX_FILE = "vsm_index.pkl"


def build_or_load_index(
    corpus_dir:     str,
    stopwords_path: str | None,
    force_rebuild:  bool = False,
    min_tf:         int  = 1,
    min_df:         int  = 1,
) -> tuple[Preprocessor, Indexer]:
    """
    Either load a cached index or build one from the raw corpus.

    Returns (preprocessor, indexer).
    """
    preprocessor = Preprocessor(stopwords_path=stopwords_path)
    indexer      = Indexer(preprocessor, min_tf=min_tf, min_df=min_df)

    if not force_rebuild and indexer.load(INDEX_FILE):
        return preprocessor, indexer

    # Build from scratch
    if not os.path.isdir(corpus_dir):
        print(f"[ERROR] Corpus directory not found: '{corpus_dir}'", file=sys.stderr)
        sys.exit(1)

    indexer.build_from_directory(corpus_dir)
    indexer.save(INDEX_FILE)
    return preprocessor, indexer


# =============================================================================
#  Interactive search loop
# =============================================================================

def interactive_session(retriever: VSMRetriever) -> None:
    """Run an interactive command-line search loop."""
    print("\n" + "=" * 60)
    print("  VSM Information Retrieval  –  Interactive Search")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        try:
            query = input("\nEnter query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if not query:
            continue

        results = retriever.search(query)

        if not results:
            print("  No documents above the alpha threshold.")
            continue

        print(f"\n  {'Rank':<5} {'Score':>10}  Document")
        print(f"  {'-'*5} {'-'*10}  {'-'*40}")
        for r in results:
            print(
                f"  {r['rank']:<5} {r['score']:>10.6f}  "
                f"{r['doc_name']}  "
                f"[{', '.join(r['query_terms'][:5])}]"
            )


# =============================================================================
#  Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VSM Information Retrieval System – Trump Speeches Corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--corpus",     required=True,
        help="Path to directory containing the Trump Speeches .txt files",
    )
    parser.add_argument(
        "--stopwords",  default=None,
        help="Path to stopwords list file",
    )
    parser.add_argument(
        "--queries",    default=None,
        help="Path to queries file (one query per line); runs batch evaluation",
    )
    parser.add_argument(
        "--rebuild",    action="store_true",
        help="Force a full index rebuild even if a cached index exists",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Start an interactive CLI search session",
    )
    parser.add_argument(
        "--gui",        action="store_true",
        help="Launch the Tkinter graphical interface",
    )
    parser.add_argument(
        "--alpha",      type=float, default=0.005,
        help="Minimum cosine similarity threshold for result inclusion",
    )
    parser.add_argument(
        "--top_k",      type=int,   default=10,
        help="Maximum number of ranked results returned per query",
    )
    parser.add_argument(
        "--min_tf",     type=int,   default=1,
        help="Minimum raw term-frequency for a term to be indexed",
    )
    parser.add_argument(
        "--min_df",     type=int,   default=2,
        help="Minimum document-frequency for a term to enter the vocabulary",
    )
    return parser.parse_args()


# =============================================================================
#  Entry point
# =============================================================================

def main() -> None:
    args = parse_args()

    # ── Build / load index ─────────────────────────────────────────────────
    preprocessor, indexer = build_or_load_index(
        corpus_dir     = args.corpus,
        stopwords_path = args.stopwords,
        force_rebuild  = args.rebuild,
        min_tf         = args.min_tf,
        min_df         = args.min_df,
    )

    # ── Instantiate retriever ─────────────────────────────────────────────
    retriever = VSMRetriever(
        indexer      = indexer,
        preprocessor = preprocessor,
        alpha        = args.alpha,
        top_k        = args.top_k,
    )

    # ── Dispatch to requested mode ────────────────────────────────────────
    if args.gui:
        from gui import launch_gui
        launch_gui(retriever)
        return

    if args.queries:
        retriever.evaluate_queries(args.queries)
        return

    if args.interactive:
        interactive_session(retriever)
        return

    # Default: print usage hint
    print(
        "\n[INFO] No action specified.  "
        "Use --queries, --interactive, or --gui."
    )
    print("Run  python main.py --help  for full options.")


if __name__ == "__main__":
    main()