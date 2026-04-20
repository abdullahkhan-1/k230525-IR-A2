"""
=============================================================================
gui.py  –  Tkinter GUI for the VSM Information Retrieval System
=============================================================================
Features
  • Free-text query entry with Enter-key shortcut
  • Results table (rank, score, document name, matched terms)
  • Cosine-similarity score bar per result
  • Index statistics panel
  • Alpha threshold & top-K sliders
  • Query history dropdown
=============================================================================
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
from vsm_retrieval import VSMRetriever


# =============================================================================
#  Colour palette & fonts
# =============================================================================
COLOURS = {
    "bg":        "#0f1117",
    "panel":     "#1a1d27",
    "accent":    "#3b82f6",
    "accent2":   "#f59e0b",
    "text":      "#e2e8f0",
    "subtext":   "#94a3b8",
    "success":   "#22c55e",
    "border":    "#2d3748",
    "row_even":  "#1e2130",
    "row_odd":   "#161925",
    "entry_bg":  "#252836",
}

FONT_TITLE  = ("Segoe UI", 20, "bold")
FONT_HEAD   = ("Segoe UI", 11, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_MONO   = ("Consolas",  9)
FONT_SMALL  = ("Segoe UI",  9)


# =============================================================================
#  Main Application Window
# =============================================================================
class VSMApp(tk.Tk):
    """Full Tkinter application for the VSM Information Retrieval system."""

    def __init__(self, retriever: VSMRetriever):
        super().__init__()
        self.retriever     = retriever
        self.query_history: list[str] = []

        # ── Window setup ────────────────────────────────────────────────────
        self.title("VSM Information Retrieval  –  Trump Speeches")
        self.geometry("1100x750")
        self.configure(bg=COLOURS["bg"])
        self.resizable(True, True)

        # ttk style overrides
        self._apply_style()

        # ── Build UI ────────────────────────────────────────────────────────
        self._build_header()
        self._build_search_bar()
        self._build_params_bar()
        self._build_results_area()
        self._build_status_bar()

        # ── Pre-fill stats ─────────────────────────────────────────────────
        self._update_status("Index ready. Enter a query above.")

    # =========================================================================
    #  Style
    # =========================================================================

    def _apply_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")

        # Treeview (results table)
        style.configure(
            "Custom.Treeview",
            background    = COLOURS["row_odd"],
            fieldbackground = COLOURS["row_odd"],
            foreground    = COLOURS["text"],
            rowheight     = 28,
            font          = FONT_BODY,
            borderwidth   = 0,
        )
        style.configure(
            "Custom.Treeview.Heading",
            background  = COLOURS["panel"],
            foreground  = COLOURS["accent"],
            font        = FONT_HEAD,
            borderwidth = 0,
        )
        style.map(
            "Custom.Treeview",
            background  = [("selected", COLOURS["accent"])],
            foreground  = [("selected", "#ffffff")],
        )

        # Scrollbar
        style.configure(
            "Custom.Vertical.TScrollbar",
            background  = COLOURS["panel"],
            troughcolor = COLOURS["bg"],
            arrowcolor  = COLOURS["subtext"],
        )

        # Combobox
        style.configure(
            "Custom.TCombobox",
            fieldbackground = COLOURS["entry_bg"],
            background      = COLOURS["panel"],
            foreground      = COLOURS["text"],
            selectbackground = COLOURS["accent"],
        )

    # =========================================================================
    #  Header
    # =========================================================================

    def _build_header(self) -> None:
        hdr = tk.Frame(self, bg=COLOURS["panel"], pady=14)
        hdr.pack(fill="x", padx=0, pady=0)

        tk.Label(
            hdr, text="🔍  VSM Information Retrieval",
            font=FONT_TITLE, bg=COLOURS["panel"], fg=COLOURS["accent"],
        ).pack(side="left", padx=20)

        # Index stats on the right
        idx  = self.retriever.indexer
        info = (
            f"📄 {idx.num_docs} documents   "
            f"📚 {len(idx.vocabulary):,} terms in vocabulary"
        )
        tk.Label(
            hdr, text=info,
            font=FONT_SMALL, bg=COLOURS["panel"], fg=COLOURS["subtext"],
        ).pack(side="right", padx=20)

    # =========================================================================
    #  Search bar
    # =========================================================================

    def _build_search_bar(self) -> None:
        bar = tk.Frame(self, bg=COLOURS["bg"], pady=10)
        bar.pack(fill="x", padx=20)

        tk.Label(
            bar, text="Query:", font=FONT_HEAD,
            bg=COLOURS["bg"], fg=COLOURS["text"],
        ).pack(side="left", padx=(0, 8))

        # Query entry box
        self.query_var = tk.StringVar()
        self.entry = tk.Entry(
            bar, textvariable=self.query_var,
            font=("Segoe UI", 12),
            bg=COLOURS["entry_bg"], fg=COLOURS["text"],
            insertbackground=COLOURS["text"],
            relief="flat", bd=6, width=65,
        )
        self.entry.pack(side="left", ipady=6, padx=(0, 10))
        self.entry.bind("<Return>", lambda _: self._run_search())
        self.entry.focus_set()

        # History dropdown
        self.hist_var = tk.StringVar()
        self.hist_cb  = ttk.Combobox(
            bar, textvariable=self.hist_var,
            style="Custom.TCombobox",
            state="readonly", width=20,
        )
        self.hist_cb.pack(side="left", padx=(0, 10))
        self.hist_cb.bind("<<ComboboxSelected>>", self._load_history)

        # Search button
        self.search_btn = tk.Button(
            bar, text="  Search  ",
            font=FONT_HEAD,
            bg=COLOURS["accent"], fg="#ffffff",
            activebackground="#2563eb", activeforeground="#ffffff",
            relief="flat", bd=0, padx=12, pady=6,
            cursor="hand2",
            command=self._run_search,
        )
        self.search_btn.pack(side="left")

        # Clear button
        tk.Button(
            bar, text="✕",
            font=FONT_HEAD,
            bg=COLOURS["panel"], fg=COLOURS["subtext"],
            activebackground=COLOURS["border"],
            relief="flat", bd=0, padx=8, pady=6,
            cursor="hand2",
            command=self._clear,
        ).pack(side="left", padx=6)

    # =========================================================================
    #  Parameters bar
    # =========================================================================

    def _build_params_bar(self) -> None:
        bar = tk.Frame(self, bg=COLOURS["panel"], pady=8)
        bar.pack(fill="x", padx=0)

        # Alpha slider
        tk.Label(
            bar, text="  α threshold:",
            font=FONT_SMALL, bg=COLOURS["panel"], fg=COLOURS["subtext"],
        ).pack(side="left", padx=(20, 4))

        self.alpha_var = tk.DoubleVar(value=self.retriever.alpha)
        alpha_lbl = tk.Label(
            bar, textvariable=self.alpha_var,
            font=FONT_MONO, bg=COLOURS["panel"], fg=COLOURS["accent2"],
            width=6,
        )
        self.alpha_var.trace_add(
            "write",
            lambda *_: alpha_lbl.config(text=f"{self.alpha_var.get():.4f}"),
        )
        tk.Scale(
            bar, variable=self.alpha_var,
            from_=0.001, to=0.1, resolution=0.001,
            orient="horizontal", length=160,
            bg=COLOURS["panel"], fg=COLOURS["text"],
            troughcolor=COLOURS["border"], highlightthickness=0,
            showvalue=False,
        ).pack(side="left")
        alpha_lbl.pack(side="left", padx=(4, 20))

        # Top-K slider
        tk.Label(
            bar, text="Top-K:",
            font=FONT_SMALL, bg=COLOURS["panel"], fg=COLOURS["subtext"],
        ).pack(side="left", padx=(0, 4))

        self.topk_var = tk.IntVar(value=self.retriever.top_k)
        topk_lbl = tk.Label(
            bar, textvariable=self.topk_var,
            font=FONT_MONO, bg=COLOURS["panel"], fg=COLOURS["accent2"],
            width=4,
        )
        tk.Scale(
            bar, variable=self.topk_var,
            from_=1, to=56, resolution=1,
            orient="horizontal", length=120,
            bg=COLOURS["panel"], fg=COLOURS["text"],
            troughcolor=COLOURS["border"], highlightthickness=0,
            showvalue=False,
        ).pack(side="left")
        topk_lbl.pack(side="left", padx=(4, 0))

    # =========================================================================
    #  Results area
    # =========================================================================

    def _build_results_area(self) -> None:
        main = tk.Frame(self, bg=COLOURS["bg"])
        main.pack(fill="both", expand=True, padx=20, pady=10)

        # ── Left: results table ──────────────────────────────────────────────
        left = tk.Frame(main, bg=COLOURS["bg"])
        left.pack(side="left", fill="both", expand=True)

        tk.Label(
            left, text="Ranked Results",
            font=FONT_HEAD, bg=COLOURS["bg"], fg=COLOURS["text"],
        ).pack(anchor="w", pady=(0, 4))

        cols = ("rank", "score", "doc_name", "query_terms")
        self.tree = ttk.Treeview(
            left, columns=cols, show="headings",
            style="Custom.Treeview", selectmode="browse",
        )
        self.tree.heading("rank",        text="#")
        self.tree.heading("score",       text="Score")
        self.tree.heading("doc_name",    text="Document")
        self.tree.heading("query_terms", text="Matched Terms")

        self.tree.column("rank",        width=40,  anchor="center")
        self.tree.column("score",       width=90,  anchor="center")
        self.tree.column("doc_name",    width=320, anchor="w")
        self.tree.column("query_terms", width=300, anchor="w")

        vsb = ttk.Scrollbar(
            left, orient="vertical",
            command=self.tree.yview,
            style="Custom.Vertical.TScrollbar",
        )
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # Alternating row tags
        self.tree.tag_configure("even", background=COLOURS["row_even"])
        self.tree.tag_configure("odd",  background=COLOURS["row_odd"])

        self.tree.bind("<<TreeviewSelect>>", self._on_row_select)

        # ── Right: detail panel ──────────────────────────────────────────────
        right = tk.Frame(main, bg=COLOURS["panel"], width=260, padx=12, pady=12)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        tk.Label(
            right, text="Document Detail",
            font=FONT_HEAD, bg=COLOURS["panel"], fg=COLOURS["accent"],
        ).pack(anchor="w", pady=(0, 8))

        self.detail_text = scrolledtext.ScrolledText(
            right,
            font=FONT_SMALL,
            bg=COLOURS["entry_bg"], fg=COLOURS["text"],
            insertbackground=COLOURS["text"],
            relief="flat", wrap="word",
            width=30, height=30,
        )
        self.detail_text.pack(fill="both", expand=True)
        self.detail_text.insert("1.0", "Select a result to see details.")
        self.detail_text.config(state="disabled")

    # =========================================================================
    #  Status bar
    # =========================================================================

    def _build_status_bar(self) -> None:
        self.status_var = tk.StringVar(value="Ready")
        bar = tk.Frame(self, bg=COLOURS["panel"], pady=4)
        bar.pack(fill="x", side="bottom")
        tk.Label(
            bar, textvariable=self.status_var,
            font=FONT_SMALL, bg=COLOURS["panel"], fg=COLOURS["subtext"],
            anchor="w",
        ).pack(side="left", padx=12)

    def _update_status(self, msg: str) -> None:
        self.status_var.set(msg)

    # =========================================================================
    #  Search logic
    # =========================================================================

    def _run_search(self) -> None:
        query = self.query_var.get().strip()
        if not query:
            return

        # Update retriever parameters from sliders
        self.retriever.alpha  = self.alpha_var.get()
        self.retriever.top_k  = self.topk_var.get()

        self.search_btn.config(state="disabled", text="Searching …")
        self._update_status(f"Searching for: '{query}' …")

        def _worker():
            results = self.retriever.search(query)
            self.after(0, lambda: self._display_results(query, results))

        threading.Thread(target=_worker, daemon=True).start()

    def _display_results(self, query: str, results: list[dict]) -> None:
        # Clear old rows
        for row in self.tree.get_children():
            self.tree.delete(row)

        self.search_btn.config(state="normal", text="  Search  ")

        if not results:
            self._update_status(
                f"No results above α={self.retriever.alpha:.4f} for: '{query}'"
            )
            return

        for r in results:
            tag = "even" if r["rank"] % 2 == 0 else "odd"
            terms_str = ", ".join(r["query_terms"][:8])
            self.tree.insert(
                "", "end",
                iid=str(r["doc_id"]),
                values=(
                    r["rank"],
                    f"{r['score']:.6f}",
                    r["doc_name"],
                    terms_str,
                ),
                tags=(tag,),
            )

        self._update_status(
            f"Found {len(results)} result(s) for '{query}'  "
            f"(α={self.retriever.alpha:.4f})"
        )

        # Add to history
        if query not in self.query_history:
            self.query_history.insert(0, query)
            self.hist_cb["values"] = self.query_history

    # =========================================================================
    #  Row-select detail panel
    # =========================================================================

    def _on_row_select(self, _event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        doc_id    = int(sel[0])
        doc_name  = self.retriever.indexer.get_doc_name(doc_id)
        doc_vec   = self.retriever.indexer.get_tfidf_vector(doc_id)
        doc_norm  = self.retriever.indexer.get_doc_norm(doc_id)

        # Top-10 terms by weight
        top_terms = sorted(doc_vec.items(), key=lambda x: x[1], reverse=True)[:10]

        text = (
            f"Document:  {doc_name}\n"
            f"Doc ID  :  {doc_id}\n"
            f"L2 Norm :  {doc_norm:.6f}\n"
            f"Unique terms: {len(doc_vec):,}\n\n"
            f"Top-10 TF-IDF Terms\n"
            f"{'─'*28}\n"
        )
        for term, w in top_terms:
            bar = "█" * int(w * 4)
            text += f"  {term:<16} {w:>7.4f}  {bar}\n"

        self.detail_text.config(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("1.0", text)
        self.detail_text.config(state="disabled")

    # =========================================================================
    #  Helpers
    # =========================================================================

    def _clear(self) -> None:
        self.query_var.set("")
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.detail_text.config(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("1.0", "Select a result to see details.")
        self.detail_text.config(state="disabled")
        self._update_status("Cleared.")
        self.entry.focus_set()

    def _load_history(self, _event=None) -> None:
        self.query_var.set(self.hist_var.get())
        self.entry.focus_set()


# =============================================================================
#  Public launcher (called from main.py)
# =============================================================================

def launch_gui(retriever: VSMRetriever) -> None:
    """Create and run the Tkinter application."""
    app = VSMApp(retriever)
    app.mainloop()


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from preprocessing import Preprocessor
    from indexer        import Indexer
    from vsm_retrieval  import VSMRetriever

    prep  = Preprocessor()
    idx   = Indexer(prep)
    # Attempt to load a saved index for quick testing
    if not idx.load("vsm_index.pkl"):
        print("[gui] No saved index found. Build one with main.py --rebuild first.")
        sys.exit(1)
    retr = VSMRetriever(idx, prep)
    launch_gui(retr)