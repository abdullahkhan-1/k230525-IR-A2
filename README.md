# VSM Information Retrieval System
### Trump Speeches Corpus — BSCS / IR Assignment

---

## Project Structure

```
vsm_ir/
├── preprocessing.py   # Tokenization, stop-word removal, lemmatization
├── indexer.py         # Inverted index, TF-IDF computation, persistence
├── vsm_retrieval.py   # Cosine similarity, ranking, batch evaluation
├── gui.py             # Tkinter graphical interface
├── main.py            # CLI entry point
├── requirements.txt
└── README.md

data/                  # ← Place your corpus files here
├── TrumpSpeeches/     # 56 speech .txt files
├── stopwords.txt      # Provided stop-word list
└── queries.txt        # 10 evaluation queries
```

---

## Setup

```bash
# 1. Install Python dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Place the corpus, stopwords, and queries inside data/
```

---

## Running the System

### Build index + run all 10 evaluation queries
```bash
python main.py \
    --corpus     data/TrumpSpeeches \
    --stopwords  data/stopwords.txt \
    --queries    data/queries.txt   \
    --rebuild
```

### Interactive CLI search
```bash
python main.py \
    --corpus     data/TrumpSpeeches \
    --stopwords  data/stopwords.txt \
    --interactive
```

### Graphical Interface (Tkinter)
```bash
python main.py \
    --corpus     data/TrumpSpeeches \
    --stopwords  data/stopwords.txt \
    --gui
```

### Re-use cached index (skip rebuild)
```bash
python main.py \
    --corpus data/TrumpSpeeches \
    --interactive
```

---

## CLI Options

| Option          | Default  | Description                                        |
|-----------------|----------|----------------------------------------------------|
| `--corpus`      | required | Path to directory of .txt speech files             |
| `--stopwords`   | None     | Path to stop-word list file                        |
| `--queries`     | None     | Path to queries file (batch evaluation)            |
| `--rebuild`     | False    | Force full index rebuild                           |
| `--interactive` | False    | Start interactive CLI search loop                  |
| `--gui`         | False    | Launch Tkinter GUI                                 |
| `--alpha`       | 0.005    | Minimum cosine similarity threshold                |
| `--top_k`       | 10       | Max results returned per query                     |
| `--min_tf`      | 1        | Min raw term-frequency per document                |
| `--min_df`      | 2        | Min document-frequency (feature selection)         |

---

## Implementation Details

### Preprocessing Pipeline
1. **Tokenization** — regex `[a-z]+(?:'[a-z]+)*`; extracts alphabetic tokens
2. **Case folding** — everything lowercased before tokenization
3. **Stop-word removal** — filters against provided stop-word list
4. **Lemmatization** — NLTK `WordNetLemmatizer` (verb-form priority)

### Index Structure
- **Inverted index** — `dict[term → dict[doc_id → raw_tf]]`
- **Feature selection** — `min_tf` and `min_df` thresholds prune low-signal terms
- **Persistence** — serialised with `pickle`; auto-loaded on next run

### TF-IDF Weighting
```
tf_weight = 1 + log10(raw_tf)       [log-normalised TF]
idf       = log10(N / df)           [standard IDF]
tf_idf    = tf_weight × idf
```

### Cosine Similarity (Query Processing)
```
sim(q, d) = (q · d) / (|q| × |d|)
```
- Query vector built with the same IDF values as the corpus
- Dot product accumulated via the inverted index (efficient sparse computation)
- Results filtered by `alpha` threshold, then ranked descending

---

## Grading Checklist

| Criterion                          | Implementation                              |
|------------------------------------|---------------------------------------------|
| Preprocessing (3 marks)            | `preprocessing.py` — all 4 steps           |
| Index formation (2 marks)          | `indexer.py` — save/load + complexity       |
| Simple VSM Queries (2 marks)       | Single-term & two-term queries              |
| Complex VSM Queries (2 marks)      | Multi-term, phrase-like queries             |
| Code Complexity (1 mark)           | Sparse accumulator, L2-norm caching         |
| GUI Bonus (2 marks)                | `gui.py` — Tkinter dark-theme interface     |
| Clean & commented code (+5%)       | Docstrings on every class and method        |
