# Search Ranking Model — Learning to Rank with LambdaMART

A machine learning pipeline that ranks search results by relevance using a **LambdaMART** model, outperforming the classic BM25 baseline by **43% on NDCG@10**.

---

## Results

| Model | NDCG@10 |
|---|---|
| BM25 (baseline) | 0.6948 |
| LambdaMART (ours) | 0.9949 |
| **Improvement** | **+43.2%** |

---

## What is Learning to Rank?

Search engines don't just find documents — they *rank* them. Given a query like `"machine learning basics"` and 10 candidate documents, the goal is to put the most relevant ones at the top.

**BM25** does this using keyword frequency (old school). **LambdaMART** learns from features like term overlap, click rates, and title matches — and directly optimizes the ranking metric (NDCG).

---

## Project Structure
---

## Quickstart

**1. Clone and set up environment**
```bash
git clone https://github.com/aanajain/Search_Ranking-Model
cd Search_Ranking-Model
python3 -m venv venv
source venv/bin/activate
pip install lightgbm rank-bm25 scikit-learn numpy pandas
```

> Mac users: also run `brew install libomp`

**2. Generate dataset**
```bash
python data.py
```

**3. Run BM25 baseline**
```bash
python baseline.py
```

**4. Train LambdaMART**
```bash
python train.py
```

**5. Evaluate**
```bash
python evaluate.py
```

---

## Features Used

| Feature | Description |
|---|---|
| `term_overlap` | How many query terms appear in the document |
| `doc_length` | Number of tokens in the document |
| `title_match` | Whether the query appears in the document title |
| `click_rate` | Historical click-through rate for this query-doc pair |

---

## Tech Stack

- **LightGBM** — gradient boosting with `lambdarank` objective
- **rank-bm25** — BM25 baseline implementation
- **scikit-learn** — group-aware train/test splitting
- **pandas / numpy** — data processing

---

## Resume Bullet

> Built a learning-to-rank model using LambdaMART (LightGBM) on 2,000 query-document pairs; improved NDCG@10 by 43% over BM25 baseline using features including term overlap, click rate, and title match signals.

---

## References

- [LambdaMART paper](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/)
- [LightGBM LambdaRank docs](https://lightgbm.readthedocs.io/en/stable/Parameters.html#objective)
- [MSLR Dataset — Microsoft Research](https://www.microsoft.com/en-us/research/project/mslr/)
