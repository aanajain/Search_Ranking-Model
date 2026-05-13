import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi

def ndcg_at_k(relevances, k=10):
    relevances = np.array(relevances[:k], dtype=float)
    dcg = np.sum(relevances / np.log2(np.arange(2, len(relevances)+2)))
    ideal = np.sort(relevances)[::-1]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal)+2)))
    return dcg / idcg if idcg > 0 else 0.0

def bm25_score(df):
    scores = []
    for qid, group in df.groupby("qid"):
        fake_docs = [
            ["term"] * int(row["term_overlap"] * 10)
            for _, row in group.iterrows()
        ]
        query_tokens = group.iloc[0]["query"].split()
        bm25 = BM25Okapi(fake_docs)
        bm25_scores = bm25.get_scores(query_tokens)
        ranked_idx = np.argsort(bm25_scores)[::-1]
        ranked_labels = group["label"].values[ranked_idx]
        scores.append(ndcg_at_k(ranked_labels))
    return np.mean(scores)

df = pd.read_csv("query_docs.csv")
bm25_ndcg = bm25_score(df)
print(f"BM25 NDCG@10: {bm25_ndcg:.4f}")
