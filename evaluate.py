import lightgbm as lgb
import pandas as pd
import numpy as np
from baseline import ndcg_at_k, bm25_score

df = pd.read_csv("query_docs.csv")
model = lgb.Booster(model_file="ranker.lgb")
FEATURES = ["term_overlap", "doc_length", "title_match", "click_rate"]

def lambdamart_score(df, model):
    scores = []
    for qid, group in df.groupby("qid"):
        X = group[FEATURES].values
        preds = model.predict(X)
        ranked_idx = np.argsort(preds)[::-1]
        ranked_labels = group["label"].values[ranked_idx]
        scores.append(ndcg_at_k(ranked_labels, k=10))
    return np.mean(scores)

bm25_ndcg  = bm25_score(df)
model_ndcg = lambdamart_score(df, model)
improvement = (model_ndcg - bm25_ndcg) / bm25_ndcg * 100

print(f"\n{'='*40}")
print(f"  BM25 NDCG@10      : {bm25_ndcg:.4f}")
print(f"  LambdaMART NDCG@10: {model_ndcg:.4f}")
print(f"  Improvement       : +{improvement:.1f}%")
print(f"{'='*40}\n")
