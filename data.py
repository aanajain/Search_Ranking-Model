import pandas as pd
import numpy as np
import random

def make_dataset(n_queries=200, docs_per_query=10):
    topics = [
        "machine learning", "python tutorial",
        "search engine", "data science", "neural network"
    ]
    rows = []
    for qid in range(n_queries):
        topic = random.choice(topics)
        query = topic + " " + random.choice(["basics", "advanced", "guide", "example"])
        for did in range(docs_per_query):
            label = np.random.choice([0,1,2], p=[0.5,0.35,0.15])
            overlap = label * random.uniform(0.3, 1.0)
            doc_len = random.randint(50, 500)
            rows.append({
                "qid": qid, "query": query, "doc_id": did,
                "label": label, "term_overlap": overlap,
                "doc_length": doc_len,
                "title_match": int(label > 0 and random.random() > 0.4),
                "click_rate": label * random.uniform(0, 0.3),
            })
    return pd.DataFrame(rows)

df = make_dataset()
df.to_csv("query_docs.csv", index=False)
print(f"Done! {len(df)} rows saved to query_docs.csv")
print(df.head())
