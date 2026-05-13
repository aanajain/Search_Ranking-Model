import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_csv("query_docs.csv")

FEATURES = ["term_overlap", "doc_length", "title_match", "click_rate"]
X = df[FEATURES].values
y = df["label"].values
groups = df["qid"].values

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
g_train = df.iloc[train_idx].groupby("qid").size().values
g_test  = df.iloc[test_idx].groupby("qid").size().values

train_data = lgb.Dataset(X_train, label=y_train, group=g_train)
test_data  = lgb.Dataset(X_test,  label=y_test,  group=g_test)

params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10],
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "verbose": -1,
}

model = lgb.train(
    params, train_data,
    num_boost_round=200,
    valid_sets=[test_data],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)]
)

model.save_model("ranker.lgb")
print("\nModel saved → ranker.lgb")
