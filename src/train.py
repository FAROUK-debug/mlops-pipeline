"""DVC stage 3: train a logistic regression on train.csv, score it on test.csv."""
import json
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import ROOT, TARGET, load_params


def train(train_df: pd.DataFrame, test_df: pd.DataFrame, p: dict):
    # one-hot whatever text is left (encoding "none"), with the same columns for train and test
    X_train = pd.get_dummies(train_df.drop(columns=TARGET), dtype=int)
    X_test = pd.get_dummies(test_df.drop(columns=TARGET), dtype=int).reindex(columns=X_train.columns, fill_value=0)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=p["max_iter"]))
    model.fit(X_train, train_df[TARGET])
    pred = model.predict(X_test)
    y = test_df[TARGET]
    metrics = {
        "accuracy": round(accuracy_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),  # share of sick patients the model catches
    }
    return model, metrics


if __name__ == "__main__":
    folder = ROOT / "data" / "processed"
    model, metrics = train(pd.read_csv(folder / "train.csv"), pd.read_csv(folder / "test.csv"), load_params()["train"])
    (ROOT / "models").mkdir(exist_ok=True)
    with open(ROOT / "models" / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(metrics)
