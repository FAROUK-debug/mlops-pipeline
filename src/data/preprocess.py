"""DVC stage 1: data/dataset.csv -> data/processed/clean.csv, using params.yaml > preprocess.
The app calls clean() too, so what you try in the app is exactly what the pipeline runs."""
import pandas as pd
from sklearn.impute import KNNImputer

from src import ROOT, TARGET, load_params

MISSING = ["median", "mean", "mode", "knn", "ffill", "drop"]
ENCODING = ["one-hot", "ordinal", "frequency", "none"]


def clean(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = df.drop(columns=[c for c in p.get("drop", []) if c != TARGET]).copy()
    num = [c for c in df.select_dtypes("number") if c != TARGET]
    cat = [c for c in df.columns if c not in num and c != TARGET]

    how = p["missing"]
    if how == "drop":
        df = df.dropna()
    elif how == "ffill":
        df = df.ffill().bfill()
    else:
        if num and how == "knn":
            df[num] = KNNImputer().fit_transform(df[num])
        elif num:
            fill = {"mean": df[num].mean(), "median": df[num].median(), "mode": df[num].mode().iloc[0]}[how]
            df[num] = df[num].fillna(fill)
        if cat:  # text columns can only take the most frequent value
            df[cat] = df[cat].fillna(df[cat].mode().iloc[0])

    enc = p["encoding"]
    if enc == "one-hot":
        df = pd.get_dummies(df, columns=cat, dtype=int)
    elif enc == "ordinal":
        for c in cat:
            df[c] = df[c].astype("category").cat.codes
    elif enc == "frequency":
        for c in cat:
            df[c] = df[c].map(df[c].value_counts(normalize=True)).round(4)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    df = clean(pd.read_csv(ROOT / "data" / "dataset.csv"), load_params()["preprocess"])
    df.to_csv(out / "clean.csv", index=False)
    print("clean.csv", df.shape)
