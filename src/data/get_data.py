"""Builds data/dataset.csv (version v1.0) from the UCI Heart Disease dataset (Cleveland, id 45).

The UCI file stores categories as numbers; we turn them into words so the encoding step in the
app has real text columns to work on. The target becomes 0 (no disease) / 1 (disease).
"""
import pandas as pd

from src import ROOT

URL = "https://archive.ics.uci.edu/static/public/45/data.csv"
LABELS = {
    "sex": {0: "female", 1: "male"},
    "cp": {1: "typical", 2: "atypical", 3: "non-anginal", 4: "asymptomatic"},
    "fbs": {0: "no", 1: "yes"},
    "restecg": {0: "normal", 1: "st-t", 2: "lvh"},
    "exang": {0: "no", 1: "yes"},
    "slope": {1: "up", 2: "flat", 3: "down"},
    "thal": {3: "normal", 6: "fixed", 7: "reversible"},
}

if __name__ == "__main__":
    df = pd.read_csv(URL)
    for column, names in LABELS.items():
        df[column] = df[column].map(names)
    df["target"] = (df.pop("num") > 0).astype(int)
    df.to_csv(ROOT / "data" / "dataset.csv", index=False)
    print(df.shape, "missing:", df.isna().sum()[lambda s: s > 0].to_dict())
