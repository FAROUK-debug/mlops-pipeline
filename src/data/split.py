"""DVC stage 2: clean.csv -> train.csv + test.csv, using params.yaml > split."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src import ROOT, TARGET, load_params


def split(df: pd.DataFrame, p: dict):
    # stratify keeps the same share of sick patients in train and test
    return train_test_split(df, test_size=p["test_size"], random_state=p["seed"], stratify=df[TARGET])


if __name__ == "__main__":
    folder = ROOT / "data" / "processed"
    train, test = split(pd.read_csv(folder / "clean.csv"), load_params()["split"])
    train.to_csv(folder / "train.csv", index=False)
    test.to_csv(folder / "test.csv", index=False)
    print("train", train.shape, "test", test.shape)
