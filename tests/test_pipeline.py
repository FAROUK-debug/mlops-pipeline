"""Every cleaning choice must give a dataset the model can train on. Run: python -m pytest tests (or python -m tests.test_pipeline)."""
import pandas as pd

from src import TARGET
from src.data.preprocess import ENCODING, MISSING, clean
from src.data.split import split
from src.train import train

from src.versioning import load_version

RAW = load_version("v1.0")  # the raw data, whatever version is checked out (needs dvc pull)


def test_every_combination_trains():
    for missing in MISSING:
        for encoding in ENCODING:
            df = clean(RAW, {"drop": ["chol"], "missing": missing, "encoding": encoding})
            assert df.isna().sum().sum() == 0, (missing, encoding)
            assert "chol" not in df and TARGET in df
            if encoding != "none":
                assert all(pd.api.types.is_numeric_dtype(df[c]) for c in df), (missing, encoding)
            tr, te = split(df, {"test_size": 0.2, "seed": 0})
            _, metrics = train(tr, te, {"C": 1.0, "max_iter": 1000})
            assert metrics["accuracy"] > 0.6, (missing, encoding, metrics)


def test_raw_is_untouched():
    before = RAW.copy()
    clean(RAW, {"drop": [], "missing": "knn", "encoding": "one-hot"})
    pd.testing.assert_frame_equal(RAW, before)


if __name__ == "__main__":
    test_every_combination_trains()
    test_raw_is_untouched()
    print("ok")
