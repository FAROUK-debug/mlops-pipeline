"""Heart disease data console: explore, clean, train, and save dataset versions with DVC.
Run from anywhere:  streamlit run app/app.py"""
import os
import subprocess
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so "src" imports work from any folder
from src import ROOT, TARGET, load_params  # noqa: E402
from src.data.preprocess import ENCODING, MISSING, clean  # noqa: E402
from src.data.split import split  # noqa: E402
from src.train import train  # noqa: E402
from src.versioning import list_versions, load_version, version_data  # noqa: E402

st.set_page_config(page_title="Heart Disease MLOps", page_icon="🫀", layout="wide")
st.title("🫀 Heart disease · data console")


@st.cache_data
def cached_version(tag):
    return load_version(tag)


# ── sidebar: which data, and the parameters ──────────────────────────────────
versions = list_versions()
WORKING = "working copy (data/dataset.csv)"
choice = st.sidebar.selectbox("Dataset version", [WORKING] + [t for t, _ in versions])
raw = cached_version(None if choice == WORKING else choice)

saved = load_params()
st.sidebar.header("Parameters")
p = {
    "preprocess": {
        "drop": st.sidebar.multiselect("Columns to drop", [c for c in raw.columns if c != TARGET],
                                       [c for c in saved["preprocess"]["drop"] if c in raw.columns]),
        "missing": st.sidebar.selectbox("Missing values", MISSING, MISSING.index(saved["preprocess"]["missing"])),
        "encoding": st.sidebar.selectbox("Encoding", ENCODING, ENCODING.index(saved["preprocess"]["encoding"])),
    },
    "split": {
        "test_size": st.sidebar.slider("Test size", 0.1, 0.5, float(saved["split"]["test_size"]), 0.05),
        "seed": int(st.sidebar.number_input("Random seed", 0, 9999, int(saved["split"]["seed"]))),
    },
    "train": saved["train"],
}
cleaned = clean(raw, p["preprocess"])
train_df, test_df = split(cleaned, p["split"])

tabs = st.tabs(["1 · Data", "2 · Charts", "3 · Cleaning impact", "4 · Before / after", "5 · Model", "6 · Versions"])

# ── 1. load and preview ──────────────────────────────────────────────────────
with tabs[0]:
    a, b, c = st.columns(3)
    a.metric("Rows", raw.shape[0])
    b.metric("Columns", raw.shape[1])
    c.metric("Missing cells", int(raw.isna().sum().sum()))
    st.dataframe(raw.head(10), use_container_width=True)
    st.dataframe(pd.DataFrame({"type": raw.dtypes.astype(str), "missing": raw.isna().sum()}), use_container_width=True)

# ── 2. visualise ─────────────────────────────────────────────────────────────
with tabs[1]:
    col = st.selectbox("Column", raw.columns)
    x = alt.X(col, bin=alt.Bin(maxbins=30)) if pd.api.types.is_numeric_dtype(raw[col]) and raw[col].nunique() > 10 \
        else alt.X(f"{col}:N")
    st.altair_chart(alt.Chart(raw).mark_bar().encode(x, y="count()", color=f"{TARGET}:N"), use_container_width=True)

    corr = raw.corr(numeric_only=True).round(2).reset_index().melt("index")
    st.altair_chart(alt.Chart(corr).mark_rect().encode(
        x="index:N", y="variable:N", color=alt.Color("value:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
        tooltip=["index", "variable", "value"]).properties(title="Correlation (numeric columns)", height=400),
        use_container_width=True)

    missing = raw.isna().sum()
    st.markdown(f"""**Takeaways**
- {raw[TARGET].mean():.0%} of patients have heart disease, so the classes are close to balanced.
- Missing values: {', '.join(f'`{c}` ({n})' for c, n in missing[missing > 0].items()) or 'none'}.
- Strongest link to the target: `{raw.corr(numeric_only=True)[TARGET].drop(TARGET).abs().idxmax()}`.""")

# ── 3. live impact of the parameters ─────────────────────────────────────────
with tabs[2]:
    a, b, c, d = st.columns(4)
    a.metric("Rows", cleaned.shape[0], cleaned.shape[0] - raw.shape[0])
    b.metric("Columns", cleaned.shape[1], cleaned.shape[1] - raw.shape[1])
    c.metric("Missing cells", int(cleaned.isna().sum().sum()), -int(raw.isna().sum().sum()))
    d.metric("Train / test", f"{len(train_df)} / {len(test_df)}")
    st.dataframe(cleaned.head(10), use_container_width=True)

# ── 4. before vs after, download ─────────────────────────────────────────────
with tabs[3]:
    left, right = st.columns(2)
    left.subheader("Before")
    left.dataframe(raw.head(15), use_container_width=True)
    right.subheader("After")
    right.dataframe(cleaned.head(15), use_container_width=True)
    st.download_button("Download cleaned CSV", cleaned.to_csv(index=False), "cleaned_dataset.csv", "text/csv")
    st.caption("The raw file is never changed here. Only “Save as new version” (tab 6) writes data/dataset.csv.")

# ── 5. a simple model ────────────────────────────────────────────────────────
with tabs[4]:
    st.write("Logistic regression on the train split, scored on the test split.")
    if st.button("Train model"):
        _, metrics = train(train_df, test_df, p["train"])
        cols = st.columns(len(metrics))
        for column, (name, value) in zip(cols, metrics.items()):
            column.metric(name, f"{value:.2%}")

# ── 6. versions (DVC + Git) ──────────────────────────────────────────────────
with tabs[5]:
    if "saved" in st.session_state:
        st.success(st.session_state.pop("saved"))
    st.subheader("Saved versions")
    st.dataframe(pd.DataFrame(versions, columns=["tag", "parameters"]), use_container_width=True, hide_index=True)

    st.subheader("Parameters as params.yaml")
    st.code(yaml.safe_dump(p, sort_keys=False), language="yaml")
    if st.button("Save params.yaml and run the DVC pipeline"):
        (ROOT / "params.yaml").write_text(yaml.safe_dump(p, sort_keys=False))
        env = {**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]}
        result = subprocess.run([sys.executable, "-m", "dvc", "repro"], cwd=ROOT, env=env, capture_output=True, text=True)
        (st.success if result.returncode == 0 else st.error)("Pipeline finished" if result.returncode == 0 else "Pipeline failed")
        st.code(result.stdout + result.stderr)

    st.subheader("Save as new version")
    with st.form("version"):
        tag = st.text_input("Version tag", placeholder="v1.1")
        message = st.text_input("Commit message", placeholder="median imputation, one-hot encoding")
        if st.form_submit_button("Save as new version"):
            try:
                version_data(cleaned, tag.strip(), message, {"from": choice, **p["preprocess"], **p["split"]})
                cached_version.clear()
                st.session_state["saved"] = f"Saved {tag}: data in DVC, pointer committed in Git, tag created."
                st.rerun()  # reload so the new tag shows in the lists
            except subprocess.CalledProcessError as e:
                st.error(f"`{' '.join(e.cmd)}` failed:\n\n{e.stderr}")
            except ValueError as e:
                st.error(str(e))
