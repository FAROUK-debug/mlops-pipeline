# Heart disease MLOps pipeline

An end-to-end MLOps lab project: a heart-disease dataset, a Streamlit console to explore and clean it, a DVC pipeline (clean â†’ split â†’ train), and dataset versions saved from the app with DVC + Git tags.

The 3 presentation slides from S1 are in [docs/S1_slides.pptx](docs/S1_slides.pptx), with speaker notes.

## 1. The problem
**Goal:** predict whether a patient has heart disease (yes / no) from 13 routine clinical measurements, such as age, chest-pain type, blood pressure, cholesterol and maximum heart rate.

**Why it matters:** heart disease is the leading cause of death worldwide. A model that flags at-risk patients from cheap measurements helps doctors decide who needs further tests first. For that use, **recall** (how many sick patients the model catches) matters more than accuracy.

## 2. The dataset
| | |
|---|---|
| Source | [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease), Cleveland clinic |
| Type | tabular (CSV) |
| Size | 303 patients, 13 features + target |
| Features | 6 numeric (age, trestbps, chol, thalach, oldpeak, ca) and 7 categorical (sex, cp, fbs, restecg, exang, slope, thal) |
| Target | `target`: 0 = no disease (54%), 1 = disease (46%) |
| Missing values | `ca` (4), `thal` (2) |

`python -m src.data.get_data` rebuilds version v1.0 from UCI. It turns the coded categories into words (for example, `cp` 4 becomes `asymptomatic`) so the encoding step works on real text columns.

## 3. Why this dataset, and next steps
- It's small and well known, with real missing values and real categorical columns. That makes every cleaning choice in the app visible.
- **Preprocessing:** imputation (median, mean, mode, KNN, forward-fill or drop rows), encoding (one-hot, ordinal or frequency), choice of columns, and a stratified train/test split.
- **Model:** logistic regression as the baseline. Random forest or gradient boosting come next.
- **Metrics:** accuracy, F1 and recall.

## Project layout
```
app/app.py               Streamlit console (S1 dashboard + S3 versioning)
src/data/get_data.py     builds data/dataset.csv (v1.0) from UCI
src/data/preprocess.py   DVC stage 1: clean()        (also used by the app)
src/data/split.py        DVC stage 2: split()        (also used by the app)
src/train.py             DVC stage 3: train(), writes models/model.pkl + metrics.json
src/versioning.py        version_data(): dvc add â†’ git commit â†’ git tag â†’ dvc push â†’ git push
params.yaml              every parameter of the pipeline
dvc.yaml / dvc.lock      the pipeline and the exact state of its last run
data/dataset.csv.dvc     Git's pointer to the data (the CSV itself lives in DVC)
tests/test_pipeline.py   every cleaning choice must give trainable data
ANSWERS.md               answers to the S3 lab questions Q1â€“Q4
```

## Run it
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
dvc pull                        # needs access to the DVC remote; otherwise: python -m src.data.get_data
streamlit run app/app.py
```
- `dvc repro` runs the pipeline. `dvc metrics show` shows the scores.
- `python -m pytest tests` runs the checks.

## The app
1. **Data:** shape, the first rows, column types and missing values.
2. **Charts:** a histogram per column (coloured by target), a correlation heatmap and some takeaways.
3. **Cleaning impact:** the parameters in the sidebar apply live: rows, columns, missing values and the train/test size.
4. **Before / after:** both side by side, plus a download of the cleaned CSV. The raw file is never changed.
5. **Model:** train a logistic regression and see accuracy, F1 and recall.
6. **Versions:**
   - lists every saved version and the parameters that made it;
   - shows a params.yaml preview and **runs the DVC pipeline**;
   - **Save as new version** with a tag and commit message.

The sidebar can load any past version (`v1.0`, `v1.1`, â€¦) straight from DVC.

## Dataset versions
| Tag | How it was made |
|---|---|
| v1.0 | raw UCI data |
| v1.1 | median imputation (no other change) |
| v1.2 | v1.1 + one-hot encoding |
| v1.3 | v1.2 without `chol` and `fbs` (the columns least linked to the target) |

`git tag -n` lists the versions with the parameters that produced each one.
