"""S3: save a dataset version from the app. Data -> DVC, metadata -> Git, a readable tag (v1.1 ...).
The tag's message holds the parameters that produced the version, so every version says how it was made."""
import io
import json
import re
import subprocess
import sys

from dvc.api import read as dvc_read
import pandas as pd

from src import ROOT

DATA = "data/dataset.csv"


def run(*cmd: str) -> str:
    return subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True).stdout


def dvc(*args: str) -> str:
    return run(sys.executable, "-m", "dvc", *args)


def list_versions() -> list[tuple[str, str]]:
    """[(tag, parameters)], newest first."""
    out = run("git", "tag", "-l", "v*", "--sort=-creatordate", "--format=%(refname:short)\t%(contents:subject)")
    return [tuple(line.split("\t", 1)) for line in out.splitlines() if "\t" in line]


def load_version(tag: str | None) -> pd.DataFrame:
    """A past version from DVC (cache or remote), or the working copy when tag is None."""
    if tag is None:
        return pd.read_csv(ROOT / DATA)
    return pd.read_csv(io.StringIO(dvc_read(DATA, repo=str(ROOT), rev=tag)))


def version_data(df: pd.DataFrame, tag: str, message: str, params: dict) -> None:
    if not re.fullmatch(r"v\d+(\.\d+)*", tag):
        raise ValueError("The tag must look like v1.2")
    if tag in dict(list_versions()):
        raise ValueError(f"{tag} already exists")
    if not message.strip():
        raise ValueError("Write a commit message")

    df.to_csv(ROOT / DATA, index=False)
    dvc("add", DATA)
    files = ["data/dataset.csv.dvc", "data/.gitignore"]
    run("git", "add", *files)
    run("git", "commit", "-m", message, "--", *files)  # commits only the data pointer, never other work
    run("git", "tag", "-a", tag, "-m", json.dumps(params))
    dvc("push")
    if run("git", "remote").strip():
        run("git", "push", "--follow-tags")


if __name__ == "__main__":
    # python -m src.versioning v1.4 "message": version the current data/dataset.csv by hand
    version_data(pd.read_csv(ROOT / DATA), sys.argv[1], sys.argv[2], {"source": "command line"})
