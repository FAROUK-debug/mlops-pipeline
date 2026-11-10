# Lab S3: answers

**Q1. Which files does `dvc init` create, and why commit them?**
`dvc init` creates the `.dvc/` folder and a `.dvcignore` file:
- `.dvc/config` holds the project's DVC settings, including the default remote (`localremote`).
- `.dvc/.gitignore` keeps the cache, temp files and `config.local` out of Git.
- `.dvcignore` lists files DVC must not track. It works like `.gitignore`.

They go into Git so everyone who clones the repo gets the same DVC setup and remote. Without them, `dvc pull` would not know where the data lives.

**Q2. What's inside `dataset.csv.dvc`, and why does Git ignore the CSV?**
It's a small text "pointer" file:
```yaml
outs:
- md5: 9dae48e301923a1577850b70e5a1d682   # fingerprint of the file's exact content
  size: 20605
  hash: md5
  path: dataset.csv
```
`dvc add` writes `dataset.csv` into `data/.gitignore`, so Git only versions the pointer. Big or frequently changing data would make the Git history heavy and slow, and Git can't diff binary or large files well. A new version of the data changes only the md5 line, so each Git commit or tag points to one exact version of the data.

**Q3. Where is the content stored?**
- **DVC cache:** `.dvc/cache/files/md5/<first 2 characters of the md5>/<the rest>`. Every version is stored once, named by its md5. Here that is `.dvc/cache/files/md5/9d/ae48e3…` for v1.0.
- **Remote:** `dvc push` copies the same files into the remote. Ours is a local folder outside the repo, `../dvc-storage/mlops-pipeline/files/md5/…`. It could be Google Drive, S3 or Azure the same way. `dvc pull` (or `dvc.api.read(..., rev="v1.1")` in the app) brings a version back from the cache or the remote.

**Q4. Why is the code in Git but the data in DVC?**
They are different kinds of files:
- Code is small text that people edit line by line, so Git's diffs, branches and reviews suit it.
- Data is large, often binary, and replaced as a whole. In Git it would bloat every clone forever.

DVC keeps the heavy files in a cache or remote, and Git keeps the small pointers. So one Git commit (or tag such as `v1.2`) still pins both the exact code and the exact data. That is what makes an experiment reproducible.
