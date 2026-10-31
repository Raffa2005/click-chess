# Repository

Public source: https://github.com/Raffa2005/click-chess

The repository contains deployment models, Python and JavaScript inference,
reproducible training/data preparation, experiment records, and held-out results.
Raw game archives and processed training arrays are excluded from git.

GitHub Actions runs rules, export, and Python/JavaScript parity tests on pushes
and pull requests. It does not download the training corpus or retrain models.

To publish a later local change from an authenticated checkout:

```sh
git add .
git commit -m "Describe the verified change"
git push origin main
```
