# Publishing this project

The trained project is complete locally. Public GitHub repository creation was
blocked by unavailable browser authentication in the execution session; no remote
repository or GitHub Actions run is claimed.

From an authenticated local GitHub CLI, after unpacking the source archive:

```sh
git init -b main
git add .
git commit -m "Train and evaluate tiny destination-conditioned chess ranker"
gh repo create tiny-chess-input --public --source=. --remote=origin --push
```

Alternatively, create an empty public repository in GitHub, then push this project
there. The included workflow runs inference/rules tests; it does not download the
training corpus or rerun training on every push.
