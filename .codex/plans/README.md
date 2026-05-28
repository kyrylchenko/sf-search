# Plans

Write implementation plans here before making code changes.

Use short, descriptive filenames, preferably:

```text
YYYY-MM-DD-topic.md
```

Plans should state the goal, affected files, intended steps, verification, known risks, and why the chosen approach was taken.

Git rule for every plan: local commits are allowed, but remote/origin operations are forbidden. Do not plan or run `git push`, `git pull`, `git fetch`, or any command targeting `origin`.

Commit rule for every plan: commit coherent local changes as work progresses, and use Conventional Commits format, for example `test: add ingestion baseline` or `feat: persist discovered panoramas`.
