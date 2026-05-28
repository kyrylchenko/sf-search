# Specs

Write design specs here for larger or ambiguous work before implementation planning.

Use short, descriptive filenames, preferably:

```text
YYYY-MM-DD-topic.md
```

Specs should capture the problem, constraints, proposed architecture, data flow, interfaces, open questions, and the reasoning behind the design.

Git rule for every spec: this repo is local-only. Do not design workflows that require `git push`, `git pull`, `git fetch`, or any origin/remote operation.

Commit rule for every spec: assume implementation work will be committed locally in coherent steps using Conventional Commits format.

Public repo rule for every spec: never include real API keys, tokens, credentials, private endpoints, private filesystem paths, personal names, private organization names, or generated private artifacts. Use placeholders in docs and tests.
