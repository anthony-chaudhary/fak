# Decisions audit log

`internal/witness` writes adjudication decisions as JSON-lines git notes on the
dedicated side ref:

```bash
git notes --ref=refs/notes/fak/decisions show <commit-sha>
```

Pre-commit refusals have no commit to anchor to, so they are recorded on git's
well-known empty-tree object:

```bash
git notes --ref=refs/notes/fak/decisions show 4b825dc642cb6eb9a060e54bf8d69288fbee4904
```

The ref is intentionally a notes ref, not a branch. It does not rewrite or move
`main`, does not change `git log main`, and is not fetched or pushed by git's
default branch refspecs. Publish or inspect it explicitly when needed:

```bash
git push origin refs/notes/fak/decisions
git fetch origin refs/notes/fak/decisions:refs/notes/fak/decisions
```

Each note line is a `witness.Decision` JSON object with `op`, `verdict`,
`reason_class`, `tree`, `refused_argv`, and/or `pathspec_assertion` populated by
the producer.
