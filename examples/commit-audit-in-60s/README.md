# fak kernel — commit-audit-in-60s: catch a commit that lies about what it did

**One command, under a minute, no API key, no model, no GPU, no network.** A commit
lands with the subject `fix: handle nulls in add()` — but its diff only edits a
`README.md`. `dos commit-audit` reads the commit and returns **CLAIM_UNWITNESSED**: a
code-effect claim the diff does not back. Then the *same subject* on a commit whose
diff actually edits `calc.py` returns **OK / diff-witnessed**. The only thing that
changed is the diff — proving the verdict is read from what git recorded, not from the
subject text anyone could have typed.

```
   commit subject: "fix: handle nulls in add()"     (author-authored → FORGEABLE)
                              │
              ┌───────────────┴────────────────┐
              ▼                                ▼
   diff touches only README.md         diff touches calc.py
   (no source file)                    (a source file)
              │                                │
              ▼                                ▼
   ⚑ CLAIM_UNWITNESSED                 ✓ OK  [diff-witnessed]
     [subject-only]                      "code-effect claim witnessed
      "the diff touches no                by a touched source file"
       SOURCE file"
              │                                │
              ▼                                ▼
        exit 1                            exit 0
```

## Prerequisites

The `dos` CLI on `PATH` (or `DOS_BIN=/path/to/dos`) and `git`. **No model, API key,
GPU, or network**: the verdict is a pure function of `(the commit subject, the files
the commit touched)` — so the demo is deterministic and CI-usable. `dos` is a tiny
Python wheel; install it once:

```bash
pip install dos-kernel==0.29.0   # pinned; the bare `dos` package is an unrelated squatter, the name is `dos-kernel`
```

## Run it

```bash
bash examples/commit-audit-in-60s/run.sh
```

The script builds a throwaway git repo in a temp directory, makes both commits, audits
each, and deletes the repo on exit — it never touches your real history. The full run
completes in a few seconds. On Windows run it under Git Bash or WSL (`bash`); on any
platform you can also just run the two commands by hand — `git commit` then
`dos commit-audit HEAD --workspace .`.

## Expected output

Two `dos commit-audit` witnesses back to back (the short SHA is the throwaway repo's
own commit and differs each run):

```
== 1/2 THE OVER-CLAIM — subject says 'fix: handle nulls in add()', diff touches only README.md ==
⚑ UNWITNESSED  5f7f6c8  [subject-only]  code-effect claim but the diff touches no SOURCE file (only: README.md) — the claim rests on the subject text

commit-audit: 1/1 commit(s) make a claim their diff does not witness.
  -> exit 1  (non-zero: the claim is NOT witnessed by the diff)

== 2/2 THE HONEST COMMIT — same subject, diff touches calc.py (a source file) ==
✓ witnessed   62a5f94  [diff-witnessed]  code-effect claim witnessed by a touched source file
  -> exit 0  (zero: the diff witnesses the claim)
```

Or run an audit by hand — from inside any git repo, this is the one command the script
calls per commit:

```bash
dos commit-audit HEAD --workspace .
```

## Why this is the interesting kind of "no"

- **The subject is forgeable; the diff is not.** Whoever writes the commit message
  authors its words — a human typing `fix: handle nulls` and an agent stamping
  `--allow-empty "shipped"` are the same shape. The *files the commit touched* are
  authored by the commit machinery itself. `dos commit-audit` grades the relationship
  between the two: does the diff do the KIND of thing the subject claims?
- **The verdict flips on the diff, not the subject.** Both commits carry the identical
  subject `fix: handle nulls in add()`. Case 1 is refused because its diff touches no
  source file; case 2 is witnessed because it touches `calc.py`. Same words, different
  verdict — the auditor never read the author's intent, only git's record.
- **The exit code IS the gate.** `0` = witnessed, `1` = an unwitnessed claim, `2` = an
  unreadable ref. That makes `dos commit-audit` a drop-in CI check or pre-push hook: a
  lying commit message fails the build the same way a failing test does, with no model
  in the loop.

## Scope — what this does **not** claim

This is a **name-level** demo of one verdict: a code-effect claim contradicted by a
doc-only diff. It does **not** check correctness (a real fix to the *wrong* bug still
passes — the auditor grades *did the diff do the kind of thing claimed*, never *was it
right*); it does not resolve issue references (`closes #42` needs an issue→files map it
does not carry); and it does not exercise DOS's plan/phase ledger (`dos verify` does
that). No adoption or benchmark claim is made here — the demo proves one thing: a
self-authored "done" is re-checked against git, not trusted.

## Files

| file | what it is |
|---|---|
| `README.md` | this walkthrough |
| `run.sh` | the one command — builds a throwaway repo, makes both commits, audits each |

Related: [`deny-in-60s/`](../deny-in-60s/README.md) (the parallel 60-second demo — a
default-deny floor refusing an irreversible call); the concept it serves —
[verify, don't trust (DOS)](../../docs/notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md).
