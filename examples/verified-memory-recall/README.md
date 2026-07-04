# verified-memory-recall — a verified, intent-ranked orientation block (#2347)

The loop reads a **markdown memory store**: a `MEMORY.md` index plus one `.md` file
per fact. The old loop-facing recall (`memoryread.RenderDigest`) dumped that store
unranked and **unverified** — so a stale note (a moved file, a renamed flag) loaded
wearing the authority of a fact.

`fak memory recall` runs the same [memq](../../internal/memq) algebra the recall core
uses, but over the notes store, and **re-verifies each note's concrete claims at
page-in**. It emits a budget-bounded, freshness-tagged orientation block:

- **`[fresh]`** — the note's concrete claim (a repo path, a git SHA, a CLI flag) still
  verifies against the working tree; the note renders.
- **`[withheld:stale_recall_artifact]`** — the claim no longer verifies; the note is
  **refused** and listed with the failing claim named as evidence. Its body is never
  rendered.
- **`[unverified]`** — a prose-only note with nothing checkable; it renders hedged.

This is a **backend, not a new driver** — `fak memory drivers` is unchanged by it.

## Run it (no key, no model, no GPU, no network)

From the repo root, with `fak` on your `PATH`:

```bash
bash examples/verified-memory-recall/run.sh
```

or straight from source:

```bash
FAK_BIN='go run ./cmd/fak' bash examples/verified-memory-recall/run.sh
```

The demo must run from inside this checkout: the default artifact verifier resolves
repo-relative paths against the git working tree, so the fresh note's claim
(`internal/memq/exec.go`) only checks out here.

Expected runtime: the run completes in seconds once `fak` is on `PATH`. It is
deterministic: the authored notes and repo-path witnesses are fixed, so re-running
against the same checkout emits the same freshness/refusal verdicts.

See [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md) for the expected transcript.

## Scope — what this does not claim

This demo does not claim to make markdown memory truthful by itself, rank every possible
operator note perfectly, or replace higher-cost review of risky recalled facts. It only
shows the read-time witness behavior for three authored notes: a fresh path claim, a
stale path claim, and a prose-only note.

## The store

Three authored notes, one per read-time verdict:

| file | claim | verdict |
|---|---|---|
| [`store/fresh.md`](store/fresh.md) | names `internal/memq/exec.go` (a path that exists) | `[fresh]` — rendered |
| [`store/stale.md`](store/stale.md) | names `internal/gonepkg/gone.go` (a path that is gone) | `[withheld:stale_recall_artifact]` — refused, claim named |
| [`store/prose.md`](store/prose.md) | a preference, nothing checkable | `[unverified]` — rendered hedged |

Delete `internal/memq/exec.go` and `fresh.md` flips to withheld; drop a real path into
`stale.md` and it flips to fresh. The verdict tracks the tree, not the author's word.

## Other surfaces

- `--json` emits the machine envelope (`rendered` / `withheld`, verdict per note; a
  withheld note never carries a body).
- `--format toon` / `--ablate-formats json,toon` re-encode the same verdicts and
  measure the byte/token cost of each encoding.
- `fak memory run --backend notes --store DIR --query-file PLAN.json` runs an authored
  query against the same backend.

Grounds the read half of epic **#2346 (R1)** — the store the loop actually consumes,
served through the verified memq path.
