# Renaming a concept — `fak rename-concept`

A concept in this repo is never one string in one file. By the time a name like
`dgxbridge` has lived for a few weeks it is a `cmd/` directory, an `internal/`
package, a `dos.toml` lane, a `.gitignore` entry, a constant in Go source, prose
in docs, rows in append-only ledgers, and two stale `.exe` artifacts. Renaming
it with an editor's find-and-replace fails the same three ways every time:

1. **A case form is missed.** `dgxbridge`, `DGXBRIDGE`, `Dgxbridge`,
   `dgx_bridge`, `DgxBridge` — each spelling is a separate search nobody runs
   all of.
2. **History gets rewritten.** A dated note under `docs/notes/` or a JSONL
   ledger under `docs/nightrun/` records what the concept *was called at the
   time*; rewriting it forges the past.
3. **Artifacts get patched.** A generated binary whose name carries the concept
   wants to be *regenerated* under the new name, not byte-patched or moved.

`fak rename-concept` turns the rename into a planned, triaged operation:

```bash
fak rename-concept --from dgxbridge --to slackbridge          # dry-run plan (default)
fak rename-concept --from dgx-bridge --to slack-bridge --json # machine-readable plan
fak rename-concept --from dgxbridge --to slackbridge --apply  # apply the mechanical share
go test ./internal/renameconcept/                             # prove the triage on a fixture tree
```

## What the plan contains

- **Variants** — the old/new spelling expanded into matched case forms (kebab,
  snake, SCREAMING, joined, Pascal, camel, Capitalized). Mark word boundaries in
  the input (`dgx-bridge`, not `dgxbridge`) and the hump forms (`DgxBridge`)
  expand too; a single-word input can't know where the hump goes, so those hits
  surface as *irregular* instead of being guessed at.
- **Path renames** — files and directories whose *name* carries the concept,
  ordered deepest-first so children move before parents.
- **Mechanical sites** — files `--apply` may rewrite: Go source, live docs,
  config (`dos.toml` lanes, `.gitignore` rules), scripts.
- **Holdouts** — sites the tool lists but never touches, each with the reason:
  - *append-only records* (`.jsonl`/`.csv`, `docs/nightrun/`) — rename forward,
    don't rewrite history (`--include-historical` overrides deliberately);
  - *dated notes* (`docs/notes/`) — same rule;
  - *binary artifacts* — regenerate from source under the new name;
  - *irregular casings* — a spelling no variant covers (`DGXBridge` for a
    single-word input) is a human decision, not a guess.
- **Commit paths** — the explicit pathspec list a shared-trunk commit of the
  rename must name (this repo forbids `git add -A`).

After `--apply`, the tool **re-scans** and reports the residual from evidence —
what remains is exactly the holdout list plus any irregular casings, never an
assumption that "sed got everything".

Scan scope: dot-directories (`.git`, `.dos`, `.fak` scratch mirrors,
`.goal-runs`, …) are tool state and stay out of scope, with the one exception of
`.github`, whose workflows genuinely carry concept names.

## Worked example: `dgxbridge` → `slackbridge`

The rename this tool was built against (a concept that embeds lab-hardware
naming and wants a transport-named replacement). A dry run on this tree,
trimmed — the live counts drift as the tree moves:

```
rename-concept dgxbridge -> slackbridge  (mechanical_with_holdouts)
  scanned 7349 file(s): 193 exact match(es), 15 irregular-case hit(s)
  variants: DGXBRIDGE Dgxbridge dgxbridge

  PATH RENAMES (3, deepest-first):
    internal/dgxbridge/dgxbridge_test.go -> internal/dgxbridge/slackbridge_test.go
    internal/dgxbridge -> internal/slackbridge
    cmd/dgxbridge -> cmd/slackbridge

  MECHANICAL (27 — --apply rewrites these):
    config   .gitignore  (3 match(es))
    go       cmd/dgxbridge/main.go  (39 match(es))
    docs     docs/concept-rename.md  (21 match(es), 3 irregular stay(s))
    config   dos.toml  (5 match(es))
    script   tools/scrub_hardware_names.py  (5 match(es))
    ...

  HOLDOUTS (4 — listed, never touched):
    binary   dgxbridge.exe  (0 match(es)) — binary artifact: regenerate from source after the rename, do not patch bytes
    binary   dgxbridge_local.exe  (0 match(es)) — binary artifact: regenerate from source after the rename, do not patch bytes
    data     docs/nightrun/collected.jsonl  (10 match(es)) — append-only record: rename forward, do not rewrite history (--include-historical overrides)
    docs     docs/notes/SLACK-CONTROL-FOUNDATION-2026-07-02.md  (1 match(es)) — historical record under docs/notes/: leave the old name in history (--include-historical overrides)
```

(Yes, this doc is itself a mechanical site in the plan it documents — a real
rename would rewrite these examples, which is exactly right.)

Note what the plan caught that a grep-for-content sweep misses: the
`internal/dgxbridge` *directory* rename, the two stale binaries that must be
regenerated rather than moved, and the boundary between live docs (rewrite) and
dated notes (leave).

The follow-ups after an `--apply` are always the same shape: `go build ./...`
(import paths under a renamed `cmd/`/`internal/` dir), `fak affected`
for the impacted tests, regenerate the artifacts, and commit the plan's
`commit_paths` explicitly.

## Keeping concepts renameable (modularity notes)

The tool makes a rename mechanical; these habits keep it *small*. Each one is a
form of the same rule — **a concept's name should have one authoritative home,
and everything else should derive from it**:

- **One package, one directory, one lane.** The cheapest rename is
  `internal/<name>/` + `cmd/fak/<name>.go` + a `dos.toml` lane entry — three
  sites that the plan's path-rename and config buckets handle mechanically.
  A concept smeared across helpers in five packages renames five times.
- **Derive identifiers from the concept, in a declared case form.** `FooBar`,
  `foobar`, `FOO_BAR` are all recoverable from one spelling *if word boundaries
  are consistent*. An undeclared hybrid (`FOObar`) is what the irregular
  detector exists to catch — every one of those is a manual decision at rename
  time.
- **Don't put concept names where history accrues.** A lane name in an
  append-only ledger row means every rename leaves a seam in the data (the
  ledger says the old name forever; queries must know both). Where a record
  will outlive the name, prefer a stable id and map it to the display name at
  read time.
- **Don't name concepts after hardware, vendors, or hosts.** The worked example
  exists because a lab-hardware name got baked into a public concept, and a
  whole scrub layer (`tools/scrub_hardware_names.py`) now has to distinguish
  the identifier from the hardware tell, forever. A transport- or role-named
  concept (`slackbridge`, `chatrelay`) never needs scrubbing.
- **Binaries and generated artifacts should be regenerable by one command.**
  The holdout list says "regenerate" — that instruction is only cheap if
  `go build ./cmd/<name>` (or the Makefile) is the whole answer.

The pure logic lives in [`internal/renameconcept`](../internal/renameconcept/renameconcept.go)
(stdlib-only, tier 1, off the hot path); the CLI shell is
[`cmd/fak/renameconcept.go`](../cmd/fak/renameconcept.go). The fixture test
(`go test ./internal/renameconcept/`) proves the triage: mechanical sites
rewritten, history byte-identical, binaries unmoved, irregular casings
surfaced.
