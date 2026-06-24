---
name: modularize
description: One focused, repeatable pass that retires the code-quality scorecard's `architecture` debt — the god-files (>1500 lines) and god-functions (>200 lines) that /quality-score flags as RISKY and explicitly defers to "a focused pass". Splits a monolith along REAL concern seams via behavior-preserving code motion (the goimports recipe: hazard-check → plan boundaries with tools/godsplit_plan.py → sed-extract → goimports -w → gofmt → verify → prove → commit by explicit path), and extracts long functions into named helpers (linear flow → helper + struct-unpack) or a named state struct with methods (closure-soup). Re-measures to PROVE architecture-debt dropped, verifies behavior with go build + go vet + go test, and commits ONLY the touched packages by explicit path. The architecture-KPI focused pass /quality-score points to. Use when `architecture` is the heaviest KPI, to drive code-debt toward 0, or on a /loop cadence to keep the kernel from re-accreting monoliths.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Write, Edit, Grep, Glob
argument-hint: "[file-or-package] (no args = take the heaviest architecture defect from the scorecard)"
---

# /modularize — split the monoliths, provably, without changing behavior

> **What this does.** `/quality-score` measures ten KPIs and retires the safe ones,
> but it marks `architecture` **RISKY** and defers it: *"do it in ONE lane, keep
> behavior identical… if you can't do it safely this pass, leave it for a focused
> pass — do NOT half-do a refactor."* **This is that focused pass.** It takes the
> god-files and god-functions that drag the `architecture` KPI to the floor and
> splits them along real concern seams — as **behavior-preserving code motion**,
> verified by the toolchain, proven by the scorecard, committed by explicit path on
> the shared trunk. In one run it took fak from code-debt **12 → 0**, score **86.2
> (B) → 95.9 (A)**.

The shape: **find the heaviest architecture defect → plan the cut (boundaries +
hazards) → split (file = code motion, function = extract) → verify (build/vet/test +
behaviour review) → prove the drop → commit ONLY your packages by explicit path.**

## The one insight that makes this safe

Moving a top-level declaration to another file **in the same package** is a *semantic
no-op in Go* — package, not file, is the scope. So a god-file split can't change
behavior, and `go build` + `go vet` + `go test` is a near-complete gate. There are
exactly **three exceptions** that make code motion NOT a no-op; `tools/godsplit_plan.py`
flags all three:

1. **per-file build tags** (`//go:build …`) — moving a decl changes which file carries the constraint;
2. **`func init()`** — init order is **filename-alphabetical** across a package, so moving an `init()` between files can silently reorder initialization;
3. **aliased imports** (`x "path"`) — `goimports -w` re-derives plain imports after a move but does **NOT** re-infer a local alias; you must copy it to the new file by hand.

If a file has a build tag or an `init()`, keep those decls where they are (or in a file
whose name preserves init order) — don't move them blind.

## Step 1 — Find the target (the scorecard builds the work-list)

```bash
python tools/code_quality_scorecard.py --no-toolchain --json   # fast; corpus.kpis architecture.defects = your work-list
```

Read `architecture`'s `defects`: each is `god-file <path> (N lines > 1500)` or
`god-function <path>:<fn> (N lines > 200)`. Attack the heaviest first, and **bias to
the most critical packages** (the kernel — `adjudicator`, `gateway`, `model`,
`ggufload` — and `cmd/fak`, the entry point — over demos/benchmarks). Record the
baseline: `architecture defects = K`, `code_debt = N`, `score = S`.

## Step 2 — Plan the cut (boundaries + hazards, computed not guessed)

The error-prone part of a clean split is cutting at the right line: a decl's doc comment
sits ABOVE its keyword, so a naive cut at `func`/`type` orphans the comment. Let the
planner compute the doc-comment-aware ranges and the three hazards:

```bash
python tools/godsplit_plan.py internal/model/kv.go            # human table
python tools/godsplit_plan.py internal/model/kv.go --json     # machine payload
```

Each decl row is `block_start..block_end <kind> <name>` — `block_start..block_end` is the
exact `sed -n 'A,Bp'` range that carries the decl **with its doc comment** (func `[NL]` sizes
are advisory; the Step 1 scorecard is the authority on what's a god-function). Group adjacent
decls into **cohesive concerns** (e.g. kv.go → `kvcache.go` for the KVCache type +
eviction, `rope.go` for rotary embedding, leaving the Session/forward core). Splitting on
real seams is the whole point — see the anti-gaming laws.

**Sanity-check the plan before trusting it** — `godsplit_plan.py` is a LINE parser, not a Go
tokenizer. Fall back to a hand cut if ANY of these hold: a decl row prints `!! INVERTED
RANGE`; the `raw_strings` hazard is > 0 (a multi-line backtick string can hide or shift a
column-0 decl — eyeball those ranges); or the plan's decl count disagrees with the truth:

```bash
grep -cE '^(func|type|var|const) ' <file>   # must equal the number of rows the plan printed
```

`goimports` is the engine of the file split and is **not preinstalled** — install it and
confirm it resolved before you cut anything:

```bash
go install golang.org/x/tools/cmd/goimports@latest
command -v goimports || { echo "goimports missing — $(go env GOPATH)/bin not on PATH"; exit 1; }
```

## Step 3 — Split

Run the snippets below with the **Bash tool** — they are POSIX sh (`sed`, the `mk()`
function, `goimports`/`gofmt`), not PowerShell. **Precondition: `gofmt -w <file>` BEFORE you
plan**, so the package clause and import block are blank-separated and no extract range can
start at the package line.

### A. God-FILE → concern files (pure code motion)

Create each new file from its block range, then delete the moved ranges **bottom-up** (so
earlier deletes don't shift later line numbers), then let `goimports` fix every import. The
package name and target dir are read from the file so the snippet is copy-paste-safe in any
package:

```bash
F=internal/model/kv.go
D=$(dirname "$F")
PKG=$(python tools/godsplit_plan.py "$F" --json | python -c 'import sys,json;print(json.load(sys.stdin)["package"])')
mk(){ { echo "package $PKG"; echo; sed -n "$2,$3p" "$F"; } > "$D/$1"; }
mk kvcache.go 10 272          # ranges = block_start..block_end from godsplit_plan.py
mk rope.go    389 577
sed -i '389,577d' "$F"; sed -i '10,272d' "$F"      # BOTTOM-UP: highest range first
goimports -w "$F" "$D/kvcache.go" "$D/rope.go"     # auto-fix imports
gofmt -w     "$F" "$D/kvcache.go" "$D/rope.go"
```

Then **re-add any aliased import by hand** to the new file (`godsplit_plan.py` lists them —
`goimports` won't re-derive a local alias, and the build fails `undefined: <alias>` without
it). If the source file leads with a license/SPDX header above `package`, copy it to each new
file too. The `sed -n 'A,Bp' > newfile` and `sed -i` here are the right tool for line-range
code motion; redirect only to **in-tree** paths — the repo-guard PreToolUse hook
(`tools/repo_guard.py`; AGENTS.md `OUT_OF_TREE_WRITE`) refuses a `> /dev/null` or `../` escape.

### B. God-FUNCTION → helpers

- **Linear flow** (e.g. a CLI `cmdServe`/`cmdGuard`): extract coherent phases into helpers.
  Return a small **struct** and unpack it to the **SAME local names** the caller already
  uses (`up, base, key := us.provider, us.baseURL, us.apiKey`) so the rest of the function
  is untouched — no cascading edits, smallest possible diff.
- **Closure-soup** (a function whose closures share ~15 mutable locals, e.g. an SSE relay):
  make the shared state a **named struct** and turn the closures into **methods** on it.
  That's genuine modularity — named, individually-readable parts — not line-shuffling.
- Avoid param explosion: when a helper would take 8+ args, group them into a struct. A
  helper with 11 positional params is worse code than the god-function you started with.

## Step 4 — Verify (behaviour is preserved — don't assume it, prove it)

`go build` and `go vet` run natively; **run the tests under WSL** — per AGENTS.md, native
`go test` is blocked by an OS Application-Control policy on freshly-compiled test binaries on
this checkout (an OS quirk, not a code failure), and `/quality-score` validates the same way:

```bash
go build ./...                                                   # whole-tree compile (catches a lost/duplicated symbol)
go vet ./<pkg>/...                                               # typecheck incl. _test.go
wsl -e bash -lc 'cd /mnt/c/work/fak && go test ./<pkg>/... -count=1'   # the real gate
```

If native `go test` happens to run on your host, you may use it — but treat an
Application-Control block on a test binary as the documented ENVIRONMENT quirk, never as a
behaviour regression in your split. For `internal/model`, run the **full** (non-`-short`)
suite so the bit-exact oracle witnesses fire. If you DELEGATED a split to subagents, have an
independent reviewer confirm behaviour-preservation (no changed strings/flags/control-flow/
exit-codes) **before** committing — read the diff in the reviewer, not in your own context.

Verify formatting via the scorecard's `format` KPI (full run), not a naive `gofmt -l .`:
this checkout is CRLF in the working tree, so a local `gofmt -l .` flags hundreds of files as
a line-ending artifact (see `tools/gofmt_debt_audit.py` for the blob method).

## Step 5 — Prove the drop

```bash
python tools/code_quality_scorecard.py --no-toolchain --json   # architecture.defects should shrink
```

State it plainly: `architecture-debt K → K', code-debt N → M, score S → S'`. The bar for a
pass is **the heaviest architecture defect retired with behaviour proven green**.

## Step 6 — Commit ONLY your packages, by explicit path

```bash
git status --porcelain                         # see which files peers are editing RIGHT NOW
git add -- <your new files only>               # NEVER git add -A on this shared tree
git commit -s \
  -m "refactor(<leaf>): split <file> by concern" \
  -m "Pure code motion, no behavior change. <file> N->M lines; extracted <a.go>, <b.go>. Verified build/vet/test green." \
  -- <orig.go> <your new files>                # path-scoped: excludes peer-staged files
git push
```

- **Route around peer WIP.** A file showing in `git status --porcelain` is being edited by a
  peer — do NOT split it this pass; pick a clean target. A path-scoped commit leaves peer-staged
  files alone (you'll see them in `git diff --cached` — that's expected; your `-- <paths>` excludes
  them from the commit).
- **Honest subject.** A split/extraction is `refactor(<scope>):` — never `feat(`/`fix(`. End every
  ship commit with a `(fak <leaf>)` trailer (`(fak gateway)`, `(fak model)`; a `cmd/<dir>` demo →
  `(fak <dir>)`).
- **Shared-trunk law.** Stay on `main` (the `OFF_TRUNK` guard refuses a branch — this is why
  worktrees don't work, see below). Commit by explicit path; if a peer's `MERGE_HEAD` is set, wait
  for it to clear; never force-push.

## Parallelizing across many targets (when there are several god-files)

The Workflow tool's `isolation: 'worktree'` **fails in this repo**: `git worktree add` creates a
branch and the trunk guard refuses it (`OFF_TRUNK`). To fan out: give each agent **one disjoint
leaf package** (nothing two agents both touch), tell each to edit **in place**, run only **scoped**
builds (`go build ./<pkg>/...`, never `go build ./...` which races peers), and run **no git** — then
the orchestrator verifies and commits each package by explicit path. Have an independent reviewer
check the delegated diffs for behaviour-preservation before committing.

## Anti-gaming laws (the measure is only as honest as the pass)

1. **Split on REAL concern seams, never arbitrary line counts.** Each new file is one cohesive
   concern (the loader, the rotary embedding, the eviction path) — not "lines 1–700 / 701–1400". A
   split that just chops a file in half to beat 1500 is gaming, not modularity.
2. **Never create a file-per-function.** Over-splitting is its own debt; group related decls.
3. **Behaviour MUST be preserved — verify, don't assume.** Pure code motion is safe *only* once the
   three hazards (build tags, init order, aliases) are cleared; an extraction can silently change a
   captured variable. Build + vet + test (+ review for delegated work) is non-negotiable.
4. **Never touch the frozen ABI** (`internal/abi`) and never add a dependency to move a number.
5. **`goimports` will not re-add a local import alias** — copy it by hand, or the build breaks
   (`undefined: <alias>`); `godsplit_plan.py` lists the aliases to re-add.

## When to run this

- When `architecture` is the heaviest KPI in `/quality-score` (it defers here on purpose).
- To drive code-debt the last mile to **0** after the safe classes are retired.
- After a feature lands a 1500+-line file or a 200+-line function in a critical package.
- On a `/loop` cadence to keep the kernel from re-accreting monoliths between releases.

The scorecard and `godsplit_plan.py` are read-only; this skill's only writes are your genuine
splits and the new concern files. It never edits the frozen ABI, never games a seam, and never
commits a behaviour change it didn't verify.
