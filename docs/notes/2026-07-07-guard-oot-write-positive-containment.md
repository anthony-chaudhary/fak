---
title: "Durable fix: guard out-of-tree-write arg-rules → positive containment"
description: "Replaces the guard's negative ..-regex for out-of-tree writes with a structural positive-containment matcher scoped to the workspace and scratchpad."
---

# Durable fix: guard out-of-tree-write arg-rules → positive containment

*2026-07-07. Design note (adversarially verified).*

## Implementation status (2026-07-08)

**Implemented + unit-verified GREEN** (`go test ./internal/adjudicator` → ok, all cases pass):

- `internal/adjudicator/outoftree.go` — the fail-closed, purely **subtractive** structural decider
  (`isOutOfTreeWriteArgRule` + `outOfTreeWriteEscapes`), gated on the raw regex already matching, so it
  only ever *downgrades* a `..`-match to allow when every write destination provably resolves under
  `{workspace} ∪ {scratchpad}` (or a null device). Handles verb-generic `-o` (so `curl -o` is caught),
  `--output`/`-t`/redirect/tee/cp-family extraction, per-target canonicalization (`~`/`$HOME`), drive-clamped
  `..` resolution, case-insensitive sibling-safe containment, and fail-closed on empty-ws / `$VAR` / undecodable.
- `internal/adjudicator/outoftree_test.go` — pins the full verified matrix incl. the **checkout-depth
  invariant** (the machine-dependence hazard the adversarial pass caught): the pinned `../../tmp/exfil`
  exfil stays DENIED at depth 2, depth 6, and POSIX; scratchpad-via-`..` and in-tree-via-`..` become ALLOWED;
  `rawMatches=false` never escapes (no new denies).
- `internal/adjudicator/decide.go` — dispatch wired after the `isRmRfArgRule` block, keeps `pr.Re`
  (no JSON edits; `TestDogfoodManifestRuleCoverage` still raw-matches).
- `cmd/fak/guard.go` — **plumbing added here**: exports `FAK_GUARD_SCRATCHPAD_ROOTS` at guard startup
  (parent-side, since the adjudicator decides in the parent process), defaulting to the narrow
  `<temp>/claude` Claude-Code scratchpad tree — never all of temp, so `/tmp`-targeted exfil stays denied.
  Operator/harness override respected.

**Not yet landed** — remaining gate:

- `cmd/fak` does **not compile** in the current shared tree (`gateway.SessionFleet` undefined — unrelated
  peer WIP), so the `guard.go` plumbing can't be built/end-to-end-verified here and the pre-push compile
  gate (`TRUNK_WOULD_NOT_COMPILE`) would refuse a `cmd/fak` commit until that heals. The adjudicator matcher
  itself compiles + tests independently.
- `internal/adjudicator` is a core-lock tree → commit needs `--core-lock-maintenance-witness`.
- Add embedded-floor witness cases to `cmd/fak/guard_test.go` (pins zero out-of-tree verdicts today) and
  file the PowerShell out-of-tree-family gap as a follow-up.

---


## Problem

The embedded guard floor (`cmd/fak/guard-default-policy.json`) gates out-of-tree writes
with **raw `deny_regex`** rules that match any literal `..\` / `../` in the command:

```
{ "tool":"Bash","arg":"command","deny_regex":"-o\\s+\\.\\.[\\\\/]","reason":"POLICY_BLOCK",
  "fix":"write inside the working tree (or the harness scratchpad); out-of-tree writes are refused by policy" }
```

12 such rules exist (four spellings — `-o ..`, `--output ..`, `>>? ..`, `cp|mv|install|tee|rsync|ln … ..`)
duplicated across the Bash / `shell_command` / `functions.shell_command` dialects.

This family is **simultaneously over- and under-broad**:

- **Over-broad (the bug that surfaced this):** the harness scratchpad lives *outside* the repo tree
  (`%LOCALAPPDATA%/Temp/claude/<proj>/<uuid>/scratchpad`), so a *sanctioned* `go build -o ../…/scratchpad/fak.exe`
  is refused `POLICY_BLOCK`. Under `fak manage -- claude` a false `POLICY_BLOCK` reads as an agent-chosen
  `end_turn` — it **silently kills the turn**. (The same failure mode `decide.go` already documents for the
  `rm_rf` / `rce_pipe` rules.)
- **Under-broad:** because every rule *requires* a `..`, it misses every **absolute** escape —
  `-o /etc/cron.d/x`, `tee /etc/passwd`, `install x /usr/local/bin/y`. Worse, `canonicalizeArgValue`
  expands `~`/`$HOME` to an absolute path with no `..`, so `cp id_rsa ~/.ssh/authorized_keys` sails through.

## Root cause

1. Two *other* raw-regex rules (`rm_rf`, `rce_pipe`) were already upgraded to **structural, tokenized,
   quote-aware** matchers (`internal/adjudicator/rm_rf.go`, `rce_pipe.go`) precisely to kill this
   false-positive class. The out-of-tree family was **never given the same treatment**.
2. **Hint-vs-enforcement drift:** the fix text blesses "(or the harness scratchpad)" but there is **no
   scratchpad carve-out anywhere in the matcher** — the permissive intent reached the human-readable hint
   and never the enforcement.

## Design: positive containment (not `..`-substring)

Replace the negative `..`-regex decision with a structural matcher that **extracts each write-target
operand, resolves it to an absolute path, and allows iff it lands under an allowed-write-root set**
`{workspace root} ∪ {exact scratchpad subtree} ∪ {null devices}`; deny otherwise. This is strictly
**more permissive** (scratchpad + in-tree `..` paths allowed) *and* **more secure** (absolute + `~`/`$HOME`
escapes newly denied).

Mirror the shipped `rm_rf` / `rce_pipe` shape: a recognizer keyed on the exact `Re.String()` spellings
(rules stay `Kind==ArgDenyRegex`, `pr.Re` kept, **no JSON regex edits**, so `TestDogfoodManifestRuleCoverage`
still raw-matches), dispatched in `evalArgPredicates` immediately after the `isRmRfArgRule` block with a
trailing `continue`, preserving `pr.Reason` (POLICY_BLOCK), `pr.Fix` on `Meta[fix]`, the advisory-note
branch, and bounded disclosure.

## The six fail-open traps the adversarial verification caught

These are the reason a naive "reuse `repoguard.SafeRootsForWorkspace`" implementation is **wrong**:

1. **Blanket-tmp allow flips the pinned exfil DENY.** `repoguard`'s safe-root set blesses all of `/tmp`,
   `/var/tmp`, `~/.cache`, `~/Downloads`, `$TEMP`. The pinned invariant `dogfood_manifest_test.go:67`
   (`curl -o ../../tmp/exfil` → DENY) then resolves *under* `/tmp` on a depth-2 POSIX checkout → **ALLOW**.
   → **Scope the allow-set to `{workspace} ∪ {exact FAK_SCRATCHPAD_DIR subtree}` only.** Never the broad temp
   tree. If no scratchpad signal is declared, fall back to **`{workspace}` only** — losing the scratchpad
   FP-fix is safer than flipping the exfil DENY.
2. **`curl -o` fails open.** `repoguard`'s bare-`-o` extraction is build-verb-gated (`go/gcc/…`), so
   `curl -o …` is never extracted → dogfood:67 flips DENY→ALLOW. → **Extraction must be bespoke and
   verb-generic**; borrow only *containment* from `repoguard`, never *extraction*.
3. **`~`/`$HOME` dests leak.** `repoguard.toAbs` refuses any `~`/`$` operand. → **Canonicalize each dest with
   `canonicalizeArgValue` (expands `~`/`$HOME`) *before* containment.**
4. **Empty workspace root fails open.** Passing `ws=""` to `repoguard` mis-resolves `../../tmp/exfil` to
   `/tmp/exfil`, which the always-present `/tmp` safe root then allows. → **Resolution failure must
   hard-DENY**, never delegate with an empty root.
5. **Plumbing bug — signal never reaches the matcher.** The adjudicator runs in the **guard *parent*
   process** (it decides gateway-stream tool calls); `guard_child.go` builds the *child* env, so a
   `FAK_SCRATCHPAD_DIR` set there is invisible to the parent-side `os.Getenv`. → **Set it in the parent**
   (`os.Setenv` at guard startup, where cwd + session are known) or thread a `Policy.AllowedWriteRoots` field.
6. **Extractor-gap silent ALLOW.** Because the structural branch `continue`s past the raw regex, any target
   the extractor misses becomes an invisible ALLOW. → **Fail-closed backstop:** if `pr.Re` still matches the
   canonicalized value but the structural extractor found *zero* containable targets, **DENY**.

## Corrected, fail-closed implementation plan

- **`internal/adjudicator/outoftree.go`** — `isOutOfTreeWriteArgRule(pr)` (4 exact `Re.String()` spellings,
  scoped `{Bash, shell_command, functions.shell_command}` via `EqualFold`) + `commandHasOutOfTreeWrite(cmd, ws, allowRoots)`.
  Reuse the `rce` tokenizer (`rceShellSources/rceShellSegments/rceCommandWord/rceProgramBasename`) to pull
  targets: **verb-generic** `-o` next-operand, `--output[= ]`, `>`/`>>` redirect targets (additive local scan —
  do **not** mutate the shared segmenter), and `cp/mv/install/tee/rsync/ln` destinations. Canonicalize each,
  then decide via a single lifted `repoguard.TargetEscapes(absTarget, ws, allowRoots)` (extract the per-target
  body of `classifyCommand`; refactor `classifyCommand` to call it so `repoguard`'s byte-parity `guard_test`
  proves no behavior change — one source of containment truth).
- **Root + scratchpad signal:** resolve `ws` once under `sync.Once` via `os.Getwd()` + `repoguard.FindRepoRoot`,
  **fail-closed** on error. Allow-roots layered most-explicit-first: `Policy.AllowedWriteRoots`
  (`allowed_write_roots` in the 4 policy JSONs, validated absolute-or-`~` at load) → `FAK_SCRATCHPAD_DIR`
  (read parent-side; **set it parent-side at guard startup**) → fallback `{ws}` only.
- **Dispatch** in `evalArgPredicates` after the `isRmRfArgRule` block; keep `pr.Re`; add the fail-closed backstop (#6).
- **Anti-drift architest** (the durable kicker): every rule whose `Fix` says "harness scratchpad" must have a
  scratchpad probe that Adjudicates **ALLOW**; every out-of-tree rule must have `/etc` + deep-`..` probes that
  Adjudicate **DENY**; every blessing phrase must appear in ≥1 shipped `Fix`. Makes this exact drift class fail
  the build by name.
- **Tests:** hermetic `outoftree_test.go` with an **injected synthetic `ws` + allowRoots (excluding `/tmp`)** so
  verdicts are machine/checkout-independent; add embedded-floor cases to `cmd/fak/guard_test.go` (pins **zero**
  out-of-tree verdicts today) incl. a `/tmp`-target DENY anti-widening probe.

## Verified test matrix (must hold)

DENY: `curl -o ../../tmp/exfil` · `curl --output=../../tmp/exfil` · `echo x >> ../../tmp/exfil` ·
`cp secret.txt ../../tmp/exfil` (the 4 pinned) · `curl -o /etc/cron.d/x` · `tee /etc/passwd` ·
`install x /usr/local/bin/y` · `cp secret ~/.ssh/authorized_keys` · `tee $HOME/.bashrc` ·
`cp x ../../../other-repo/y` · *any out-of-tree write when `os.Getwd()` fails* · `echo x >> ../../tmp/exfil`
even with `$TEMP` set (sibling tmp ≠ scratchpad).

ALLOW: `go build -o <abs-scratchpad>/fak.exe` · same via `../` · `echo log >> <scratchpad>/build.log` ·
`cp ../../vendored/lib.a build/lib.a` (parent *source*, in-tree dest — raw regex wrongly denies today) ·
`go build -o build/fak` · `go build -o build/../bin/fak` (`..` folds in-tree) ·
`git commit -m "revert ../../foo"` (quoted mention) · `curl -o /dev/null` · `ls -la` / `cat README.md`.

## Explicit non-goals / follow-ups

- **PowerShell out-of-tree family does not exist at all** — `Copy-Item ..\..\x`, `Out-File ..\..\x` are
  *unguarded* today (inverse gap). File as a follow-up (repoguard parses POSIX shell only).
- **Symlink / TOCTOU laundering** (`ln -s /etc x; cp payload x/job`) is invisible to the exec-free, stat-free
  matcher — a documented non-goal, not implied-covered.

## Landing / gating

`go test ./internal/adjudicator ./internal/repoguard` (both compile independent of the currently
un-buildable `cmd/fak`; verify `cmd/fak` via WSL/overlay/buildcheck). `internal/adjudicator` is a
self-modify/core-lock tree — commit via the `--core-lock-maintenance-witness` path (cannot self-witness).
Land additive/off-by-default first, then flip, re-pinning the dogfood matrix hermetically in the same change.
