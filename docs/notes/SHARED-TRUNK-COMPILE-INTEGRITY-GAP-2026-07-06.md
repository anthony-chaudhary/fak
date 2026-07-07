---
title: "The shared-trunk compile-integrity gap: lease disjointness is not build isolation"
description: "A live incident (a peer's uncommitted call to an undefined runSuperloopDrive broke every other session's cmd/fak build) exposes a fourth axis the git-admin epic's three gaps do not cover. Disjoint file leases (dos arbitrate) guarantee ownership disjointness at file granularity; Go compiles at package granularity, so two agents holding disjoint leases in the same package still share one build graph. On a shared dirty tree an agent cannot witness its own change — the repo's own witness-not-self-report floor breaks down. The runbook's archive-HEAD build already knows the fix but leaves it manual. Proposal: make the atomic commit prove it compiles in isolation — a static self-containment rung at the commit seam and an affected-scoped isolated build rung at the push seam, both fail-closed with a structured reason."
---

# The shared-trunk compile-integrity gap

Date: 2026-07-06
Relates to: epic [#822](https://github.com/anthony-chaudhary/fak/issues/822) /
[#834](https://github.com/anthony-chaudhary/fak/issues/834) (git-admin), and the two
sibling notes it should be read against —
[`SHARED-TRUNK-VS-PER-AGENT-ISOLATION-2026-06-25`](SHARED-TRUNK-VS-PER-AGENT-ISOLATION-2026-06-25.md)
(the positioning half) and
[`SHARED-CLONE-INTEGRATION-PASS-2026-06-29`](SHARED-CLONE-INTEGRATION-PASS-2026-06-29.md)
(the commit runbook).

## 1. The incident (witnessed, not hypothetical)

A session added a test file to `cmd/fak`, ran `go test ./cmd/fak/`, and got a clean
`ok cmd/fak 0.056s`. Minutes later the same command failed:

```
cmd/fak/superloop.go:55:10: undefined: runSuperloopDrive
```

`runSuperloopDrive` is *referenced* at `superloop.go:55` and *defined nowhere* — not in
the working tree, not at `HEAD`. A **peer** session, mid-authoring, had added the call to
its own leased file (`superloop.go`, the `cmd` lane) but not yet written the definition.
Because Go compiles a package as one unit, that half-edit broke the compilation of the
**entire** `cmd/fak` package — including the first session's unrelated, already-green test
file. The first session could no longer build, let alone witness, its own change. It had
to reconstruct `HEAD` in a throwaway worktree, drop its one file in, and build *there* to
prove its change was green.

This is not an exotic race. It is the steady-state condition of a shared working tree with
N concurrent authors, and it has happened before at the commit seam:
[`RELEASESTATUS-SHIP-REBIND-1448`](RELEASESTATUS-SHIP-REBIND-1448-2026-06-29.md) ("a
shared-trunk commit race crossed the ship-stamps") and
[`RELEASE-SIGNIFICANCE-FLOOR-REBIND-1389`](RELEASE-SIGNIFICANCE-FLOOR-REBIND-1389-2026-06-29.md)
("an index.lock race swept the significance-floor work into a milestone-tick commit").

## 2. The axis: file-lease disjointness ≠ build-graph disjointness

The shared-trunk bet rests on **disjoint-lease admission**: two workers run concurrently
only if their declared file trees are pairwise disjoint (`dos arbitrate`;
[`SHARED-TRUNK-VS-PER-AGENT-ISOLATION-2026-06-25`](SHARED-TRUNK-VS-PER-AGENT-ISOLATION-2026-06-25.md)
§2). That guarantee is real, and it is at **file granularity**.

But the unit of compilation is not the file — it is the **package** (in Go) or the module
graph more generally. `cmd/fak` is one `package main` spanning dozens of files across
several leaves (`superloop.go`, `guard*.go`, `affected.go`, ...). Two agents can hold
perfectly disjoint *file* leases inside that one package and still share **one build
graph**. So:

> Disjoint leases make edits *ownership*-disjoint. They do **not** make them
> *build*-disjoint. A peer's in-package half-edit compiles together with your file, and a
> broken half-edit fails your build for a symbol you never touched.

The deeper cost is to the repo's founding discipline. fak's floor is **witness, not
self-report**: don't believe a claim, verify it against an artifact (`dos verify`,
`dos commit-audit`, the `pre-push` `dos review` gate). But *witnessing your own change*
means building/testing it — and on a shared dirty tree, `go build`/`go test`/`fak affected`
all observe the **union of every session's uncommitted WIP**, not your change. A red result
may be a peer's; a green result is unattributable. **The shared dirty tree is the one place
the witness floor cannot stand**, because the thing being witnessed is not isolable.

This axis is *orthogonal* to the epic's three named gaps
([`…ISOLATION…`](SHARED-TRUNK-VS-PER-AGENT-ISOLATION-2026-06-25.md) §5), which are all
about git-*hazard* adjudication and cross-machine *coordination*:

| # | Epic gap | Class |
|---|---|---|
| 1 | Shell-laundering evades the argv prefilter | git-hazard adjudication |
| 2 | Stateful laws a stateless prefilter can't decide (OFF_TRUNK, peer index, MERGE_HEAD) | git-hazard adjudication |
| 3 | Cross-machine atomicity (two clones, one trunk) | coordination / distribution |
| **4** | **Compile-integrity of the atomic unit reaching the trunk** | **build integrity** |

None of gaps 1–3 asks "does the committed/pushed tree actually compile?" That is gap 4,
and it is unfilled.

## 3. What exists today — and where it stops

The fix is already *known*; it is just not *enforced*.

- **The runbook already prescribes it, manually.**
  [`SHARED-CLONE-INTEGRATION-PASS-2026-06-29`](SHARED-CLONE-INTEGRATION-PASS-2026-06-29.md)
  step 6 ("Gate the push"): `git archive HEAD | tar -x -C "$d"` into a dir *outside* the
  clone, then `go build ./... && go vet ./...`. The archive of `HEAD` is "the live tree
  minus the uncommitted remainder — exactly what origin will receive." Its step-3
  **Self-contained** axis is *precisely* the `runSuperloopDrive` shape: "no non-test file
  references a **new** top-level symbol that does not exist at `HEAD` and is defined only in
  another still-uncommitted file." This is advisory discipline — nothing runs it for you.

- **No git hook compiles anything.** The `pre-commit` hook
  ([`tools/githooks/pre-commit`](../../tools/githooks/pre-commit)) runs seven *content*
  gates — leak scan, secret shapes, doc placement, broken links, file admission, index
  sync, provenance — none of which build. The `pre-push` hook
  ([`tools/githooks/pre-push`](../../tools/githooks/pre-push)) runs `dos review`
  (claim-honesty: does each subject match its diff) plus `fak hygiene TIER_DECLARED`
  (architest tier drift) — also no build. **A pathspec commit that references an
  undefined symbol passes every gate and can be pushed.** `dos review` will happily
  witness a `feat:`/`fix:` commit that touches source; it never checks the source compiles.

- **`gitgate.CheckCollectiveCommit`** ([`internal/gitgate/gitgate.go:310`](../../internal/gitgate/gitgate.go))
  enforces the *structural* collective-commit invariants (pathspec-only, no peer-staged
  index, no foreign `MERGE_HEAD`). Structural, not a build.

- **`fak affected`** ([`cmd/fak/affected.go`](../../cmd/fak/affected.go)) is the closest
  existing machinery: it computes the packages a change touches (`affectedtests.Select`),
  runs `go test` on only those, and even re-runs the **baseline at `HEAD`** to attribute a
  failure to "mine" vs pre-existing. But it diffs the **working tree vs HEAD**, and on a
  shared clone the working tree is *collective* — so "mine" is "the whole fleet's
  uncommitted delta," not this session's change. It measures the dirty union; it cannot
  isolate one author. Its affected-package selection is exactly the primitive gap 4 needs;
  its *input tree* is the wrong one.

## 4. Proposal: make the atomic commit prove it stands alone

"Better atomic commits" = raise the definition of *atomic unit* from **"a coherent set of
file changes"** to **"a coherent set of file changes proven to compile against `HEAD` in
isolation from peers' WIP."** Two enforced rungs, cheap-static first, build second, both
fail-closed with a structured reason and the repo's standard `block|warn|off` + one-shot
escape ladder:

### Rung A — commit seam: static self-containment (`COMMIT_NOT_SELF_CONTAINED`)

Mechanize the runbook's step-3 Self-contained axis as a **new `fak hooks pre-commit`
gate** (no build, pure static analysis, ~ms):

- For each staged **non-test** `.go` file, collect the top-level symbols it references that
  do **not** exist at `HEAD`.
- If any such symbol is **not** defined in the staged set either, the commit references a
  symbol that exists only in still-uncommitted (peer or own) work → refuse
  `COMMIT_NOT_SELF_CONTAINED`, naming the symbol and the file.

This catches the `runSuperloopDrive` shape *at the commit seam, before push, without
compiling* — a purely local, fast, deterministic check. It is the static lower bound on
"self-contained"; it cannot prove type-correctness, only that no new top-level name is
dangling. That is enough to kill the dominant failure mode.

### Rung B — push seam: affected-scoped isolated build (`TRUNK_WOULD_NOT_COMPILE`)

Add a build rung to `pre-push`, *after* the existing `dos review` rung, that does what the
runbook says by hand — but scoped and enforced:

1. Reconstruct the pushed state in isolation: `git archive HEAD | tar -x` into a scratch
   dir **outside** the clone (never touches the shared dirty tree — the same reason the
   runbook uses a tempdir).
2. Compute the **affected packages** of `origin/main..HEAD` (reuse `affectedtests.Select`),
   and `go build` **only those + their dependents** in the scratch tree — not `./...`. On a
   7k-file tree this is the difference between seconds and minutes, which is what lets the
   gate run on *every* push instead of being skipped.
3. `go build` clean → allow. Build error → refuse `TRUNK_WOULD_NOT_COMPILE`, block by
   default (`FLEET_BUILD_GUARD=block|warn|off`, one-shot `FLEET_ALLOW_BUILD_BREAK=1`).

Push-seam is the load-bearing placement: a broken *local* commit is private and
recoverable; a broken *pushed* trunk poisons every peer that fetches. The runbook already
names the push as the atomic boundary ("peers only ever see the post-push state"). Rung B
makes "the post-push state compiles" a **witnessed** property of origin instead of an
assumed one — the same move fak makes everywhere else, now applied to its own trunk's
buildability.

## 5. Why this keeps the bet (and where it stops)

This does **not** retreat to worktree/VM-per-agent — the path
[`…ISOLATION…`](SHARED-TRUNK-VS-PER-AGENT-ISOLATION-2026-06-25.md) deliberately argues
against. It *hardens* the shared-trunk bet on the one axis the three git-hazard gaps miss.
It reuses seams that already exist (the hook idiom, `affectedtests.Select`, the archive-HEAD
technique) rather than adding a new subsystem.

Honesty caveats, in the house style:

- **Isolation is at the seam, not during editing.** Rungs A/B make the *committed/pushed*
  unit sound; they do nothing for a session building the *live dirty tree* mid-work. That
  session still needs the throwaway-worktree trick to witness itself before committing —
  which suggests a small `fak witness-mine` helper (archive HEAD + apply only *my* leased
  paths + affected-build) as the inner-loop complement to `fak affected`. Noted, not
  designed here.
- **Rung A is a lower bound.** Static top-level-symbol resolution catches dangling *names*,
  not type errors or signature drift. It is deliberately not a compiler; Rung B is where
  real type-checking happens.
- **Cost is real even when scoped.** A push touching a widely-depended-on package (e.g. a
  root `internal/` leaf) fans out to many dependents; the affected build is bounded by the
  reverse-dependency cone, not by the diff size. `warn` mode exists for exactly the settling
  period.
- **Windows caveat.** Native Windows clones hit an OS test-binary block (AGENTS.md), so
  Rung B must be `go build`, not `go test` — buildability, not the test suite. Test
  witnessing stays with `fak affected` / CI.
- **Cross-machine is still only visibility.** Rung B gates a single clone's push. Two
  clones racing the same trunk remain gap 3 (`refs/fak/locks` gives visibility, not a
  barrier); a compile gate on each push narrows but does not close the interleaving window.

## 6. Non-goals

- Not a merge queue and not per-agent branches — the shared trunk stays.
- Not a full type-check at the commit seam (Rung A is static-symbol only; Rung B, at push,
  is the real build).
- Not a replacement for `fak affected` or CI — a buildability floor on the trunk, below the
  test floor.
- Not a change anyone should land silently: `pre-commit`/`pre-push` are admission floor and
  are **park-by-default / sensitive**
  ([`SHARED-CLONE-INTEGRATION-PASS…`](SHARED-CLONE-INTEGRATION-PASS-2026-06-29.md) step 3).
  This note is the proposal; the hook change wants explicit review, a `warn`-mode soak, and
  its own issue before it becomes an enforced `block`.
