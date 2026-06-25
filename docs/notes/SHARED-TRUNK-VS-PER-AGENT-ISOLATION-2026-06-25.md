---
title: "Shared live trunk vs. branch / worktree / VM-per-agent — the trade fak takes the other side of"
description: "The field converged on one answer to fleet concurrency: never share a tree. Cursor background agents, Devin, Sculptor, OpenHands, container-use, and GitFarm each give an agent its own branch, worktree, or VM, then gate integration through a human merge. fak takes the contrarian side — a single live trunk with pre-call adjudication and disjoint-lease admission. This note names that trade honestly: what each side buys, what it costs, why the field's choice validates fak's bet while giving it nothing to import for the hard part, and where the three named git-admin gaps stand today."
---

# Shared live trunk vs. branch / worktree / VM-per-agent

Date: 2026-06-25
Epic: [#822](https://github.com/anthony-chaudhary/fak/issues/822) (git-admin) — the
positioning half of the acceptance checklist. Child: [#834](https://github.com/anthony-chaudhary/fak/issues/834).

## 1. The field has converged, and it converged away from us

Ask any production agent fleet how it stops two agents from clobbering each other's
git state and you get one answer: **never share a tree.** The isolation unit varies,
the principle does not.

- **Cursor background agents** — branch per agent.
- **Cognition's Devin** — branch/workspace per agent, merge gated.
- **Anthropic's Sculptor** — isolated workspace per agent.
- **OpenHands** — per-task workspace.
- **Dagger's `container-use`** — a container *and* a git branch per agent. The clearest
  open implementation of the pattern: <https://github.com/dagger/container-use>.
- **Google's GitFarm** — per-agent isolation at fleet scale:
  <https://research.google/pubs/gitfarm/>.

Every one of them then gates integration through a **human merge**. Isolation removes
the concurrency hazard by construction; the merge re-introduces a serialization point a
person owns. The conflict never happens on a shared tree because there is no shared tree
until a human says so.

## 2. fak takes the other side

fak runs a **single live trunk (`main`)** that every session commits to directly. There
is no per-agent branch, no per-agent worktree, no VM. Concurrency safety comes from two
mechanisms that fire *before* a hazardous action, not from isolating the tree:

1. **Pre-call adjudication.** The kernel inspects every git command at the tool-call
   boundary and refuses a structurally-decidable hazard (force-push, `commit --amend`,
   `add -A`, `--no-verify`, off-trunk branch) *before* `git` runs. See
   [`internal/gitgate/gitgate.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/gitgate/gitgate.go)
   and the design note
   [`RESEARCH-git-in-kernel-prefilters-2026-06-22.md`](RESEARCH-git-in-kernel-prefilters-2026-06-22.md).
2. **Disjoint-lease admission.** Two workers may run concurrently only if their declared
   file trees are pairwise disjoint (`dos arbitrate`). The shared index is committed
   by explicit path, never `git add -A`, so a session lands exactly its own files.

The bet: **deterministic, pre-call refusal of structurally-decidable git hazards on a
shared, live trunk, with disjoint-lease admission** makes a shared tree workable without
a per-agent isolation unit — and gives the fleet a single, always-current trunk instead
of N divergent branches awaiting a merge.

## 3. The trade, named

| | Branch / worktree / VM per agent (the field) | Shared live trunk (fak) |
|---|---|---|
| Concurrency hazard | Removed by construction (no shared state) | Refused pre-call + admitted by disjoint lease |
| Integration | Human merge per agent | Direct commit; merge gate not required for safety |
| Divergence | N branches drift from trunk until merged | Trunk is always current; no drift to reconcile |
| Cost of isolation | A workspace/container/VM per agent | None — one tree |
| Failure mode | Merge backlog; stale branches; conflict at merge time | A laundered or stateful hazard that the prefilter cannot decide |
| What a person still owns | The merge | Review (the trunk is shared, not unreviewed) |

Neither side is free. The field pays for isolation with divergence and a merge backlog;
fak pays for a shared trunk with the burden of deciding hazards *before* they run. The
honest framing is not "fak removes the human" — it is that fak moves the human from a
**per-agent merge gate** to **review of a single live trunk**, and bets that pre-call
refusal plus disjoint leases is enough to keep that trunk safe between reviews.

## 4. Why the field gives us nothing to import

This is the load-bearing point. The field's convergence **validates** fak's contrarian
choice — everyone agrees uncoordinated multi-agent git on a shared tree is dangerous —
but because every surveyed system *avoids* the shared-tree problem rather than *solving*
it, **the field offers no importable technique for the hard part fak actually does.**

> No surveyed system adjudicates a git command *before it runs* against a live shared
> trunk and refuses it deterministically. That is fak's actual edge.

So fak's entire leverage lives in three named gaps between the current prefilter and that
edge being airtight. SOTA exists to *borrow from* for each gap, but only with the honesty
caveats below — none of it is a drop-in.

## 5. The three gaps, and where they stand today

1. **Shell-laundering evades the argv prefilter.** A hazard hidden inside `bash -c '…'`,
   a backtick subshell, `eval`, or `$(…)` never reaches the conservative tokenizer.
   *Status:* an unwrap pass now lifts quoted-shell / substitution sources back into the
   same hazard rules before the tokenizer (shipped, [#823](https://github.com/anthony-chaudhary/fak/issues/823)).
   The symlink-escape half — a path that prefix-matches a lease but resolves through a
   symlink to a target outside it — is closed on the executor side
   ([`internal/safecommit`](https://github.com/anthony-chaudhary/fak/blob/main/internal/safecommit/safecommit.go),
   `SYMLINK_ESCAPE`, [#827](https://github.com/anthony-chaudhary/fak/issues/827), CLOSED),
   resolving realpaths where the filesystem is touched and keeping
   `gitgate.treeContains` a pure logical check by design.
2. **Stateful laws a stateless prefilter cannot decide pre-call.** `OFF_TRUNK`, a peer's
   already-staged index, a foreign `MERGE_HEAD` — none are decidable from argv alone.
   *Status:* the safecommit executor now guards the peer-staged index and a foreign
   `MERGE_HEAD` before the pathspec commit
   ([#822](https://github.com/anthony-chaudhary/fak/issues/822)), and a
   `reference-transaction` hook refuses an off-trunk ref update at the git level
   ([`tools/githooks/`](https://github.com/anthony-chaudhary/fak/tree/main/tools/githooks)).
   These remain per-clone; a one-command installer + `core.hooksPath` pin so the floor is
   present on every clone is still open.
3. **Cross-machine atomicity.** Two clones on two machines committing the same trunk have
   no shared admission barrier. *Status:* `refs/fak/locks` ref-namespaced leases give
   cross-machine *visibility* ([#825](https://github.com/anthony-chaudhary/fak/issues/825)).
   Generalizing `CheckCollectiveCommit()` from single-tree to a multi-clone barrier on top
   of it ([#831](https://github.com/anthony-chaudhary/fak/issues/831)) is still open and
   buys distribution, not atomicity (see below).

## 6. Honesty caveats (carried verbatim from the research synthesis)

These travel with the epic and every child that borrows the technique. State them plainly;
do not let a borrowed name imply more than it buys.

- **`mvdan/sh` (shell-AST parse layer).** It parses shell *syntax*; it **cannot** resolve
  `$VAR`, `alias`, or `eval`'d dynamic strings — those stay out of reach and must **fail
  closed, not silently pass.** (fak shipped the unwrap pass on the standard library to
  keep the single static binary; the AST-parser acceptance box is superseded by that
  choice, not unmet.)
- **`grite`-style distributed coordination (the cross-machine barrier).** It provides
  **distribution, not atomicity**, across clones. Its reported **"78%→0%" conflict
  reduction is synthetic**, not a production measurement — treat it as a **design target,
  not evidence.**
- **Signature attestation (`sigstore`/`gitsign`).** A git SHA is **already
  content-addressed**, so a signature adds only the **envelope** — who/what attested — not
  integrity of the tree itself. Scope the claim to **identity, nothing more.**
- **"AgenticFlict"-style analyses.** They **size the concurrency-conflict problem** across
  agent fleets — how bad uncoordinated multi-agent git gets. They do **not** measure fak's
  residual after adjudication. Use them to **size the problem, never to claim fak's
  benefit.**

## 7. Non-goals

- This does **not** make the argv prefilter a full shell. Laundered/dynamic forms
  (`$VAR`, `alias`, `eval`'d strings) stay undecidable and must **fail closed.**
- A signature envelope adds **identity, not tree integrity** — the SHA already
  content-addresses the tree.
- The cross-machine broker buys **distribution/coordination, not transactional atomicity**
  across clones, and not a guarantee against a determined operator with raw `git` and shell
  access.
- None of this replaces the human review the rest of the field relies on at merge time.
  fak's bet is that deterministic pre-call refusal + disjoint leases makes a **shared live
  trunk** workable, not that it makes review unnecessary.

---

Source: the 56-agent research workflow behind epic #822 (parallel SOTA sweeps →
adversarial per-claim verification against primary sources → synthesis grounded in fak's
code). Field-prior-art claims were checked against fetched URLs; gap-status claims against
the tree at HEAD on 2026-06-25.
