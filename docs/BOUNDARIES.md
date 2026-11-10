---
title: "Boundaries — who owns what, where new work goes"
description: "The one-pager for boundary questions: which entry doc you are reading, the one commit command, public/private authority slices, generated vs authored paths, ownership, frozen surfaces, CLI families, and refusal recovery — every row points at its canonical authority."
---

# Boundaries: who owns what, where new work goes

One page for the many engineers joining this repo. When the question is "which layer,
which doc, or which command owns this?", start here. This page routes; it does not
duplicate. Every row points at the canonical authority — read that, don't copy from here.

## 1. Which entry doc am I reading?

| Doc | Who it's for | Job |
|---|---|---|
| [README.md](../README.md) | Everyone | Product front door: what fak is, why, how to try it |
| [START-HERE.md](../START-HERE.md) | Humans with a task | Task router: pick your job, get one destination + next action |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Contributors | Contributor contract: setup, route table, checks, commit rules |
| [AGENTS.md](../AGENTS.md) | Coding agents | Canonical agent instructions: build/test/run, repo map, hard rules |
| [CLAUDE.md](../CLAUDE.md) · [AGENT.md](../AGENT.md) · [GEMINI.md](../GEMINI.md) · [.github/copilot-instructions.md](../.github/copilot-instructions.md) | Per-harness stubs | Short auto-loaded stubs that defer to AGENTS.md; each harness loads its own file |
| [llms.txt](../llms.txt) | Agents / answer engines | Curated doc map: current authorities per task |
| [INDEX.md](../INDEX.md) | Name-based lookup | Exhaustive repo map: find a doc, component, or artifact by name |

Rule of thumb: humans → START-HERE; agents → AGENTS.md (the four stubs are just
pointers); exhaustive lookup → INDEX.md.

## 2. The commit command — one answer

Commit via `fak commit --path <p> -m "<subject> (fak <leaf>)"` or `fak sweep --apply`.
DCO sign-off is automatic; the `(fak <leaf>)` trailer binds the ship stamp so `dos verify`
can bind it. Raw `git commit -s ... -- <paths>` is **emergency-only**, permitted only when
the `fak` binary is unbuilt. Never use `git add -A` or uncoordinated raw commits on the
shared trunk. Authority: [AGENTS.md](../AGENTS.md) ("Commit exactly one issue through
explicit paths") and [CONTRIBUTING.md](../CONTRIBUTING.md) ("Commit small and by explicit
path").

## 3. Public vs private — which doc owns which slice

| Slice | Owning doc |
|---|---|
| Process split: public runtime vs private autonomous dev factory | [dev-process-private-boundary.md](dev-process-private-boundary.md) |
| Harness/SDK seam + the `pkg/` export invariant | [architecture/public-private-harness-boundary.md](architecture/public-private-harness-boundary.md) |
| Dev-tooling CWD rule (companion dev harness) | [architecture/companion-dev-harness.md](architecture/companion-dev-harness.md) |
| GPU/lab slice: what stays private | [gpu-server-private-boundary.md](gpu-server-private-boundary.md) |
| Lab access: the public pointer into the private channel | [private-comms-channel.md](private-comms-channel.md) |

Point, don't duplicate: each doc owns its slice; this page only routes.

## 4. Generated vs authored

| Path | Generator / gate | Marker |
|---|---|---|
| `internal/devhandoff/commands_gen.go` | `go run ./cmd/fak-dev index ownership --write-manifest` | Gate-locked trio, [VERB-LIFECYCLE.md](VERB-LIFECYCLE.md) §4 |
| `docs/generated/verb-surface.md` | `go run ./cmd/verbsdoc -write` | `VERB_SURFACE_SYNC` staleness gate |
| `cmd/fak/verbs_gen.go` | `go run ./cmd/fak-dev index verbs --write-usage` | Feeds `fak help <verb>` |
| `docs/generated/disambiguation/*` | Concept-disambiguation scorecard | In-file do-not-edit marker |
| `docs/benchmarks/published/INDEX.md` | Benchmark publish pipeline | In-file do-not-edit marker |
| `docs/FAQ.md` | `tools/gen_structured_data.py` | JSON-LD block marked "do not edit by hand" |
| `docs/claims-salience-register.md` | `tools/claims_salience_register.py` | Frontmatter "Do not edit by hand" |
| `llms-full.txt` | AEO feed generator | Regenerated; not hand-edited |
| `docs/industry-scorecard/` | Generated from `tools/industry_scorecard.data/` | Regenerated from `.data` input |

Direction note: `.data/` directories under `tools/*.data` are **generator inputs**
(authored data); the output side is the generated doc. Edit the input, regenerate the
output — never hand-edit the output.

## 5. Ownership

- Repo owner: [@anthony-chaudhary](https://github.com/anthony-chaudhary) —
  [.github/CODEOWNERS](../.github/CODEOWNERS) carries the default plus the frozen-surface rows.
- Lane roots: [dos.toml `[lanes]`](../dos.toml) declares lane names; leaf-per-directory
  partitioning in [PARTITION.md](../PARTITION.md).
- ABI human ownership: [GOVERNANCE.md](../GOVERNANCE.md) — ABI changes are additive-only
  and human-owned.

## 6. Frozen surfaces

- Hard-self core-lock globs: `internal/adjudicator/**`, `internal/abi/**`,
  `internal/corelocks/**` — declared in
  [internal/corelocks/testdata/core_locks.toml](../internal/corelocks/testdata/core_locks.toml).
  Landing an in-agent edit there raises `CORE_SELF_MODIFY` unless an independent
  maintenance witness confirms it.
- Naming collision: `SELF_MODIFY` = runtime guard-floor refusal (the agent editing its own
  kernel/config; [POLICY.md](../POLICY.md)); `CORE_SELF_MODIFY` = commit/land-time
  core-lock refusal ([dos.toml](../dos.toml)). Same word family, different moment.
- "Frozen ABI" = `internal/abi` (golden-pinned, human-owned per GOVERNANCE.md) — distinct
  from `pkg/abi`, the exported alias surface for external consumers.

## 7. CLI families

| Family | Install | Owns | First verb |
|---|---|---|---|
| `fak` | `go install github.com/anthony-chaudhary/fak/cmd/fak@latest` | Runtime product: guard, serve, agent, session, policy | `fak preflight` |
| `fak-dev` | `go install ./cmd/fak-dev` (from the checkout) | Repo development: scorecards, benches, issue/docs tooling | `fak-dev ci-preflight` |
| `dos` | `pip install dos-kernel` | Work admission: leases, truth, liveness, decisions | `dos arbitrate` |

Authority: [cli-reference.md](cli-reference.md) "Verbs" (the fak vs fak-dev split) and
[fak-vs-dos.md](fak-vs-dos.md) (FAK runs agents; DOS admits work).

## 8. Where new code goes

Compact decision tree (full route table: [CONTRIBUTING.md](../CONTRIBUTING.md),
"Choose the route for your change"):

1. Existing Go package or CLI behavior → the owning package under `internal/` or
   `cmd/fak/`; proof = a behavior test that fails before the fix and passes after it.
2. New subsystem capability → [EXTENDING.md](../EXTENDING.md): extend through leaves
   (`fak new-leaf`), avoid editing the core registry directly; architecture gate plus
   correctness witness.
3. New tool or automation → a Go sub-module under `tools/<name>/` or a `cmd/` application.
   Native Go programs only; random shell/PowerShell scripts are refused by
   `FILE_ADMISSION` (AGENTS.md hard rule).
4. Documentation → the audience route in [INDEX.md](../INDEX.md); every changed local link
   must resolve.
5. One leaf per directory: [PARTITION.md](../PARTITION.md) — how the kernel splits work
   across lanes and leaves.

## 9. Refusals & recovery

Every guard refusal names a token from the closed `[reasons.*]` vocabulary in
[dos.toml](../dos.toml). Recover, don't fight it:

- `fak recover <TOKEN>` — the concrete recovery plan when one exists.
- `dos man wedge <TOKEN> --explain` — summary, category, fix, and references.

Full protocol: [AGENTS.md](../AGENTS.md), "If the kernel refuses you".

## 10. Change protocol for this page

New boundaries land here in the same commit as the boundary itself: when a doc takes
ownership of a slice, a surface freezes, or a generated path appears, update the matching
row above in that commit. This page is a router — if a row starts duplicating its target
instead of pointing at it, the row is wrong.
