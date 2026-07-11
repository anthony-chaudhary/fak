---
title: "Borrow scout: OpenShell (+ broad deepagents re-witness) → fak (2026-07-10)"
description: >
  A /study-repo scout pass over OpenShell @ 614c8c1 (Apache-2.0; agent-sandbox governance
  kernel — a Z3/SMT-backed policy prover, default-deny egress floor, kernel-truth principal
  identity, and an accepted-risk waiver ledger) plus a BROADER re-witness of langchain-ai/
  deepagents @ c87e8fe than the prior narrow sandbox-trio pass (CONCEPT-STUDY-DEEPAGENTS,
  #4200/#4203). ~14 candidates dogfooded against fak's own seams (adjudicator, knownbad,
  harnessprofile, conformance, ifc/egressfloor, attemptbudget). Decisive signal: fak is MATURE
  on this governance axis — per-model harness profiles, injection/egress floor, accepted-risk
  ledger, schema conformance, and abstain-on-ambiguity are all PRESENT; the deepagents middleware
  re-witness is entirely PRESENT/DIVERGENT (nothing new past #4200). ONE witnessed PARTIAL:
  fak's `dogfood_manifest` anti-widening lock proves the same property as openshell-prover's
  `finding_delta` (a policy change adds no new reach) but as a BINARY CI tripwire, not a
  structured per-category reach-delta that can auto-ratify a narrow proposed grant. All borrows
  INSPIRE (Apache-2.0, Rust→Go clean-room); no bytes vendored. FILED: epic #4218
  (`epic(openshell-study)`) + leaf #4220 (the P1 reach-delta borrow). An earlier draft deferred
  filing over apparently-inconsistent `gh` output; root cause was cross-repo `gh issue list
  --search` pollution (other repos' issues with colliding numbers), resolved by pinning
  `--repo anthony-chaudhary/fak` — see "Filed".
metadata:
  type: project
---

# Borrow scout: OpenShell → fak (2026-07-10)

## What was studied

- **Primary repo:** **OpenShell** — an agent-sandbox **governance kernel** (Rust). Pinned SHA
  `614c8c164d15024ac44c709c98dcc23bb3350fa5` (`614c8c1`, HEAD dated 2026-07-10), cloned read-only
  to scratch. **Apache-2.0** (`LICENSE`) ⇒ a port is INSPIRE / clean-room reimplementation in Go;
  no source copied. Subsystem mined: `crates/openshell-prover/src/` — `model.rs` (a **Z3-backed
  reachability model** of a sandbox policy, `model.rs:30`), `queries.rs` (`finding_delta`, the
  new-vs-baseline finding diff, `queries.rs:10`), `finding.rs` (the four reachability categories +
  the **"delta empty = candidate for auto-approval; any finding = human review"** contract,
  `finding.rs:6-10`), `accepted_risks.rs` (the operator waiver ledger, 157 LoC), `credentials.rs`,
  `registry.rs` (binary registry / principal identity), `report.rs`, `policy.rs`.
- **Secondary (broad re-witness):** **langchain-ai/deepagents** @ `c87e8fe` — the *middleware*
  plane (context summarization/eviction, subagents/memory/skills, HITL, per-model harness profiles
  incl. the NVIDIA Nemotron profile), i.e. the parts the **prior** deepagents pass
  ([CONCEPT-STUDY-DEEPAGENTS](CONCEPT-STUDY-DEEPAGENTS-2026-07-10.md), epic #4200 / leaf #4203)
  did **not** cover — that pass mined only the sandbox integration trio (provider registry, HITL
  gating, dependency preflight). This pass extends the witness to the middleware; the honest
  result is that the extension is entirely PRESENT/DIVERGENT (below), so **#4200 is not reopened**.

## Method

Fan-out read of the load-bearing modules at each pin, each candidate grounded at a real
`path:line`, then dogfooded fak's own seams (raw `Grep` of `internal/adjudicator`,
`internal/knownbad`, `internal/harnessprofile`, `internal/conformance`, `internal/ifc` +
`internal/egressfloor`, `internal/attemptbudget`, `cmd/fak/guard_park.go`) before grading
PRESENT / PARTIAL / ABSENT / DIVERGENT. `fak_feature_query` overflowed the tool budget
(300–830 KB blobs) exactly as the sibling passes report; fell back to `fak_index_*` + raw grep.

## License gate

OpenShell is **Apache-2.0**; deepagents is **MIT**. The one documented borrow is a small
architectural idiom **re-implemented in Go** — `inspire`, no bytes vendored. Both licenses permit
even a byte copy with attribution; the anchors below are provenance only.

## Decisive finding

fak is **mature on OpenShell's governance axis**. The blog thesis that framed this pass — *"tune
the harness per-model, not the model"* — is already **shipped**: `internal/harnessprofile/` +
`internal/policy/harness-profiles.json` + `cmd/fak/guard_harness_profiles_test.go`, tracked under
the live epic **#1951** (universal harness profiles). Injection/egress defense, the accepted-risk
ledger, schema conformance, and abstain-on-ambiguity are all PRESENT. The **one** survivor is a
*structuring* gap: fak proves the anti-widening property as a binary CI tripwire, where
openshell-prover proves it as a **structured, per-category, per-path reach-delta** that can
auto-ratify a narrow grant.

## Witness table

| # | Candidate (source anchor) | fak witness | Verdict |
|---|---|---|---|
| **P1** | **Structured reach-delta gate on a proposed policy change** — compile a proposed sandbox policy to a Z3 reachability model (`openshell-prover/model.rs:30`), run 4 reachability queries, and `finding_delta` (`queries.rs:10`) computes which findings are NEW vs the baseline; **empty delta ⇒ auto-approvable, any new finding ⇒ human review** (`finding.rs:6-10`). Categories are typed & per-path: `credential_reach_expansion`, `link_local_reach`, `capability_expansion` (`finding.rs:14-38`) | fak's `internal/adjudicator/dogfood_manifest_test.go` is an anti-widening **lock** — "a change that silently widens the floor fails here" — but it is a **binary CI tripwire** (widened / not), not a typed per-category, per-path delta, and it **cannot auto-ratify**: it can only fail CI. fak's complain-mode (`decide.go` `Complain`, `complain_would_deny_test.go`) is admit-and-**log** (`would_deny` forensics), not prove-then-grant. No Z3/SMT reachability model over an adjudicator `Policy` change exists | **PARTIAL — ready to file (see below)** |
| C1 | Accepted-risk waiver ledger: an operator-accepted finding is suppressed so `finding_delta` re-gates only NEW reach (`accepted_risks.rs`) | `internal/knownbad/knownbad.go` — categorical `Signature(reason_class, tree_globs, failure_hash)`, `StatusRevoked`+`RevokeReason` (operator judgement), `StatusResolved` on a witness, bounded-TTL self-heal, append-to-supersede latest-row-wins | **PRESENT — no borrow.** fak's knownbad IS a categorical accepted-risk ledger with reason + witness gate |
| C2 | Per-model harness profile (tool allow/deny, prompt fragments, context thresholds) incl. the NVIDIA Nemotron profile — the blog's core thesis | `internal/harnessprofile/` + `internal/policy/harness-profiles.json` + `guard_harness_profiles_test.go`; live epic **#1951**, `docs/notes/UNIVERSAL-HARNESS-PROFILES-2026-07-01.md` | **PRESENT — no borrow.** fak converged on the blog's thesis independently |
| C3 | Default-deny egress floor; SSRF/link-local reachability as a first-class finding | `internal/ifc` (taint high-water sink-gate) + `internal/egressfloor` — egress gated on *information-flow taint*, not a static host allowlist | **PRESENT — fak ahead** (IFC taint > static reachability set) |
| C4 | Kernel-truth principal identity: bind an action to the binary via unforgeable substrate (binary registry), **abstain on ambiguity** | fak binds leases to pid+`SessionID` and treats **git history as substrate**; the adjudicator returns `VerdictDefer` (fail-to-abstain → default-deny) on unprovable cases (`decide.go`) | **PRESENT-in-discipline.** The enforcement primitive (`/proc/exe`) is Linux-only (off-axis for a portable Go binary); the *discipline* (read substrate, abstain on ambiguity) is present |
| C5 | Test-time schema conformance for structured verdict/finding records | `internal/conformance/conformance.go` + `internal/policy/flag_bypass_capfloor_conformance_test.go` + `internal/issuecontract/contract.go` + `docs/standards/*schema*` | **PRESENT — no borrow** |
| C6 | Fail-closed policy adjudication (nothing affirmatively allowed ⇒ default-deny) | `adjudicator.Policy` zero value is fail-closed; DEFAULT_DENY fold, restrictiveness lattice (`decide.go:36-64`) | **PRESENT — no borrow** |
| D1 | deepagents budget-exhaustion → synthesize a partial answer from gathered tool results | fak routes budget exhaustion to **HELD/human triage** (`internal/attemptbudget`, BlockReason+Route) or **park-then-fail-loud** (`guard_park.go`), never an unverified synthesized terminal | **DIVERGENT — no borrow.** fak deliberately distrusts an unverified worker's self-authored partial answer (verify-don't-trust) |
| D2 | deepagents rubric "witness before done": an LLM grader satisfies a goal-derived rubric before completion | fak's `dos_verify` witnesses *shipping* from git/ledger substrate — a **deterministic** oracle, not an LLM judge | **DIVERGENT — fak's witness is stronger** (deterministic, unforgeable) |
| D3 | deepagents hooks = non-blocking telemetry; gating is separate middleware | fak fuses: guard hooks CAN deny (trust enforcement at the tool-call boundary) | **DIVERGENT — deliberate** |
| D4 | deepagents non-destructive compaction / oversized-result offload with a recovery pointer | `mcp__fak__fak_context_restore`/`_spans` (content-address handles, trust-gated page-in) + `internal/ctxmmu` | **PRESENT — no borrow** |
| D5 | deepagents managed-binary auto-download (ripgrep) | one-static-binary philosophy; already dropped by #4200 (C5) | **off-axis — no borrow** |

## The one borrow — FILED as #4220 (INSPIRE from OpenShell Apache-2.0 @ 614c8c1)

> **Filed:** leaf **#4220** under study epic **#4218**. The spec below is the issue's source of truth.

- **Title:** `feat(adjudicator): structured per-category reach-delta over a Policy floor change (borrow openshell-prover finding_delta)`
- **Labels (as filed):** `priority/P2`, `research`, `rsi` — the study-epic house convention, mirroring #4200/#4203.
- **Verdict:** PARTIAL. `inspire`.
- **Seam:** `internal/adjudicator/dogfood_manifest_test.go` (the anti-widening lock, today a binary
  pass/fail) + `internal/adjudicator/decide.go` (`Policy` floor) + complain-mode
  (`complain_would_deny_test.go`) + `internal/knownbad` (the accepted-risk waiver, already present).
- **Borrow:** compute a **typed, per-category, per-path reach-delta** for a proposed `Policy` change
  (categories analogous to openshell-prover's: *new-tool-permitted*, *new-egress-host/method*,
  *new-write-tree/self-modify-reach*), so (i) the dogfood lock reports **which** category widened
  instead of a bare boolean, and (ii) an **empty delta** becomes an auto-ratifiable narrow grant on
  the complain-mode → floor-promotion path (**propose ≠ ratify**: the agent authors the narrowest
  rule; a deterministic referee auto-approves iff the delta is empty), while any new finding routes
  to human review — reusing `knownbad`'s accepted-risk suppression so an accepted widening does not
  re-gate. fak already has every piece (complain-mode = propose, dogfood-lock = the delta primitive,
  knownbad = the waiver); this composes them into openshell-prover's structured-finding-delta shape.
- **First checkable step:** extract `internal/reachdelta` (`Category`, `Finding{Category, Path}`,
  `Delta(base, proposed Policy) []Finding`), refactor `dogfood_manifest_test.go` to assert on
  `Delta(...) == ∅` (byte-identical pass/fail today), then add the empty-delta auto-ratify path
  behind a default-off flag. `go test ./internal/adjudicator` stays green.
- **NOT** a formal-methods borrow: fak does **not** vendor Z3/SMT — the delta is a structural diff
  over the declarative `Policy` (allow/deny/prefix/egress/self-modify sets), not an SMT proof. The
  borrow is the *finding-delta discipline* (typed categories + empty-delta-auto-approve), not the
  solver.

## Filed + not filed

- **P1 FILED — epic #4218, leaf #4220.** An earlier draft deferred filing over apparently-inconsistent
  `gh` output: `gh issue list --search` showed *"#287 OpenShell fail-closed policy attestation"* while
  `gh issue view 287` resolved to an unrelated *"Vulkan Backend Optimization [CLOSED]"* (same for #10,
  #9). **Root cause:** the unpinned `--search` was returning **cross-repo** results (other repos'
  issues with colliding numbers — two different #30, a duplicated #1951), while `gh issue view` hit the
  local `origin`. Pinning `--repo anthony-chaudhary/fak` made list/view consistent, confirmed the
  sibling issues (#4200/#4203/#3940/#1951) really exist here, and showed `openshell in:title` is
  **empty** in the fak repo — so the study is non-duplicative (the "Epic: openshell" / "fail-closed
  policy attestation" hits were other repos). Active gh account `claude-ai-netra` holds push+triage.
- **C4 kernel-truth `/proc/exe` identity** — enforcement primitive is Linux-only; held (no portable
  fak seam), the discipline is already present via `VerdictDefer` abstain + git-substrate leases.
- **Over-tooling / needless-planning-tool detector** (a deepagents/evals idiom: assert ≤N tool
  calls, catch a cargo-culted planning tool) — `cmd/fak/antipatternscore.go` shows no tool-call-count
  seam, but no crisp fak *mis-behaving* seam was located this pass; held for a future pass per the
  field-borrow "no file:line seam, no issue" rule.

## Companions / cross-links

Same genre as the sibling foreign-repo study passes:
[deepagents](CONCEPT-STUDY-DEEPAGENTS-2026-07-10.md) (#4200/#4203 — narrow sandbox-trio pass this
note extends), [claude-code](CLAUDECODE-STUDY-2026-07-10.md) (#4040),
[puppetmaster](study-puppetmaster-borrow-scout-2026-07-10.md) (#3940),
[kernel-design-agents](study-kernel-design-agents-borrow-scout-2026-07-10.md). A crawl is not a
borrow; a study is not a ship — this note + epic #4218 / leaf #4220 grow backlog; ancestry
(`Fixes #4220` on trunk) resolves the borrow later, by a different worker.
