---
title: "Borrowing from google/mcp-security: a 12-axis witness against fak (2026-07-10)"
description: "Study of google/mcp-security @ fb807e9 (Apache-2.0) — the negative space of a guard kernel: all enforcement lives in perimeter + prose, none in-band. 12 distilled borrows witnessed on-axis against fak: 3 PRESENT, 2 DIVERGENT, 6 PARTIAL (filed), 1 adjudicated-not-filed."
---

# Borrowing from google/mcp-security: a 12-axis witness against fak (2026-07-10)

Source studied: **`github.com/google/mcp-security`** @ **`fb807e9`** (Apache-2.0) — Google's official MCP
servers + agents for their security stack (Google SecOps/Chronicle, SOAR, GTI/VirusTotal, SCC), plus a
`run-with-google-adk` reference agent. The exercise, as in the [kvcached](BORROW-KVCACHED-STUDY-2026-07-10.md)
and [vLLM M2](CONCEPT-STUDY-VLLM-M2-2026-07-10.md) passes: harvest every *mechanism* that looks like a guard
move, distil to repo-agnostic axes, then **witness** each on-axis against fak (dogfooding `fak_feature_query`
+ raw grep/read), classifying **PRESENT / PARTIAL / ABSENT / DIVERGENT** with real fak seams — never
rubber-stamping. Witness default was **PRESENT** (adversarially try to disprove the borrow first) as a
false-ABSENT guard: filing something fak already has is the failure mode.

## License gate

google/mcp-security is **Apache-2.0**; fak is **Apache-2.0** — integration would be license-legal, but every
candidate is **Python → Go**, so all borrows are **`inspire`** (clean-room reimplementation, source cited).
**No bytes vendored.**

## Headline: this repo is the *negative space* of a guard kernel

The most valuable finding is not any single borrow — it is **where the enforcement isn't**. Across ~295 SOAR
modules, four MCP servers, and the ADK reference agent, google/mcp-security delegates essentially **all**
safety to two places fak deliberately refuses to trust:

1. **The perimeter, not the call.** Real controls live *outside* the agent loop: Cloud Run / IAM, an external
   Model-Armor-style sidecar, org-policy allowlists, `--integrations` deny-by-default tool exposure, and
   pre-side-effect **resource-size refusals** (1000 logs / 10 MB / 50 MB). Structural, but all at the edge.
2. **English prose, not verdicts.** The in-loop "guardrails" are sentences. System prompts assert
   *"All authentication actions are automatically approved"* and *"priority is only an initial indicator"* —
   enforced by nothing. Destructive SIEM operations (close case, run action) have **no confirmation gate**,
   only a docstring warning + a **model-chosen `force` boolean**. The ADK `before_model`/`before_tool`
   callbacks that *look* like guardrails are pure **cost/UX**: trim history by user-turn, memoize
   `list_tools`, filter the tool set. None adjudicate safety.

That is precisely the enforcement fak makes **structural and in-band** (typed verdicts on the call/result,
provenance stamps, closed refusal vocabulary). So the repo reads as a validation of fak's thesis — *detection
and prose are evadable; structure is not* — and as a **map of the seams where even fak still leans on
convention** where a rung would be stronger. The 6 PARTIALs below are exactly those seams.

## Method

N reader agents over the load-bearing modules produced **89 raw candidate mechanisms**; deduplication (6×
"memoize list_tools", 5× "trim history by user-turn", 4× "closed-enum arg guard", 3× deny-by-default, …)
collapsed them to **12 distinct borrows**. Each was witnessed by an independent adversarial prover against
fak @ `3f61864`.

## Tally

12 axes: **3 PRESENT** (fak already had it, often exceeded), **2 DIVERGENT** (fak deliberately differs —
correct as-is), **6 PARTIAL** (real on-axis gap → filed), **1 adjudicated-not-filed** (capability→tool
fallback ladder; out of scope for a guard behind the harness).

## Axis-by-axis

| # | Axis (google/mcp-security mechanism) | source_ref @ fb807e9 | Verdict | fak seam | Disposition |
|---|---|---|---|---|---|
| B1 | allow-with-**edit** third verdict (rewrite call args, proceed) | `run-with-google-adk/.../callbacks.py:64` | **PRESENT** | `abi/types.go:208` `VerdictTransform` + `kernel.go:397` mutate `c.Args` → dispatch; test `kernel_test.go:381` | recorded (exceeded: typed sum verdict, not a magic `None`) |
| B3 | legible truncation truth-bit ("more exists") | `server/scc/scc_mcp.py:147` | **PRESENT** | `agent/anthropic_elide.go:47` marker w/ exact omitted count + `_stale.go:331` restore handle | recorded (exceeded: re-fetch, not just a bit) |
| B5 | scoped/curated child env at spawn | `.../ae_remote_deployment_sec.py:48`; `cloudrun_deploy_run.sh:35` | **PRESENT** | `policy.StripInheritedSecrets` (`guard_child.go:709`) + default-deny `InheritedTable` | recorded (exceeded: allow-list subset + secret-refs) |
| B6 | tool_set_name stamp + snake_case de-collision | `.../extensions.py:59`; `secops-soar/.../utils.py:55` | **DIVERGENT** | harness emits `mcp__server__tool`; `gateway/ctxfootprint.go:112` parses it; fak reserves `mcp__fak__` | recorded (aggregator's job, one layer up) |
| B10 | non-leaking error envelope (fixed literals, no raw exception) | `server/gti/gti_mcp/utils.py:70` | **DIVERGENT** | `gateway/http_upstream_error.go:16` scrubs the *external* boundary; tool-result raw error kept **by design** for self-correction | recorded (scrub has no payoff on the local tool-result plane) |
| B2 | principal bound to session, args can't re-declare identity | `run-with-google-adk/.../callbacks.py:83` | **PARTIAL** | anti-spoof core present (`vdso.MetaPrincipal` host-set), but no adjudicator consumes it for authz & args' `user=` passes through (`ifc`/`adjudicator`) | **FILE** (xref #2412 #3953 #2397) |
| B4 | reject bad enum arg + echo the **legal set** as repair hint | `server/gti/gti_mcp/tools/files.py:176` | **PARTIAL** | `Meta["fix"]` seam + closed sets both exist but never wired; no enum rung; fix hints are static text (`adjudicator/decide.go:1095`) | **FILE** (xref #4047) |
| B7 | partial-reveal secret mask (recognizable prefix for verify) | `run-with-google-adk/run-adk-agent.sh:22` | **PARTIAL** | `secretgate` reveals zero bytes; `Finding` has digest+len but no prefix hint (`secretgate.go:71`) | **FILE** |
| B8 | media-type / shape contract on the result | `server/gti/gti_mcp/tools/files.py:411` | **PARTIAL** | shape-sniff primitive (`headroom.Detect`) + provider-boundary check exist, but no tool-result→context shape contract; HTML login page below marker threshold enters as data | **FILE** (under epic #1217) |
| B9 | guard-injected conservative default bounds (limit/hours_back/truncate) | `.../security_events.py:33`; `gti/.../utils.py:20`; `files.py:279` | **PARTIAL** | `ArgMaxBytes` only *denies* over-length strings; no clamp-forward, no item/time default injection (`adjudicator/decide.go:153`) | **FILE** |
| B11 | benign-bloat structural compaction (empty-prune + whitelist projection) | `gti/.../utils.py:119`; `scc_mcp.py:118`; `utils.py:141` | **PARTIAL** | `ctxmmu` is all-or-nothing opaque page-out; readable digest arm is default-inert line-truncation (`ctxmmu/mmu.go:183`, `wirescreen/digester.go:80`) | **FILE** (under epic #1217) |
| B12 | capability→tool indirection w/ remote-first/local-fallback | `.../google-secops/TOOL_MAPPING.md:11` | **adjudicated / not filed** | `selfquery` resolves capability-cards→ready calls via a resolver (`selfquery.go:417`); the missing piece is only the aggregator-style fallback ladder | not filed (out of scope, like B6) |

## Filed (all leaves, provenance = this note)

- **B2 — actor/identity provenance at the tool-call gate.** `feat(ifc)`: stamp the session-authenticated
  **actor** onto the `ToolCall` and refuse any call whose args re-declare/override the principal
  (identity-spoof gate — the actor-provenance companion to the r20 data-taint stamp). fak already carries a
  host-set principal for *cache isolation*; the gap is that **no adjudicator consumes it for authz** and an
  injected `user=admin` in a tool's args is passed through untouched. Cross-ref #2412 (principal labels on
  the *inbound* plane), #3953 (login-time seat↔identity hijack), epic #2397 (agentgraph principal-tagged
  messaging) — all adjacent planes; this is specifically the **tool-call-args** plane.
- **B4 — echo the legal value set on a bad enum arg.** `grammar/arg rung`: on rejecting an unknown
  closed-vocab argument, echo the full legal set into the refusal `Meta["fix"]` as a did-you-mean
  self-repair hint. The self-repair-hint seam and the closed sets (`grammar` enum, `DurabilityVocabulary`,
  `abi.ReasonNames`) both exist but are never wired; every fix is static hand-authored text. Cross-ref #4047
  (closed-vocab label clamp at the issue actuator — same family, different actuator).
- **B7 — recognizable-prefix secret hint for operator verification.** `secretgate`: add an operator-facing
  first-N-bytes + length hint to `Finding`/redaction mark so a human can *verify which* credential was seen,
  gated so it never re-enters model context and re-screens clean. Distinct from the zero-byte quarantine
  sentinel (which is correct as the default).
- **B8 — result shape/media-type contract.** Add a `ResultAdmitter` rung (in `normgate`, reusing
  `headroom.Detect` + a new `abi` result-shape reason) that detects a tool result of the wrong shape (an
  HTML error/login page where structured data was contracted) and converts it to a structured `{error}`
  before it enters context. **Child of epic #1217 (ctxsafety).**
- **B9 — clamp-forward default bounds on unbounded tool args.** `adjudicator`: a clamp-forward arg-bound
  kind (emit `VerdictTransform` injecting a conservative default when a numeric bound is missing/over a
  ceiling) so an unbounded historical/flood scan can never be the *default* call — inject `limit`,
  `hours_back`, truncate over-long fields. `ArgMaxBytes` only *denies* over-length strings today.
- **B11 — structure-aware benign-result compactor.** `ctxmmu`: a deterministic, default-available compactor
  (recursive empty/null-field prune + positive-whitelist schema projection) that keeps a small **readable**
  projection in context, instead of the current all-or-nothing opaque page-out or a `<4KB` verbatim
  passthrough. **Child of epic #1217 (ctxsafety).**

## Recorded present / divergent (not filed)

- **B1 (transform-and-proceed)**, **B3 (legible truncation)**, **B5 (scoped child env)** — PRESENT and in
  each case *exceeded* the borrow (typed sum verdict; re-fetch handle; allow-list-subset + secret-refs).
- **B6 (tool_set_name + de-collision)** — DIVERGENT: the harness is the MCP aggregator and emits
  collision-resistant `mcp__server__tool` names; fak sits behind it, parses the server segment, and reserves
  `mcp__fak__`. Both sub-properties are satisfied one layer down and correctly relied upon.
- **B10 (non-leaking error envelope)** — DIVERGENT: fak already returns fixed-literal scrubbed bounded
  errors on the *external* serve boundary (`upstream4xxStatus`, `scrubForbiddenDetail`); on the *local*
  tool-result plane it intentionally preserves the raw exception because the wrapped child needs it to
  self-correct, and injection/secret leakage there is already covered by `normgate`/`secretgate`.

## Parked (not filed)

- **B12 (capability→tool fallback ladder):** `selfquery` already resolves capability cards to ready-to-run
  calls through a resolver seam; the only missing sub-property is the *aggregator-style* remote-first/
  local-fallback ladder, which — like B6 — is a job for the layer that mounts multiple concrete backends,
  not for a guard sitting behind a single harness. Revisit only if fak ever fronts multiple tool backends.

## Dedup notes

- **B2 is not #2412/#3953/#2397.** #2412 labels *inbound messages* by principal; #3953 catches a seat↔identity
  mismatch *at /login*; #2397 is the agentgraph messaging epic. B2 is the **tool-call adjudication** plane —
  refuse args that re-declare the principal — a distinct seam; cross-referenced, not duped.
- **B4 is not #4047.** #4047 clamps a hallucinated **issue label** to a closed vocab at the issue-edit
  actuator; B4 echoes the legal set of a **tool argument** enum in the call refusal. Same "closed-vocabulary"
  family, different actuator; cross-referenced.
- **B8 + B11 nest under epic #1217 (ctxsafety)** — the value-preservation floor — as new children, not
  new epics.
- **B7 / B9** returned no adjacent open issue on a sharp title/body search; filed clean.
