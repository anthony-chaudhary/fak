# Concept study: professorpalmer/Puppetmaster @ `5de5cd5` — witnessed borrows (second pass)

**Date:** 2026-08-08
**Source:** <https://github.com/professorpalmer/Puppetmaster>
**Pinned:** `5de5cd5858e891503a11df8303a619dcfb0a13f4` — *"Release v1.21.13: passive
rate-limit harvest and preemptive admission"* (2026-08-05)
**License:** MIT (© 2026 Cary). fak is Apache-2.0, so MIT→Apache vendoring would be
permitted with attribution — but the source is Python and the value is the technique, so
**every row below is routed INSPIRE (clean-room)**. Nothing was copied.
**Epic:** #5961. **Filed:** #5962, #5963, #5967, #5968, #5974, #5975, #5976, #5977, #5979, #5981.

`Companions:` [`study-repo`](../../.claude/skills/study-repo/SKILL.md) ·
[`field-borrow`](../../.claude/skills/field-borrow/SKILL.md) · epic #5961 ·
first pass `study-puppetmaster-borrow-scout-2026-07-10.md`

## This is a second pass — what it deliberately does not re-derive

The first pass (2026-07-10) pinned `9597e5b` (v1.18.0), opened ~18 modules, and filed one
borrow (#3940, MMR redundancy suppression in `internal/recall`). Everything it resolved
PRESENT stays out of scope here and was not re-examined: all-1h cache placement,
tool-offload + savings, budgeted skill injection, the egress/SSRF floor, the fencing lease +
liveness pair, the preflight probe, mid-stream cancellation, forced-tool structured output,
and the multi-worker stitch.

This pass mines the **v1.18.0 → v1.21.13 delta** plus the roughly 100 modules the first pass
never opened.

## Their worldview (reconstructed from their own docs)

- **A supervisor over vendor CLIs, not a framework for authoring an agent**
  (`docs/COMPARISON.md`: "a supervisor that drives the agent CLIs/SDKs you already pay
  for"). Their scarce resource is subscription quota and money; fak's is trust and
  determinism.
- **"Agents should not share transcript history. They should share durable state."**
  (`docs/WHY.md`) — hence a WAL-SQLite layer every process reads. This single choice is why
  their rate-limit knowledge survives a process boundary and fak's does not.
- **Their stated edge is mixed workloads and long horizons, explicitly not winning one hard
  implementation against a strong steered agent** (`AGENTS.md`), with a swarm named a
  non-goal for "one coupled feature."
- **The load-bearing one.** They are single-author and say so ("young and single-author
  (daily-driver beta)"), while publishing hard numbers with a `bench/` receipt per claim
  (`docs/CLAIMS.md`: 35.1% router savings; 40 follow-up queries at $0.00). A solo author who
  publishes numbers builds **mechanical anti-self-deception** into the tool: a KPI fenced at
  the write site so it cannot count itself, a savings report that debits its own overhead, a
  freshness probe that refuses to say "fresh" when it could not afford to check, a citation
  verified by opening the file.

That last point is the thesis of this study, and it predicted where the gaps were. fak's
kernel is the part that **does not believe the agents**. Puppetmaster's ledger is the part
that **does not believe itself**. Adjacent, not identical — and six of the ten filed rows
came out of the second discipline.

## Fan-out coverage

Eight parallel readers over the never-opened subsystems, ~140 files at the pinned SHA:

| Reader | Modules opened | Self-reported unopened |
|---|---|---|
| ratelimit-resilience | `rate_limit_headers`, `rate_limit_state`, `provider_circuit`, `provider_health`, `failure`, `invocation_gate` + 3 test files | none in scope |
| accounting-audit | `audit`, `execution_provenance`, `memory_cost_log`, `reads_log`, `artifact_bounds`, `tool_batch` | none (flagged `receipt.py` as adjacent) |
| quality-control | `claim_conflicts`, `execution_provenance`, `acceptance_criteria`, `quality`, `validation` | none |
| routing-catalog | `router` (988L), `model_registry`, `orchestrator` | `static_catalog.py` partial by choice |
| mcp-surface-secrets | `mcp_server`, `job_brief`, `secret_entry`, `fs_permissions`, `diagnostics`, `mcp_registry` | none |
| runid-worker-lifecycle | `run_id`, `worker_runtime`, `swarm_launch`, `win_process` | none (`lifecycle.py` is not worker lifecycle) |
| codegraph-context | `codegraph` (+ repair/usage/index_runner), `prewalk`, `affected` | none |
| gates-evaluators | `gates` (840L), `rules` (842L), `evaluators` (447L) | none |

**Completeness critic: DID NOT RUN.** The critic agent and the quality-control witness agent
both died on upstream 429s with a provider-named wait longer than the client could hold
(`agents_error: 2` of 17). The quality-control axes were re-witnessed by hand and are sound
— rows 7, 8, 12 below are my own reads, not a subagent's. But the *independent* critic pass
that the skill's depth floor requires was not completed, so the honest status of this study
is **deep but not critic-attested**. Coverage above is per-reader self-report over an
assigned list, which is weaker. Re-running the critic against this note is the next
checkable step for the study itself.

## Candidate table

Axis-level verdicts. A PRESENT here means fak covers *the specific axis*, not the umbrella.

| # | Borrow | Source `path:line@5de5cd5` | AXIS | Their-worldview reason | Witness (fak seam) | Route | Filed |
|---|---|---|---|---|---|---|---|
| 1 | Freshness probe returns `unknown` when it could not afford to check | `codegraph.py:1480` | not-checked ≠ fresh | publishes numbers under one name; a timed-out "fresh" corrupts them | **ABSENT** — `internal/devindex/freshness.go:66,71-73` (unreadable source ⇒ no finding ⇒ green); `conceptcatalog/freshness.go:108,112` returns `Fresh:true` *with* the error; `cmd/fak/concept.go:52` prints `{"fresh":true}` then exits 1 | inspire | **#5962** |
| 2 | A relay's borrowed 401 reclassified before durable health is written | `failure.py:200` | whose 401 is it — relay vs local credential | multi-key rotation over paid relays is the product | **PARTIAL** — `internal/accounts/orgwall.go:143` returns `NeedsLogin` unconditionally, zero body inspection, while the 403 arm below runs two fences; relays are live at `internal/apihostprobe/apihostprobe.go:28-63` | inspire | **#5963** |
| 3 | Durable passive rate-limit harvest ledger | `rate_limit_state.py:379`, schema `:38`, coalesce `:223` | quota knowledge survives the process, at zero extra requests | durable shared state, not shared transcripts | **PARTIAL** — `accountobs.go:265-327` parses the full family triple; only non-test consumers are `Report` (`:404`) and `PrometheusText` (`:470`). `cmd/fak/guard.go:895` wires the observer but only to a one-shot goalpark latch (`:859-874`) | inspire | **#5967** |
| 4 | Preemptive admission over harvested remaining quota | `rate_limit_headers.py:51` → `rate_limit_state.py:437` | preempt on an *advertised* wall, not only a witnessed one | flat-rate pool; a spent 429 is a wasted seat | **PARTIAL** — every fak preemption fires off a witnessed wall (`guardrotate.go:150-161` after a live 429; `tools/launch_admission.py:181-191` off a caller bool). `cmd/fak/accounts_headroom.go:47-48` says it outright: "NOT a continuous remaining-quota number" | inspire | **#5968** |
| 5 | A displaced lease holder discards its own finished work | `worker_runtime.py:102` | suppression at the *write boundary*, not store-side rejection | several processes legitimately hold one task across a reclaim | **PARTIAL** — `internal/safesync/writer_lease.go:289` just `return`s and tells nobody; also `internal/leaseref/fence.go:99` | inspire | **#5974** |
| 6 | The memory feature debits its own injection tokens | `memory_cost_log.py:5` | the feature's cost appears in the report claiming its savings | a savings % whose denominator omits the feature's cost is marketing | **PARTIAL** — `internal/memvaluescore/score.go:331` already has the per-injection ledger row and `:91` weights it, but **every term is a credit**; no token or cost column exists. fak's gross/marginal/net sweep (`cachevaluereport/fleet_benefit.go:113-124`, #2807) reprices the numerator only | inspire | **#5975** |
| 7 | Mechanical source-verification of a cited `path:line` | `claim_conflicts.py:400`, out-of-range `:433` | a fabricated anchor caught without a model, with ambiguity never read as a lie | a confident finding citing a nonexistent line is their costliest failure | **ABSENT** — `internal/claimcheck` grades rhetoric (Q1–Q6), never opens a file. `codelint/packs.go:229` and `dispatchaudit/signatures.go:131` parse/match `path:line` but verify nothing. `dos_citation_resolve` is legal cases | inspire | **#5976** |
| 8 | Finding-reuse keyed on working-tree bytes, fail-closed | `validation.py:297` | reuse honesty on a *dirty* checkout | agents work over a tree being edited under them | **PARTIAL** — fak owns the idiom for KV prefixes (`cachemeta/sysprompt_fingerprint.go:65,117`) and bench grids (`bench/fanrun.go:177`), never for findings | inspire | **#5977** |
| 9 | Oracle gate: `available` separate from `passed` | `gates.py:600` | "judge absent" vs "judge answered unreadably" | an unconfigured laptop must not brick; garbage must not approve | **PARTIAL** — `internal/safecommit/safecommit.go:889` collapses an unparseable verdict into `ReviewUnavailable`, which is a no-op pass | inspire | **#5979** |
| 10 | Rank plan-billed seats by nominal price, bill at $0 | `model_registry.py:134` | cheapest-sufficient *inside one flat-rate pool* | every option is already paid for, so real dollars rank nothing | **ABSENT** — `internal/fleetaccounts/route.go:106` `routeRank` has no cost term at all (weight, load, product, tag) | inspire | **#5981** |

### Earned dismissals — PRESENT on-axis

| Borrow | Source | fak covers it at |
|---|---|---|
| Explicit unknown, never a fabricated zero | `execution_provenance.py` | pervasive house rule: `accountobs.go:13`, `assumecheck/registry.go:10`, `answershape/oai_conformance.go:98`, `balance/balance.go:140`, ~14 more |
| Cost-drift denominator restricted to the reconciled subset | `audit.py:122` | `internal/gateway/metrics.go:89`; corroborated `cachevaluereport/devsession.go:32,111` |
| Never re-open an offload path when shrinking to a byte cap | `artifact_bounds.py:417` | `internal/ctxmmu/mmu.go:378` (`PageIn`), pointer stub `:128` |
| Terminal status derived from artifacts, vetoing a self-reported COMPLETE | `worker_runtime.py:305` | `internal/dispatchaudit/dispatchaudit.go:266` |
| Windows descendant tree-kill | `win_process.py:31` | `internal/procguard/collect_windows_native.go:256` |
| Typed plan-time refusal before any task row is created | `orchestrator.py:1757` | `internal/agent/spawn_place.go:192` |

### Earned dismissals — DIVERGENT, with the tradeoff named

- **Affected-test selection** (`affected.py:4`). They *refuse* to infer which tests cover a
  change and require declarative glob rules, because they must serve a repo in any language.
  fak is Go-only and therefore owns the real import graph including test imports
  (`cmd/fak/affected.go:1-27`). Adopting their design would trade a sound graph for a
  hand-maintained convention — a strict loss for fak's user, a necessity for theirs.
- **`Retry-After` may never preempt** (`rate_limit_headers.py:51`). Correct for them,
  unnecessary for fak, and the reason is the key shape: their `admission_key` is
  credential-free, so honouring a per-key `Retry-After` would idle the whole pool. fak keys
  cooldowns at the credential bucket (`internal/accounts/accounts.go:109-132`
  → `uuid:`/`tok:`/`apikey:`), so `cooldown.go:188` → `:162` cools exactly the seat that was
  throttled. This was **filed as a gap and then retracted** on the second witness pass — see
  the correction on #5968.
- **Half-open breaker probe released back to OPEN** (`provider_circuit.py:374`) and
  **zombie-sweep on append** (`mcp_server.py:246`) — fak solves both structurally rather
  than by bookkeeping, via `internal/accounts/canary.go:55-75` and `internal/flock` (lease
  expiry, not a released flag) and `cmd/fak/guard_split.go:391`.

## Corrections this pass produced

- `docs/notes/2026-07-04-oauth-ratelimit-cache-read-finding.md:49` claims
  `UpstreamResponseObserver` "is assigned only in a test" and that `accountobs` has zero
  production callers. **Now stale** — the observer is assigned at `cmd/fak/guard.go:895`.
  The accurate residual gap is narrower and is what #5967 files: the seam is wired, but only
  to a one-shot latch, so nothing accumulates and nothing outlives the process.
- #5968 was filed claiming fak lacks the `Retry-After` rule, then corrected in-thread once
  the credential-bucket keying was verified. Recorded here because the retraction is the
  useful artifact, not the original claim.

## Unmined PARTIALs — witnessed, not filed

Real gaps with a named seam, left unfiled this pass to keep the epic shippable. Each is a
candidate for a follow-on: runtime-owned monotone-down ratchet baseline
(`gates.py:327` → `internal/hooks/gate_godfile.go:110`); judge capability floor derived from
the implementer's routed model (`gates.py:496` → `internal/modelroute/audit_identity.go:124`);
job-start-frozen rubric epoch (`gates.py:680` → `internal/modelroute/crossaudit.go:147`);
model-free anchor battery gating an evaluator promotion (`evaluators.py:396` →
`internal/modelroute/crossaudit_calibration_run.go:26`); write-site KPI domain fence
(`reads_log.py:31` → `internal/metrics/provider_cache.go:9`); parallel-safe tool-batch
segmentation (`tool_batch.py:168` → `internal/agent/loop.go:584`, today a sequential
executor); merge-on-write for the fleet-shared cooldown store (`rate_limit_state.py:223` →
`internal/accounts/cooldown.go:300`, where a concurrent writer can drop a peer's cooldown).
