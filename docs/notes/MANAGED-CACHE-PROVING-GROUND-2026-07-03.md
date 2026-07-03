# The managed-cache proving ground: real guarded sessions as the regression floor (2026-07-03)

**The goal this serves:** every managed-cache lever (epic #1844's family) must be
provable on REAL traffic — not on the frozen `tau2-smoke` replay — before it can
claim a saving or flip default-on. fak's durable ledgers already record that real
traffic: every `fak guard -- claude` exit appends to
`docs/nightrun/cache-savings.jsonl`, every MCP serve exit to
`docs/nightrun/gateway-usage.jsonl`, every `fak run` kernel session with KV reuse
to `docs/nightrun/cache-value.jsonl`. This note records the durable spine that
turns those ledgers into a reusable proving ground, its first live reading, and
the follow-on work parked as **contract-validated, ready-to-file issues** (the
filing session's guard policy refuses direct `gh issue create`; any authorized
session or producer can file them verbatim — same convention as the 2026-07-02
worker-unit conservation backlog note).

## What shipped (2026-07-03)

- **`tools/managed_cache_proving_ground.py`** — read-only, stdlib-only,
  deterministic (the report is a pure function of the ledger bytes; `as_of` is
  the max row timestamp, never the wall clock). Three jobs:
  1. **VALIDATE** every ledger row against its contract: exact schema string,
     closed mechanism/session_type vocabulary, non-negative counters, and the
     Track-2 savings identity `saved = 0.9·cache_read − 0.25·cache_creation`
     (`internal/cachevaluereport/track2.go` `providerSavedTokenEquiv`).
  2. **FOLD** each managed-cache concept to a rung on a closed evidence ladder:
     `UNIMPLEMENTED(0) → UNWIRED(1) → CHANNEL_READY(2) → SILENT_ZERO(3) →
     EVIDENCED(4)`. Rung upgrades auto-detect where possible — e.g. the moment a
     `ttl`-bearing counter key ships in a gateway-usage row, C6 climbs past
     UNWIRED with no tool change.
  3. **RATCHET** (`--check`) against the pinned baseline
     (`tools/managed_cache_proving_ground.data/baseline.json`) with a closed
     regression vocabulary: `SCHEMA_DRIFT`, `REGRESSION_ROW_COUNT` (an
     append-only ledger shrank), `REGRESSION_RUNG` (a concept fell down the
     ladder), `REGRESSION_VIOLATIONS`. Counts may only grow, rungs may only
     climb, violations may only shrink — concurrent sessions appending rows keep
     it green; a ledger rewrite or a lever losing its durable witness turns it
     red.
- **`tools/managed_cache_proving_ground_test.py`** — fixture ledgers drive the
  row contracts, the rung fold, and every ratchet trip; a live smoke folds the
  committed ledgers so the evidence can never silently stop parsing.
- **`make cache-proving`** — wired into `make ci` next to `scorecard-ratchet`.
  Re-pin after an intentional rung climb with `--write-baseline`.

## First live reading (as of 2026-07-03T11:54Z)

59 cache-savings rows (all `guard`, all `provider_prompt_cache`, 0 formula
mismatches), 400 gateway-usage rows (all `serve`, 0 guard), 9 cache-value rows
(all `run` kernel sessions). Row-contract violations: 0.

| rung | concept | reading | next step |
|---|---|---|---|
| 4 EVIDENCED | `provider_prompt_cache_passthrough` | 59 rows, ~165M net token-equiv | grow the population; every guard exit appends |
| 4 EVIDENCED | `kv_prefix_reuse` (Track 1) | 3,441 reused tokens, kernel sessions only | structurally 0 on the guard proxy path |
| 3 SILENT_ZERO | `compaction_shed` (#1407) | 59 provider rows prove the writer ran at every exit; 0 shed rows = a witnessed zero (anchor starvation) | de-starve anchors; the first real shed row auto-climbs this |
| 2 CHANNEL_READY | `guard_usage_plane` (#1601) | counter family exists; serve exits write it; guard exits do not | parked issue B below |
| 1 UNWIRED | `ttl_upgrade_1h` (C6, #1614) | lever shipped (`cmd/fak/guard_managed_cache.go`); only witness is the in-process `fak_gateway_cache_ttl_upgrade_total` — no durable field | parked issue A below |
| 1 UNWIRED | `breakpoint_placement` (#1603/#806) | witnessed in test only; on real rows fak-placed is indistinguishable from client passthrough | parked issue C below |
| 0 UNIMPLEMENTED | `uncached_remainder_shrink` (C7) | no lever in the tree | implement + register as ablation feature + mechanism in one change |

This is the same population the FAK-GUARD-CACHE-VALUE-SHARE-2026-07-01 note
proved "fak share = 0.0000% on guard" against — the proving ground makes that
baseline re-derivable on demand and regression-gated, and gives every lever
above a rung it must visibly climb.

## Scope fences

- The proving ground grades **witness reachability**, never savings quality: a
  rung-4 concept has real evidence rows, not a proven delta. Deltas are the
  ablation harness's job (#1844 C2/C3); this spine is the population and the
  floor those arms diff against.
- The share/attribution math stays in `fak cachevalue report` — the spine
  validates and counts, it does not re-price.
- `gh` filing is guard-refused in this session class, so the follow-ons are
  parked below with contract verdicts instead of filed numbers.

## Ready-to-file issues (contract verdict: `ready` — see filing instructions)

Filing: `gh issue create --title <title> --body-file <body>` with labels
`enhancement` + `prompt-caching`, then link from epic #1844. Bodies are verbatim
below; re-validate any edit with `fak issue contract --from-issues` before
syncing.

---

### Issue A — `feat(gateway): persist managed-cache TTL-upgrade outcomes into the gateway-usage ledger (C6 durable witness)`

**Managed cache** · the C6 lever fires invisibly: its only witness dies with the process.

#### Parent context
Epic #1844 (C6) / #1601 usage-plane epic. The managed-cache 1h TTL upgrade shipped (`cmd/fak/guard_managed_cache.go`, `agent.UpgradeAnthropicStableCacheTTL1h`, 8b618eec) gated to API-key-billed Anthropic sessions, but its only witness is the in-process `/metrics` counter `fak_gateway_cache_ttl_upgrade_total` (`internal/gateway/metrics.go` `ttlUpgrades`).

#### Current state
`internal/gatewayusageledger/ledger.go` `Counters` carries the full served-turn counter family but no TTL-upgrade fields, so a real managed-cache-active session leaves zero durable evidence. The managed-cache proving ground (`tools/managed_cache_proving_ground.py`) therefore holds `ttl_upgrade_1h` at rung 1 UNWIRED; its probe auto-climbs the rung the moment any `ttl`-bearing counter key appears in a gateway-usage row.

#### Why now
C6 is the first fak-authored wire-side lever that can move the guard-path fak share above the proven 0.0000% (FAK-GUARD-CACHE-VALUE-SHARE-2026-07-01). Until its outcomes persist, no sweep row, no share movement, and no default-on decision (#1844 acceptance) can bind to real traffic.

#### Working spine
Mirror the metrics bucket into the durable row: add `cache_ttl_upgrades_upgraded` plus per-refusal-reason counters (the closed `agent.TTLUpgradeReason*` vocabulary) to `gatewayusageledger.Counters`, fill them from the gateway's exported snapshot where the other counters are filled, and let the existing exit/periodic writers carry them. No new writer, no new file.

#### In scope
The `Counters` fields + JSON keys; the gateway accessor that exposes the `ttlUpgrades` snapshot; the fill site(s) for serve (and guard once its exit row exists); unit tests in `internal/gatewayusageledger` and the gateway fill path.

#### Out of scope
The guard usage-exit row itself (sibling issue); any savings-ledger (Track 2) mechanism row for TTL upgrades; changing the AUTO posture rule; `/metrics` output.

#### Done condition
A session with the lever active appends a gateway-usage row whose counters include the TTL-upgrade outcome family; `tools/managed_cache_proving_ground.py --json` shows `ttl_upgrade_1h` at CHANNEL_READY (or EVIDENCED with a nonzero upgraded count) with no tool change.

#### Witness
`go test ./internal/gatewayusageledger ./internal/gateway` green; `python3 tools/managed_cache_proving_ground_test.py` green; the rung climb visible in the `--json` report and the re-pinned `tools/managed_cache_proving_ground.data/baseline.json`.

#### Acceptance gate
Same as Done condition; plus `make cache-proving` green after the baseline re-pin.

#### Work unit
One worker owns fields + fill + tests + baseline re-pin in one sitting.

#### Expected steps
4

#### Assumptions
- The closed `agent.TTLUpgradeReason*` vocabulary is stable enough to serialize by name.
- Adding counter fields is schema-compatible (readers tolerate new keys; `fak-gateway-usage-ledger/1` stays).

#### Confusion risks
- Do not persist per-request rows — the ledger is per-session snapshots; the upgrade counters are cumulative like every sibling counter.
- A zero panel with the lever active is signal (every head ineligible), not a bug; keep the outcome buckets so the zero stays explained.

#### Coordination
- `internal/gateway` is under active multi-session edit; verify the lane via `dos_arbitrate` before writing.
- The guard exit-row sibling issue touches the same fill helper — land whichever first, rebase the other.

#### Trigger
Filed from the 2026-07-03 managed-cache proving-ground audit: rung 1 UNWIRED because no durable channel exists.

#### Batch policy
One issue; deduped against #1614 (closed: the lever) and #1601 (the epic: usage-plane gaps) — this is the TTL-counter slice only; update rather than re-file.

#### Likely files
`internal/gatewayusageledger/ledger.go`, `internal/gateway/metrics.go`, `internal/gateway/gateway.go`, `internal/gatewayusageledger/ledger_test.go`

#### Lane
`gateway`

#### Closure binding
Closed by the ship commit adding the counter family, stamped `(fak gateway)` and referencing this issue; the proving-ground rung climb in the same change is the binding witness.

#### Ship discipline
- Trunk only; commit by explicit path; Conventional-Commits subject + `(fak gateway)` stamp.
- Honest-scope fence: persisting outcomes proves reachability, not savings — no share claim until a real API-key session shows nonzero `upgraded`.

_Self-contained: depends only on already-landed C6 lever code._

---

### Issue B — `feat(guard): append a gateway-usage exit row at guard teardown`

**Managed cache** · the flagship session type is invisible in the usage ledger every lever needs.

#### Parent context
Epic #1601 (durable observability for fak's own usage) and #1844 (the proving ground's population). Guard exits already append Track-2 savings rows (`appendObservedCacheSavingsTo`, `cmd/fak/cachevalue_savings.go`); serve exits already append gateway-usage rows.

#### Current state
All 400 gateway-usage rows are `session_type="serve"`; `rg gatewayusageledger cmd/fak/guard.go` is empty — guard teardown writes no counter snapshot, so guard sessions have no durable counter trail (compaction attempts, deny counts, kv counters, and — once the sibling issue lands — TTL-upgrade outcomes). The managed-cache proving ground holds `guard_usage_plane` at rung 2 CHANNEL_READY.

#### Why now
Every guard-path cache lever's counters (C6 outcomes, compaction attempt family, future placement counters) ride this row. Without it the proving ground can watch only the savings ledger's two mechanisms, and #1844's live ablation arms have no per-session counter population to diff.

#### Working spine
At guard teardown, where the savings writer already runs with the live `AdjudicationSummary`, build `gatewayusageledger.Counters` the same way `cmd/fak/serve.go` does and append one `kind="exit"`, `session_type="guard"` row. Same writer, same ledger file, same honesty fence (counts and totals only, never prompt bytes).

#### In scope
The guard teardown append + its context label (`claude` etc. — mirror the savings row's context); a test at the guard-format/teardown seam mirroring the serve-side ledger test idioms.

#### Out of scope
New counter fields (sibling issue A); periodic in-flight snapshots for guard; backfilling historical sessions; the savings ledger.

#### Done condition
A real `fak guard -- claude` session exit appends a gateway-usage row with `session_type="guard"` and live counters; `tools/managed_cache_proving_ground.py --json` shows `guard_usage_plane` at EVIDENCED with no tool change.

#### Witness
`go test ./cmd/fak` green on the new seam test; a live guard session's exit row visible in `docs/nightrun/gateway-usage.jsonl`; the rung climb in the proving-ground report and the re-pinned baseline.

#### Acceptance gate
Same as Done condition; plus `make cache-proving` green after the baseline re-pin.

#### Work unit
One worker owns the teardown append + test + baseline re-pin in one sitting.

#### Expected steps
3

#### Assumptions
- Guard teardown has access to the same counter snapshot the guard-exit banner already renders (`AdjudicationSummary` and kernel counters).
- Appending from guard is safe under concurrent sessions (the ledger append is already line-atomic for serve).

#### Confusion risks
- Do not conflate this with the savings ledger — both fire at the same teardown but carry different schemas to different files.
- `session_type` must be `"guard"`, not the context label; the proving ground folds on it.

#### Coordination
- `cmd/**` is a live lane in the current fleet (held by a sibling session on 2026-07-03); verify via `dos_arbitrate` before writing.

#### Trigger
Filed from the 2026-07-03 managed-cache proving-ground audit: 0 of 400 usage rows are guard.

#### Batch policy
One issue; child of #1601 (records the epic-level gap) — this is the guard-exit slice only; update rather than re-file.

#### Likely files
`cmd/fak/guard.go`, `cmd/fak/cachevalue_savings.go`, `internal/gatewayusageledger/ledger.go` (read-only), `cmd/fak/guard_format.go`

#### Lane
`cmd`

#### Closure binding
Closed by the ship commit adding the guard exit row, stamped `(fak cmd)` and referencing this issue; the first real guard row in the committed ledger is the binding witness.

#### Ship discipline
- Trunk only; commit by explicit path; Conventional-Commits subject + `(fak cmd)` stamp.
- Honest-scope fence: an exit row proves the trail exists; it says nothing about what the counters contain.

_Self-contained: no dependency on issue A (lands with whatever counter fields exist)._

---

### Issue C — `feat(gateway): stamp breakpoint-placement attribution onto cache-savings rows`

**Managed cache** · fak's one already-witnessed placement saving is unprovable on real traffic.

#### Parent context
#1603 (breakpoint plan for no-cache_control callers) / #806 (the placement-savings witness, `internal/gateway/provider_cache_fak_placement_savings_test.go`) / epic #1844 C3 (bind savings to the per-owner split).

#### Current state
When fak places `cache_control` for a caller that sent none, the resulting provider-cache saving lands in the same `provider_prompt_cache` savings row as pure passthrough — `internal/cachevaluereport/track2.go` `NewSavingsRows` carries no placement field, so on real rows fak's authorship is invisible and the proving ground (`tools/managed_cache_proving_ground.py`) holds `breakpoint_placement` at rung 1 UNWIRED. The 2026-07-01 share note proved placement is identity on the recorded population (Claude Code marks its own head), so the moment a no-breakpoint caller population exists, attribution must already be in the row or the evidence is lost.

#### Why now
This is the cheapest attribution fix in the #1844 family: one field at write time versus forensically unrecoverable after the fact. It also unblocks the C3 acceptance ("the ablation table shows, per arm, its provider vs fak token-equiv delta") for the placement arm.

#### Working spine
Thread the gateway's per-session placement outcome (fak-placed count vs `already_set` identity — the same closed vocabulary the placement code already returns) into `SavingsObservation`, and stamp it on the provider row as a `placement` dimension (e.g. `"fak_placed"` / `"already_set"` / `"mixed"`) plus the placed-breakpoint count. Schema stays `fak-cache-savings-ledger/1` (additive key); the proving ground's mechanism vocabulary is untouched.

#### In scope
`SavingsObservation` + `SavingsRow` fields, the guard-exit fill (`cmd/fak/cachevalue_savings.go`), the gateway accessor for the placement outcome, tests in `internal/cachevaluereport` and the fill seam.

#### Out of scope
Changing placement behavior itself (#1603 owns that); a new mechanism string; re-attributing historical rows; `fak cachevalue report` table changes (follow-on once rows carry the field).

#### Done condition
A session where fak placed breakpoints writes a provider row stamped `placement:"fak_placed"`; a Claude Code session writes `placement:"already_set"`; existing readers still parse (additive key).

#### Witness
`go test ./internal/cachevaluereport ./cmd/fak` green; a fixture-driven row showing the stamped field; `python3 tools/managed_cache_proving_ground_test.py` green (row contract tolerates the new key).

#### Acceptance gate
Same as Done condition; plus `make cache-proving` green.

#### Work unit
One worker owns fields + fill + tests in one sitting.

#### Expected steps
4

#### Assumptions
- The gateway already knows per-session whether it placed breakpoints (the #806 witness exercises exactly that seam).
- Additive JSON keys do not violate the ledger schema contract.

#### Confusion risks
- `already_set` rows must NOT be counted as fak-authored anywhere — the field exists precisely to keep the passthrough baseline honest.
- Do not invent a new mechanism string; the owner split stays `MechanismSavings`' job.

#### Coordination
- Same `internal/gateway` lane pressure as issue A; verify via `dos_arbitrate` before writing.

#### Trigger
Filed from the 2026-07-03 managed-cache proving-ground audit: rung 1 UNWIRED for lack of an attribution channel.

#### Batch policy
One issue; deduped against #1603 (behavior) and #806 (test witness) — this is the durable-attribution slice only; update rather than re-file.

#### Likely files
`internal/cachevaluereport/track2.go`, `cmd/fak/cachevalue_savings.go`, `internal/gateway/adjudicate_proposed.go`, `internal/cachevaluereport/track2_test.go`

#### Lane
`gateway`

#### Closure binding
Closed by the ship commit adding the placement stamp, stamped `(fak gateway)` and referencing this issue; the stamped fixture row in tests is the binding witness.

#### Ship discipline
- Trunk only; commit by explicit path; Conventional-Commits subject + `(fak gateway)` stamp.
- Honest-scope fence: attribution proves authorship of the breakpoint, not a saving delta — the delta stays the ablation arm's claim.

_Self-contained: depends only on already-landed placement code._

---

### Issue D — `feat(cachevalue): carry session identity on cache-savings rows so per-session joins and ablation-arm diffs bind`

**Managed cache** · the proving ground's populations cannot be joined session-to-session.

#### Parent context
Epic #1844 C2/C3 (live ablation arms diffing real traffic) and the proving-ground spine (`tools/managed_cache_proving_ground.py`, 2026-07-03 note). Gateway-usage rows carry `pid` + `unix_millis` + optional `session_id`; savings rows carry none of these.

#### Current state
`internal/cachevaluereport/track2.go` `SavingsRow` has no PID/session identity — a savings row can only be joined to its usage row (or to a future ablation arm label) by timestamp proximity, which is lossy under concurrent sessions (the fleet routinely runs several guards at once; 59 rows already interleave multiple boxes/sessions).

#### Why now
The live ablation arm (#1844 C2) needs "this session ran arm X" to bind a savings row to an arm. Retroactive joining cannot be repaired; every session that exits before this lands is permanently un-joinable.

#### Working spine
Add `pid`, `unix_millis`, and optional `session_id`/`arm` fields to `SavingsRow` (additive; schema string unchanged), filled at the guard/serve teardown where `NewSavingsRows` is called — the same identity triple the usage ledger already writes, so the join key is shared by construction.

#### In scope
`SavingsRow` fields + fill at both append sites; tests; a short join example in the proving-ground note or tool `--json` output (per-session grouping when identity is present).

#### Out of scope
The ablation arm itself; back-filling old rows; any join UI; changing `fak cachevalue report` aggregation.

#### Done condition
New savings rows carry the identity triple; a proving-ground `--json` report can group savings rows by session where identity exists; old rows still parse.

#### Witness
`go test ./internal/cachevaluereport ./cmd/fak` green; `python3 tools/managed_cache_proving_ground_test.py` green with an identity-bearing fixture row.

#### Acceptance gate
Same as Done condition; plus `make cache-proving` green.

#### Work unit
One worker owns fields + fills + tests in one sitting.

#### Expected steps
3

#### Assumptions
- PID + unix_millis is sufficient identity across restarts (the usage ledger already relies on exactly this).
- Additive JSON keys keep `fak-cache-savings-ledger/1` valid.

#### Confusion risks
- Identity is for JOINING, not deduplication — do not use it to drop "duplicate" rows; multiple rows per session are legal (one per mechanism).
- Keep the honesty fence: identity fields are process metadata, never prompt bytes.

#### Coordination
- Touches the same files as issue C (`track2.go`, `cachevalue_savings.go`); land in either order, rebase the second.

#### Trigger
Filed from the 2026-07-03 managed-cache proving-ground audit: per-session joins between the two guard-exit ledgers are timestamp-proximity only.

#### Batch policy
One issue; no existing issue covers savings-row identity (checked 2026-07-03); update rather than re-file.

#### Likely files
`internal/cachevaluereport/track2.go`, `cmd/fak/cachevalue_savings.go`, `internal/cachevaluereport/track2_test.go`

#### Lane
`gateway`

#### Closure binding
Closed by the ship commit adding the identity fields, stamped `(fak gateway)` and referencing this issue; an identity-bearing row in the committed ledger is the binding witness.

#### Ship discipline
- Trunk only; commit by explicit path; Conventional-Commits subject + `(fak gateway)` stamp.
- Honest-scope fence: identity enables joins; it proves nothing about cache behavior by itself.

_Self-contained: independent of issues A–C._

---

## Not to be confused with

- **`tools/cache_value_ledger.py`** gates the dogfood-run cache multiplier on its
  own experiments ledger; **`tools/vcache_scorecard_gate.py`** gates the
  synthetic Zipf score; **`tools/cache_curve.py`** models decay axes. None of
  them read the real-session nightrun ledgers; this spine does only that.
- **`fak cachevalue report`** prices and attributes; the proving ground
  validates, folds rungs, and ratchets. The two read the same rows.
