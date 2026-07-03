# Concept: the unbounded memory-value score — giving memory the P&L the cache already has (2026-07-03)

Status: concept + first rung shipping with this note · Track: cache-optimization
(agent memory + reuse) · Sibling doctrine:
[`AGENTIC-DEV-ANTIPATTERNS-2026-07-02.md`](AGENTIC-DEV-ANTIPATTERNS-2026-07-02.md) §4,
[`CONCEPT-MEMORY-IN-THE-LOOP-2026-07-02.md`](CONCEPT-MEMORY-IN-THE-LOOP-2026-07-02.md) R4.

## The seam in one paragraph

fak is now witnessing **true value in limited cases** — the cache side. The
cache-savings ledger (`docs/nightrun/cache-savings.jsonl`) appended three real
net-dollar rows today alone (net_usd 4.23, 18.87, 1.29 — OBSERVED,
provider_prompt_cache, guard sessions), and the cache-value roll-up
(`docs/cache-value-rollup.md`) trends that evidence in two honest tracks. The
memory side just shipped its first real verb — `fak memory recall` (#2346 R1,
commits a6c824eb / 018f04bd), which re-verifies every concrete artifact claim at
page-in and *withholds* a stale note with the failing claim named — but has
**zero value accounting**: no ledger row is appended when a recall renders or
withholds, no score exists in the control pane, no trend exists anywhere. The
cache got a P&L; memory got none. Pulling on the seam = giving memory the same
shape the cache value already has, under the same honesty standards.

## Why the score must be unbounded

The scorecard family already learned this twice. A bounded 0-100 grade answers
"how much of a *finite* baseline is covered" — right for friction-debt, a
category error for a never-done program (`internal/worktype`:
`bounded-metric-chase` is a named antipattern). Memory value is a never-done
program on both sides:

- there is always one more turn a verified recall can orient, one more stale
  note the gate can refuse before it corrupts a decision — value *accumulates*,
  it does not converge on 100%;
- there is always one more way a store can rot — rot *pressure* has no ceiling
  either (a store with 500 stale notes should read 500-notes bad, not "F").

The exemplars: `experience_frontier` (agent-readiness card — unbounded,
monotone, fails low), `heaviness_pressure` (lower = lighter, unbounded), the
cadence ledger's unbounded standing score (`internal/cadencereport`), and the
antipatterns triple (§4). This note applies that established triple to memory.

## The three axes

1. **`memory_value_frontier`** (unbounded, higher = better) — the accumulated,
   *witnessed* value memory has delivered, folded from the recall-events ledger
   (`docs/nightrun/memory-value.jsonl`, schema `fak-memory-value-ledger/1`,
   sibling of the cache-savings ledger). Weighted event terms, weights on the
   established {2,4,8} severity scale:
   - `stale_withheld` × 8 — the differentiator event: a stale memory was
     refused *before injection*, i.e. a sev-8 decision-corruption
     (`frozen-selfreport-memory`) that the raw-MEMORY.md alternative would have
     served as fact. This is the net-true-value framing
     (`docs/standards/net-true-value.md` Q1): measured against the real
     alternative — the unranked, unverified MEMORY.md dump — not a strawman.
   - `fresh_rendered` × 2 — a claim-verified orientation block delivered to a
     turn.
   - `lesson_distilled` × 4 — reserved for the R3 rung (a witnessed-turn lesson
     admitted through the promotion gates); 0 until that rung exists.
   The frontier moves ONLY on realized events from the ledger — never on store
   size (a big store is `unbounded-ephemera`, not value), never on capability
   presence (the agent-readiness card owns affordance counting). No ledger ⇒
   frontier 0, reported as `not yet` — it fails low, never high.
2. **`memory_rot_pressure`** (unbounded, lower = better) — Σ severity × live
   instances over the committed mirror store, from deterministic read-only
   checks: stale artifact claims (sev 4 — would corrupt a decision if injected;
   the recall gate is what stands between them and a turn), dangling index rows,
   orphan fact files, broken `[[wikilinks]]`, frontmatter-grammar violations
   (sev 2 each — the G-group antipatterns `stale-crossref` /
   `dangling-doc-pointer` applied to the store itself).
3. **`memory_debt`** (floored at 0, the ratchet axis) — the HARD subset only:
   store-structural defects mendable in the working tree (index↔file bijection,
   frontmatter grammar). **Two rot kinds are deliberately excluded from debt**
   and stay OBSERVED soft pressure: stale claims (a peer's ordinary commit can
   make a note's SHA/path claim stale without anyone touching memory, so
   stale-claim counts would red the gate for every peer with no memory edit to
   mend — the same external-drift rule that keeps history-window counts out of
   HARD debt in the antipatterns card) and broken `[[wikilinks]]` (the store
   grammar sanctions an unresolved link as a *forward reference* — a memory
   worth writing later — so its mend is a judgment call, not mechanical).

Pressure is the health trend, frontier is the value trend, only debt gates —
the same reading discipline as the antipatterns card (§7).

## Rung 1 shipping with this note

`internal/memvaluescore` (+ tests) and the `cmd/fak/memvaluescore.go` shim:
deterministic, read-only; computes all three axes today. Frontier terms read
the ledger if present (absent today ⇒ 0, `not yet`); pressure/debt are computed
live from the committed mirror (`.claude/memory`). Claim verification **reuses
the shipped recall grammar directly** (`recall.ExtractArtifactClaims` +
`recall.DefaultArtifactVerifier` — the same seam `fak memory recall` gates
page-ins with), so the card and the recall verb can never disagree about what
"stale" means. The pythongate ratchet is why this is Go, not a `tools/*.py`
card: a new Python tool is refused with NEW_PYTHON_TOOL by design. The fold
rides `pkg/scorecard`, so the payload speaks the control-pane envelope
(`memory_debt` as the debt key) and registration is a one-row change when the
pane is clean. The package test pins the live-tree floor: the committed mirror
holds zero structural debt (the conflation-card floor-is-zero precedent — this
is the card's own R4 ratchet rung).

## Next steps (queue layer — parked contract-ready, blocked on dirty files)

Each of these touches a file currently dirty with in-flight peer work
(`tools/scorecard_control_pane.py`, `tools/scorecard_baseline.json`, `dos.toml`,
`cmd/fak/memory_recall.go`), so per the peer-sweep-commit fence they are parked
as issue bodies rather than edited now:

1. **The ledger append seam** — `fak memory recall` appends one
   `fak-memory-value-ledger/1` row per invocation (date, store, intent hash,
   `fresh`/`withheld_stale`/`lessons` counts, est tokens — the field grammar
   `memvaluescore.FoldLedger` already reads). This is the R4 recall-value
   witness's raw material and what makes the frontier move. Blocked on the
   `cmd` lane clearing (`cmd/fak/memory_recall.go` is dirty).
2. **The verb wiring** — the one-line `case "memory-value-scorecard":` in
   `cmd/fak/main.go` (the shim function ships now, unwired; main.go is dirty
   with unrelated in-flight work).
3. **Control-pane registration + baseline pin** — one row
   (`key: memory_value, debt: memory_debt, cmd: go run ./cmd/fak
   memory-value-scorecard --json`) plus the baseline re-pin in the same commit
   (the sota-card precedent). Blocked on the pane/baseline clearing, and on
   step 2.
4. **`dos.toml` lane row** — `memvaluescore = ["internal/memvaluescore/**"]`
   plus the `[lanes].concurrent`/autopick entries (the resume/logvault
   new-leaf reconciliation pattern; until it lands the package ships under the
   `cmd` unit with its shim).
4. **The larger-value rungs** (the memory-in-the-loop ladder, unchanged): R2
   inject verified recall at loop-turn start — that is what multiplies event
   volume from "limited cases" (manual invocations) to every loop turn; R3
   lesson distillation feeding `lesson_distilled`; a store-fork observer
   (`forked-memory-store` is live in this workspace today: the committed
   mirror and the session auto-memory store have disjoint contents).

## Honest fences

- The frontier counts *events the gate witnessed*, not outcomes proven
  end-to-end: a `fresh_rendered` row says a verified block was delivered, not
  that the turn used it well. The end-to-end claim (fewer re-discovery turns /
  repeated refusals) needs the R4 fixture benchmark and stays `not yet` until
  it exists.
- Event weights are severity conventions ({2,4,8}), not measured dollar
  equivalences. The cache ledger has prices; memory events do not (yet). Do not
  quote the frontier in dollars.
- A young store scores a small frontier and near-zero pressure — that is the
  honest reading (little value delivered, little rot), not a defect of the
  score.
- `unverifiable` ≠ `stale`: only a claim the verifier positively decided is
  missing counts stale; prose-only notes stay unverifiable and cost nothing.

## Cross-references

Issues: #2346/#2347 (recall R1), #1559 (loop contract), #2077 (re-verify at
injection), #2141–#2145 (fleet lessons ledger), #782 (memory ECC). Docs:
`docs/cache-value-rollup.md` (the shape being mirrored),
`docs/standards/net-true-value.md`, `CONTEXT-IS-NOT-MEMORY.md`,
`docs/integrations/agent-memory.md`.
