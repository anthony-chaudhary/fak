# Handoff — #2817 per-fire net-score + fire-gate tuning (gen/next)

**Issue:** #2817 — *feat(rsi): score each compaction fire by net value and feed it back so
negative-net fires are suppressed by the tuned policy.* Parent epic #2783 (workstream D:
Score + RSI). Tracking issue: the issue records that `internal/cachevaluereport/**` is under
an active exclusive lease and that code lands via that lane. This note is the handoff/status
artifact for the near-term-foundation primitive that landed ahead of that live seam.

## What landed

The pure primitive is committed on `main` at **`internal/rsiloop/firenettune.go`** (+ acceptance
test `internal/rsiloop/firenettune_test.go`). It sits beside `forksaving.go`: that file measures
a per-fork prefix-reuse saving; this one scores a per-fire compaction receipt and tunes the fire
gate's threshold against the scored corpus.

Public API (all pure, no I/O — deterministic, unit-testable):

- `CompactionFireObs` → `ScoreCompactionFire` → `CompactionFireReceipt` — prices one fire's
  observed facts on the canonical `cacheprice` basis (#2798): `ShedSavingTokenEquiv` (B: shed
  middle × ReadMultiplier × realized horizon), `BurstCostTokenEquiv` (A2: burst suffix ×
  (Write−Read) multiplier), and the headline **`NetScoreTokenEquiv`** (signed B2 net — a
  value-destroying fire reads negative; not floored at zero, per #1303).
- `FirePolicy{MinHorizonMargin}` + `Fires` — the fire/bail threshold the loop tunes. The gate
  fires iff a receipt's ex-ante `PredictedHorizonMargin >= MinHorizonMargin`. It gates ONLY on
  the ex-ante feature, never the ex-post net (feeding the net back would be circular).
- `MeanPerFireNet(corpus, policy)` — the corpus acceptance metric: mean over ALL receipts of
  each fire's realized net under the policy (bailed = 0). Suppressing a positive-net fire lowers
  it, so the tuner cannot cheat by bailing everything.
- `TuneFirePolicy(corpus)` → `TuneResult` (`Lift`, `Summary`) — the RSI tuning pass. Sweeps the
  distinct predicted-margin values in the corpus (plus max+1), returns the margin that maximizes
  mean per-fire net, ties→smallest margin. Never degrades the baseline (baseline is a candidate).
- `CompactionFireScoreRow` / `NewCompactionFireScoreRow` / `CompactionFireScoreSchema`
  (`fak-compaction-fire-score/1`) — the durable witness row a caller appends on the live A4
  receipt seam.

## How it satisfies the issue

- **Why** (per-fire net B2/A2 is the learning signal) → `NetScoreTokenEquiv` is exactly B2 net
  over the A2 debit.
- **What** (attach a net score to each A4 receipt; RSI tunes fire/bail against it) →
  `NewCompactionFireScoreRow` attaches the score; `TuneFirePolicy` tunes the threshold.
- **Acceptance** (over a corpus, mean per-fire net rises after tuning) → witnessed by
  `TestTuneRaisesMeanPerFireNet`.

## Gate actually run

`go test ./internal/rsiloop/ -run "TestScoreCompactionFireNetSign|TestTuneRaisesMeanPerFireNet|TestTuneNeverDegrades|TestNewCompactionFireScoreRow" -count=1` → **PASS** (4/4), `go vet ./internal/rsiloop/` clean. `TestTuneRaisesMeanPerFireNet` is the contract test: over a fixed corpus spanning positive- and negative-net fires, mean per-fire net rises after tuning and the known negative-net fire is bailed.

## Generation frame (gen/next — required at close)

- **Promotion evidence (→ now):** a corpus of REAL per-fire receipts off live guard sessions
  (the A4 receipt seam, once `internal/cachevaluereport` ships it) whose tuned margin lifts
  fleet-median per-fire net and stays stable across replays — that earns wiring the tuned margin
  into the live `CacheBurstPaysBack` gate in `internal/agent/anthropic_compact.go`.
- **Demotion/retirement evidence:** if the tuned margin is ~0 across real corpora (the ex-ante
  estimate is already unbiased — no thin-headroom negative-net cluster to bail), `Lift()` is 0,
  the loop learns nothing, and this folds back into the plain `CacheBurstPaysBack` gate.
- **Invalidating assumption:** that a per-fire receipt can attribute the realized horizon to the
  fire that shed it. If the shed saving cannot be isolated per fire on the live seam (interleaved
  fires, unattributable session end), the ex-post net is unmeasurable and this stays a modeled
  primitive driven by the frozen corpus, not a witnessed one.

## Wiring seam left open (for the leased lane)

Everything here is pure ahead of its live usage seam, mirroring how `forksaving.go` landed ahead
of its. The live wiring — append `NewCompactionFireScoreRow` onto the A4 compaction receipt in
`internal/cachevaluereport/**`, replay the journal through `TuneFirePolicy`, and feed the tuned
`MinHorizonMargin` back into `CacheBurstPaysBack` — is OUT of scope here (that lane is under an
exclusive lease) and lands via that lane.

## Coordination note (operator action) — mis-bound commit

The primitive's code (`firenettune.go` + test) is on `main` but landed under commit
**`a65598a18`**, whose subject is *"feat(rsiloop): add FireNet gap-threshold tuning (#4268)
(fak rsiloop)"*. That commit cites **#4268**, which is an unrelated issue ("Milestone hygiene:
G0 at-risk + 6 milestones missing due dates") — a sibling session on the shared trunk swept these
in-flight files into its own commit before they were committed under #2817 (the shared-tree
`git add` hazard the git laws warn about). Consequence: the file header and this note both cite
#2817, but no commit subject did, so the closure auditor would not bind/close #2817 from
`a65598a18`. **This note is the #2817-citing witness that closes the tracking issue; the code
itself is already present and green.** If an operator wants the code commit rebound too, correct
`a65598a18`'s issue reference via the repair manifest (history rewrite is refused to the worker).
