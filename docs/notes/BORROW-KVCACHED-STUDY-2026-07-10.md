# Borrowing from kvcached: a 12-axis witness against fak (2026-07-10)

Source studied: `github.com/ovg-project/kvcached` @ `fea40de` (Apache-2.0) — a virtual-memory KV-cache allocator (paged, watermarked free-lists, background prealloc, a controller that hibernates idle models, and a `kvtop` meter). The exercise: extract each *mechanism* as a repo-agnostic axis, then **witness** it against fak on-axis (dogfooding `fak_feature_query` + raw grep/read), classifying **PRESENT / PARTIAL / ABSENT** with real fak seams — never rubber-stamping a verdict. Borrow mode throughout is **inspire** (reimplement the idea in Go), never vendoring bytes.

## Tally
12 axes: **3 PRESENT** (fak already had it), **8 PARTIAL**, **0 ABSENT**, **1 re-witnessed** after a transient rate-limit. 6 distinct un-filed borrows filed as leaves; 1 folded into an existing issue; 1 parked (blocked on a not-yet-existent seam); 3 recorded as already-present.

## Axis-by-axis

| # | Axis (kvcached mechanism) | Verdict | fak seam | Disposition |
|---|---|---|---|---|
| 1 | min-across-resources admission + floor while shrinking | PARTIAL (med) | `dispatchtick/preflight.go:249` `EvaluatePreflight` (min present; contraction-floor absent) | **#4038** |
| 2 | single-flight fault-in (coalesce a wake stampede) | PARTIAL (high; self-diagnosed) | `microagent/hibernate.go:158` `Wake`; `cmd/tokendemo/cold.go:309` | **#4034** |
| 3 | min-dwell / anti-flap before re-transition | **PRESENT** | `serving_autoscaler` readyToAct cooldown; accounts/cooldown | recorded (residual: MinSleep floor on `Wake`) |
| 4 | decommit hysteresis band (park warm, decommit past cap) | PARTIAL (high) | `microagent/hibernate.go:239` `Release` | **#4035** (merged w/ #8) |
| 5 | signed retention cap (unlimited/off/capped in one dial) | PARTIAL (med) | `radixkv.go:348` `evictToBudget` (positive-only); `ctxresidency` (signed, other pool) | **#4039** |
| 6 | externally-written setpoint, grow-now/shrink-on-drain | PARTIAL (high) | `serving_autoscaler.go:415` (present for serving; dispatch cap not setpoint-driven) | **#4036** |
| 7 | context overcommit fail-loud | **PRESENT** | `ctxplan` pagefault/faithful | recorded |
| 8 | warm prewarm pool (background refill below low-water) | PARTIAL (med) | KV layer shipped (#810/#809/#1072); worker layer = only a min-count floor (`preflight_churn.go:59`, WorkerFloor #3368) | **#4035** (merged w/ #4) |
| 9 | refusal carries remedy + class | **PRESENT** | `refusal_notes.go` | recorded (residual: quantitative requested=/available=) |
| 10 | reserved/used/free three-band meter + pressure color | PARTIAL (high) | `ctxvalue.go:74/173/311` (scale exists); `tui_ablate.go:430` `ablateBarCells` (3-band renderer exists, wired elsewhere) | **#4037** |
| 11 | rebind every dropped span to a shared safe tombstone | PARTIAL (high) | `gateway/ctxrestore.go:44-72`; `anthropic_compact.go:1080` | folded into **#3062** (comment) |
| 12 | conserved-accounting witness (barrier-synced sum test) | PARTIAL (med) | technique present in blob/microagent/inkernel; not on charge/reserve | **parked** — blocked until a mutable charge/reserve seam exists |

## Filed (all leaves, provenance = this note)
- **#4034** — per-key single-flight on `HibernationStore.Wake` (wake-stampede coalescing).
- **#4035** — two-watermark warm band on hibernation `ResidentCap` (warm-park before decommit + warm-refill below low-water). Merges axes 4 + 8; the two halves are the release/acquire sides of one band. KV-layer twin already shipped (#810/#809/#1072); this is the worker/agent-process layer.
- **#4036** — externally-written live concurrency setpoint for dispatch, grow-now/shrink-on-drain (sibling of #3368).
- **#4037** — reclaimable-reserved band + pressure color in live meters (reuse `ablateBarCells` + existing 50/80 thresholds).
- **#4038** — floor the shrinking dimension to a pending-contraction target in `EvaluatePreflight` (sibling of #4036, cross-ref #3368).
- **#4039** — one signed retention dial for the radixkv prefix cache (`<0`/`0`/`>0`, matching `ctxresidency`'s grammar).
- **#3062** (existing) — commented: kvcached's `init_with_zero_` rebind-to-shared-safe-page is the direct analog of this issue's restorable-span generalization; suggested an evicted-vs-never-real sentinel as an acceptance sub-point.

## Parked (not filed)
- **Axis 12 (conserved-accounting witness):** the barrier-synced conserved-sum race test is high-value but its subject — a *mutable* charge/reserve accounting seam — does not exist yet. Filing a test for a non-existent seam is noise; revisit when such a counter lands (natural home: the #4036/#4038 dispatch-capacity work).

## Dedup notes
- #810 (CLOSED) + epics #809/#1072 already own the **KV-layer** prewarm/warm-admission — so the warm-band and warm-pool borrows were framed at the **worker/microagent-hibernate** layer, not KV.
- #3368 (predictive floor + reactive clamp-up) is adjacent to #4036/#4038 but a distinct mechanism (growth-side, predictive) — cross-referenced, not duped.
- #2929/#1178 (session hibernate/rehydrate) and #3242/#3165 (cold-**build** GOCACHE tax) are adjacent to #4034/#4035 but distinct — cross-referenced.

_Not committed: branch is `main` and the working tree carried extensive unrelated uncommitted changes at study time. This note is on disk as an untracked artifact._
