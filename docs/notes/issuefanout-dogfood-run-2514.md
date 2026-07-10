# Dogfood readout: `fak issue fanout` on this repo's own live work (#2514)

**Date:** 2026-07-09 · **Issue:** #2514 (`fanout-issuefanout-dogfood-self-run`) ·
**Epic:** #2510 · **Spine:** `5b8f0bd1` (`internal/issuefanout` + `fak issue fanout`)

This is committed evidence, not a claim: the issue-fanout planner was run against
this repo's **real** backlog and integration path — not a synthetic fixture — and
every defect the run surfaced is filed. The binary was built from the trunk under
test (`go build ./cmd/fak`, exit 0) so the readout reflects current code, not an
installed stale copy.

## Assumption checked first — the spine still builds/runs on trunk

```
$ go build ./cmd/fak            # exit 0
$ go test ./internal/issuefanout -count=1
ok  github.com/anthony-chaudhary/fak/internal/issuefanout   0.181s
```

## Run 1 — plan the real spine (`--leaf issuefanout --spine 5b8f0bd1`)

The planner emits **15 contract-ready follow-ons** in fixed taxonomy order
(qa×3, dogfood×2, product×3, observability×2, integration×3, docs×1, release×1).
Cross-checked against the live tracker, those 15 marker keys map **1:1** onto the
actually-filed issues #2511–#2525:

| key (`fanout-issuefanout-…`) | filed issue |
|---|---|
| qa-edge-sweep / qa-failure-paths / qa-determinism | #2511 / #2512 / #2513 |
| dogfood-self-run / dogfood-usage-ledger | #2514 / #2515 |
| product-cli-reference / product-lcd-demo / product-error-ux | #2516 / #2517 / #2518 |
| obs-outcome-counters / obs-scorecard | #2519 / #2520 |
| int-guard-gate / int-dos-wiring / int-superloop | #2521 / #2522 / #2523 |
| docs-doctrine-linkage / release-claims | #2524 / #2525 |

**Promotion evidence:** the planner's output *is* the repo's real fan-out backlog
for this leaf — the taxonomy is not aspirational, it shipped as 15 live issues.
Three further hand-authored follow-ons (#2530/#2531/#2532) carry honest `ext-*`
marker keys, i.e. extensions *beyond* the taxonomy, not drift within it.

## Run 2 — adoption meter on the real filed markers

```
$ fak issue fanout --adoption --leaves issuefanout --markers <18 live keys>
spine-fanout adoption: 1/1 shipped leaves cleared the fan-out floor (>=3), 0 gap(s)
  [ok ] issuefanout              18/3 follow-on(s) filed
```

The honesty meter reads the real tracker state (15 taxonomy + 3 `ext-*` = 18) and
correctly clears the floor. Meter works on live data.

## Run 3 — the spine's advertised integration (`cohort --from-plan`)

The `5b8f0bd1` commit promises the JSON plan "feeds straight into `fak issue
cohort --from-plan` for wave planning." It does parse and grade — but the wave
plan is **degenerate**:

```
$ fak issue fanout … --json > plan.json
$ fak issue cohort --from-plan plan.json
issue-cohort: 15 candidate(s) -> 15 dispatchable, 0 to-split, 0 triage, 0 refused
  concurrency: 15 wave(s), peak 1 at once, 105 colliding pair(s)
```

Peak concurrency **1**; C(15,2)=**105** collision pairs; **15 serial waves**.

## Defect surfaced → filed #3716 (`fanout-issuefanout-candidate-path-derivation`)

`expand()` stamps every candidate with the spine leaf's own tree —
`paths=["internal/<leaf>/"]`, `lane=<leaf>`, `coordination="Stay inside
internal/<leaf>/"` — regardless of where that follow-on's work actually lands.
**8 of 15** candidates declare `internal/issuefanout/` while their own
`in_scope`/`witness` text names a different tree:

| candidate | declared `paths` | its own scope names |
|---|---|---|
| docs-doctrine-linkage | `internal/issuefanout/` | docs/INDEX.md, llms.txt, AGENTS.md, README |
| release-claims | `internal/issuefanout/` | CLAIMS.md, docs/releases/ |
| int-dos-wiring | `internal/issuefanout/` | dos.toml |
| int-guard-gate | `internal/issuefanout/` | internal/hooks/ (PreCommitGates) |
| int-superloop | `internal/issuefanout/` | the super-loop/dispatch path |
| product-cli-reference | `internal/issuefanout/` | docs/cli-reference.md |
| product-lcd-demo | `internal/issuefanout/` | cmd/*demo, examples/ |
| dogfood-self-run | `internal/issuefanout/` | docs/notes/ or a ledger |

Two witnessed harms:

1. The cohort wave-planner reads `paths`, so identical paths false-collide every
   follow-on → peak concurrency 1 (Run 3). Rows in genuinely disjoint trees
   (docs/, dos.toml, internal/hooks/, CLAIMS.md) *could* run concurrently but the
   planner hides that from its own downstream.
2. **Live proof on the flagship example:** this very dogfood issue (#2514,
   `dogfood-self-run`) declares `lane=issuefanout`, yet the live dispatcher routed
   it to the **`docs`** lane and this readout lands under `docs/notes/` — because
   that is where the work actually is. The planner's own routing is wrong on the
   one candidate we can check against reality end-to-end.

Filed as **#3716** (labels: fanout, class:dev, priority/P2, gen/now); fixing is
out of scope for #2514 (harden what exists; file, don't grow the spine).

## Demotion / retirement evidence

Nothing was demoted or retired. The dogfood does *not* upgrade any claim: the
planner's "pure planner, filing stays with the caller" boundary is intact, and the
cohort integration is left as-is with its rough edge filed rather than patched.
`--live` native filing is already tracked at #2531; not duplicated here.

## One invalidating assumption

This run exercised a **single** leaf (`issuefanout`) whose backlog was authored by
the same taxonomy that generated it — so the 1:1 match in Run 1 is partly
self-fulfilling. The adoption meter was fed only that leaf's markers; it has **not**
been run across the repo's other recently-shipped leaves, most of which predate the
spine-fanout default and would read as gaps. The honest scope of this dogfood is
"the planner works correctly on its own leaf and its integration has one filed
defect," **not** "the fan-out default is adopted repo-wide" — that broader honesty
meter is what #2532 (fleet-wide adoption scorecard) is for.

## Witness

- Readout: `docs/notes/issuefanout-dogfood-run-2514.md` (this file).
- Defect filed: **#3716** — `fanout-issuefanout-candidate-path-derivation`.
