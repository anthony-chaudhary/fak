---
title: "Repeated tool-result dedup: the replayed bound over the existing corpus (2026-08-07)"
description: "Counterfactual replay of the shipped cross-turn fold over all 3,288 native-Codex rollouts: 158 of 300 REPEATED_TOOL_RESULT windows collapse, 165 MB shed, 0 destroyed lines — and 52.4% of the audited duplicate bytes are structurally unreachable by any within-wire mechanism."
---

# Repeated tool-result dedup: the replayed bound (2026-08-07)

This is the **measurable half** of [#5254](https://github.com/anthony-chaudhary/fak/issues/5254)
item 3. It is not the DoD's own witness, and it does not claim to be — see
"What this does NOT establish" at the bottom, which restates the unmeetable dimensions as
`n/a` with reasons rather than inflating them.

- Mechanism under test: `agent.dedupMessagesCrossTurn` — the cross-turn verbatim-span fold
  `2ab350829c` added to the decoded wire. Design note:
  `docs/notes/COMPACTION-REPEATED-TOOL-RESULT-DEDUP-2026-07-19.md`.
- Scorer: `session.ScanCompactRolloutReplay` (`internal/session/compactregrowth_replay.go`),
  which drives the candidate over the **same** post-fire windows the #4768 regrowth audit
  already measures and re-applies the audit's own anomaly rule to the folded bodies.
- Harness: `internal/agent/message_elide_corpus_replay_test.go`.

## Reproduce

```
FAK_COMPACT_CORPUS="$HOME/.codex/sessions" \
  go test ./internal/agent/ -run TestCorpusReplayRepeatedToolResultFold -timeout 3000s -count=1 -v
```

Run 2026-08-07, 168.6 s, exit 0.

## Corpus (the denominator)

| quantity | value |
|---|---|
| rollouts scanned | 3,288 (0 unreadable) |
| compaction fires | 1,591 |
| post-fire windows carrying >= 1 tool result | **1,544** |
| tool-result rows / bytes in those windows | 277,684 / 1,195,667,003 |
| rows clipped by the scanner's 128 KB head bound | 21 (0.008%) |

Reconciles with the published attribution witness (`regrowth-witness-2026-07-18.md`:
1,510 windows, 296 anomalies, 12,458,846 dup bytes) and with the 2026-08-06 guarded-cohort
re-measure (1,587 windows, 300 anomalies). The corpus grows between passes; the class share
does not move.

## BEFORE — what the audit sees

| quantity | value |
|---|---|
| windows carrying `REPEATED_TOOL_RESULT` | **300 / 1,544 (19.4%)** |
| duplicate tool-result rows | 1,610 |
| duplicate tool-result bytes | 12,673,440 |

## REACH — how much of that a within-wire fold can even see

This is the result that most changes what #5254 can claim.

| split | rows | bytes | share of dup bytes |
|---|---|---|---|
| **in-window** (earlier copy is inside the same post-fire wire) | 812 | 6,027,548 | **47.6%** |
| **cross-fire** (earlier copy precedes the fire) | 798 | 6,645,892 | **52.4% — UNREACHABLE** |

The audit's duplicate table is session-wide and spans compaction fires, so it correctly types a
row a duplicate of something that was *compacted out of the wire*. A gateway-side fold has
nothing left to match that against. **Just over half the duplicate bytes #5254 is named after
are structurally out of reach of the mechanism #5254 shipped** — and they are out of reach of
any within-wire mechanism, not just this one.

A further 37 of the 812 in-window duplicate rows (4.6%) are under the fold's 8-line minimum-run
floor and cannot fold for that reason alone.

## AFTER — the replayed bound

| quantity | before | after | delta |
|---|---|---|---|
| windows carrying `REPEATED_TOOL_RESULT` | 300 | 142 | **158 collapsed (52.7%)** |
| duplicate rows | 1,610 | 648 | -59.8% |
| duplicate bytes | 12,673,440 | 5,680,466 | **-55.2%** |

Rows folded: 49,716. **Bytes shed: 165,366,011 (13.8% of all tool-result bytes in the measured
windows).**

### The 165 MB is 13x the "duplication" the audit measured — and that is a finding, not a typo

48,161 of the 49,716 folds (96.9%) are **partial**: the fold collapsed a shared line *span*
inside a body that is not byte-identical to any other body. Only 1,555 folds are whole-body
duplicates. The audit's `dup_bytes` keys on `{content hash, ROW length}` at whole-row
granularity, so it cannot see span-level repetition at all.

**Consequence for the DoD.** `REPEATED_TOOL_RESULT` window share and `tool_result/*` `dup_bytes`
— the two quantities item 3 names — undercount this mechanism's actual effect by roughly an
order of magnitude (12.7 MB of "duplication" measured vs 165 MB actually foldable). They are the
wrong instrument even setting aside the rollout-vs-wire blindness already documented in the
design note.

## LOSS — the false-positive rate

The mechanism is span-level, so the correctness property is **lossless-by-relocation**: every
line the fold removes must still be reachable verbatim in a strictly-earlier body of the same
window. A body-level check ("anything that is not a whole-body duplicate must survive
byte-identical") is the wrong test — two genuinely different `git status` runs legitimately
share a long identical run — and scoring it that way reported a spurious 598/2,906 "false
positive" rate on a pilot slice before the check was corrected.

| quantity | value |
|---|---|
| lines removed and relocated (still reachable earlier) | 2,903,131 |
| **lines destroyed (false positives)** | **0** |
| false-positive rate | **0 / 2,903,131 = 0.000%** |
| windows whose span count changed | 0 |

**The zero is not vacuous.** The same check, on the same corpus, run against the *full* shipped
pass (dedup **+** the pre-existing size-gated head+tail elision) reports **1,196,294 destroyed
lines** — head+tail is deliberately lossy-but-bounded and predates this issue. So the detector
demonstrably fires on real data; it reports 0 for the dedup level specifically.

For scale, the full pass sheds 316,877,983 bytes (26.5% of tool-result bytes) across 60,199
rows; the dedup level contributes 165 MB of that losslessly.

## Fidelity caveats

- The replay hands the fold **tool-result bodies only**. A real wire interleaves other roles, so
  the protected recent window (`elideRecentKeepMsgs` = 4 *messages*) shields fewer tool results
  in production than here. The reported win is therefore a **conservative floor**.
- Bodies are unwrapped from the JSONL `output` JSON string before folding, because that is what
  the decoded wire carries. Folding the escaped form would see one line per body and understate
  reach to zero.
- ~96% of this corpus never crossed fak's wire (design note, "Corpus provenance"). This replay is
  a *counterfactual* — "what would the mechanism have collapsed here" — not a claim that these
  bytes were saved in the field.
- The audit's duplicate key includes the full row length, so byte-identical outputs under call
  ids of differing width do not match (pinned by `TestRegrowthReplayDupKeyIsRowLengthSensitive`).
  The 300-window / 12.67 MB figure is therefore itself a lower bound.

## What this does NOT establish

| DoD item 3 dimension | status |
|---|---|
| `REPEATED_TOOL_RESULT` window share reduced **on new sessions** | **n/a** — requires guarded post-port sessions that fire compaction; as of 2026-08-06 there are 2 guarded rollouts and 0 fires since the port. Not measurable in any single run. |
| `tool_result/*` `dup_bytes` reduced **on new sessions** | **n/a** — same population blocker, *and* the metric is blind by construction: the fold rewrites what fak forwards upstream while the Codex rollout row the audit mines is written before fak sees the turn. |

**What would settle it, and when.** Not a `compact-audit` re-run. The wire-side counters
`fak_gateway_uncached_trim_results_total` / `fak_gateway_uncached_trim_shed_tokens_total` already
count this fold on every guarded turn with no dependence on a compaction fire; an A/B on one
session (`--elide-result-bytes 0` to disable) attributes the dedup share directly. That is
measurable as soon as one guarded Codex session runs, not after a cohort accumulates.
