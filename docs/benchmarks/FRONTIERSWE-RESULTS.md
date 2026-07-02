---
title: "FrontierSWE time-to-solution — results (GATED, no number yet)"
description: "The authority page for fak's FrontierSWE time-to-solution (TTS) claim. Created empty and gated: no wall-clock or turn-count TTS number is recorded until the official FrontierSWE grader has produced both arms' reward.json and score-parity holds. States the claim boundary, the reserved result shape, the artifact path, and the reproduce command so the row is ready to fill the moment a witnessed run exists."
---

# FrontierSWE time-to-solution — results (GATED)

FrontierSWE time-to-solution is fak's claim to reach the same task score in far less wall-clock. No number is recorded here yet.

This is the single authority page for that claim. It follows the discipline of every number in [BENCHMARK-AUTHORITY.md](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md): one place, traced to a source commit and an artifact. It is created empty and gated on purpose, so the claim boundary is written down before a run exists. That keeps an ungoverned "10x faster" from leaking into the README or a Slack post ahead of the evidence.

Child C15 ([#1721](https://github.com/anthony-chaudhary/fak/issues/1721)) of epic [#1706](https://github.com/anthony-chaudhary/fak/issues/1706).

Try the offline projection now (a floor, not the measured win):

```bash
fak frontierswe describe --tts
```

## The claim boundary

You may quote no time-to-solution number from this page, no wall-clock ratio, no turn-count ratio, no "Nx faster", until both of these hold:

1. the official FrontierSWE grader has produced both arms' `reward.json` (C13, [#1719](https://github.com/anthony-chaudhary/fak/issues/1719)); and
2. score-parity holds, so the fak arm's `correctness`/`speedup` is at least the raw arm's (C11, [#1717](https://github.com/anthony-chaudhary/fak/issues/1717)).

The parity gate is load-bearing. A faster run that scored lower is a regression, not a win. FrontierSWE scores partial credit: `reward.json` gives `correctness` in [0,1], then a `speedup` tier once `correctness` reaches 1.0. So "faster" only means something when you hold it at equal quality.

## What is witnessed today (and is not a TTS number)

These offline-spine facts back the eventual row. None of them is a time-to-solution measurement.

| Fact | State | Evidence |
|---|---|---|
| Task model, 17-task catalog, and 3 scoring families | landed | `internal/frontierswe` (C1–C2); `fak frontierswe describe` (offline) |
| Go scorer reproduces `scripts/score_from_reward.py` | landed, green | `go test ./internal/frontierswe` returns `ok`; parity fixtures and [`FRONTIERSWE-SCORING-PARITY.md`](FRONTIERSWE-SCORING-PARITY.md) |
| Deterministic TTS projection (a floor, not a measurement) | landed | `fak frontierswe describe --tts` (C4/C5) |
| Per-turn reuse fold from `fak serve` `/metrics` | landed | `fak frontierswe cache-witness` (C8) |
| Raw-vs-fak run, grade, and TTS metric | pending | C9 / C12 / C13 / C14, the residual this page waits on |

The projection answers a what-if: how much re-prefill work would fak eliminate at a given reuse rate? That is the falsifiable hypothesis. The grader-backed measurement is what this page will record.

## Reserved row shape (filled on a witnessed run)

Once steps 4 to 8 of the [runbook](FRONTIERSWE-TTS-RUNBOOK.md) have run and the parity gate is green, this table is filled from the graded artifacts, one row per (task, model), raw arm against fak arm:

| Field | Raw arm | fak arm | Source |
|---|---|---|---|
| `correctness` in [0,1] | _gated_ | _gated_ | official grader `reward.json` |
| `speedup` tier, if `correctness` reaches 1.0 | _gated_ | _gated_ | official grader `reward.json` |
| Gated leaderboard score | _gated_ | _gated_ | `internal/frontierswe` scorer, pinned to the oracle |
| Wall-clock to `correctness` 1.0 | _gated_ | _gated_ | C14 per-trial TTS metric |
| Turns to first `correct` | _gated_ | _gated_ | C14 sub-metric |
| Score-parity holds? | — | _gated_ | C11 gate (green before any TTS ratio is quoted) |
| TTS ratio `T_fak / T_raw` | — | _gated_ | derived only after parity holds |

Provenance labels on the filled row follow the house rule. fak's own KV-prefix reuse is WITNESSED. The model's `correctness`/`speedup` and any provider `cache_read` are OBSERVED. The `--tts` projection is MODELED. They are reported side by side and never summed.

## Artifact path and reproduce command (stubbed, ready to fill)

- Artifact, committed on a witnessed run: `experiments/frontierswe/tts-<task>-<model>-<date>.json`, the raw-vs-fak graded comparison in FrontierSWE's own vocabulary (C12 `compare` output).
- Reproduce the offline projection only (not the measurement):

  ```bash
  fak frontierswe describe --tts
  ```

- Reproduce the measured run (the gated pipeline, per the runbook): raw arm, then fak arm (C6/C7 routing), then `fak frontierswe cache-witness`, then the official grader (C13), then the parity gate (C11), then the TTS metric (C14), then `fak frontierswe compare` (C12). See [FRONTIERSWE-TTS-RUNBOOK.md](FRONTIERSWE-TTS-RUNBOOK.md).

## Honesty fences

- This page claims no FrontierSWE result today. The `--tts` projection is a deterministic geometry floor, not a measurement, and must never be quoted as the TTS win.
- The scoring runtime being green (`go test ./internal/frontierswe`) proves fak can re-derive the leaderboard score from a `reward.json`. It does not prove fak has run a task, or that any TTS gap exists.
- The BENCHMARK-AUTHORITY row for this claim carries the same GATED boundary. It holds the artifact path and reproduce command so it is ready to fill, and it carries no number until the gate opens.

## Files

- Runbook, the raw-vs-fak recipe: [`FRONTIERSWE-TTS-RUNBOOK.md`](FRONTIERSWE-TTS-RUNBOOK.md)
- Scoring parity to the published oracle: [`FRONTIERSWE-SCORING-PARITY.md`](FRONTIERSWE-SCORING-PARITY.md)
- Scoring/TTS runtime: `internal/frontierswe`; offline CLI: `cmd/fak/frontierswe.go`
- Authority index and row: [`BENCHMARK-AUTHORITY.md`](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md), under "FrontierSWE time-to-solution (GATED)"

---

*Written on a host with no FrontierSWE sandbox and no official grader. Every result cell is `gated` by design. Fill them by running the witnessed pipeline in the runbook, once the score-parity gate can be evaluated.*
