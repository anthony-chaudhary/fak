---
title: "FrontierSWE time-to-solution — the raw-vs-fak runbook (GATED)"
description: "The exact end-to-end recipe for a raw-vs-fak FrontierSWE time-to-solution (TTS) comparison: discover the tasks offline, run the raw harness arm and the fak-routed arm on the same task/model/budget, witness per-turn prefix reuse, grade both arms with the official grader, and only then, after score-parity holds, read a TTS ratio. States the claim boundary up front: no TTS number is claimed until the official grader has run and score-parity holds."
---

# FrontierSWE time-to-solution — raw-vs-fak runbook (GATED)

This is the recipe to prove fak's FrontierSWE thesis: reach the same `reward.json` in far less wall-clock. No TTS number is claimed until a grader has run.

A FrontierSWE trial runs an agent loop for up to 20 hours per task (`task.toml` sets `[agent] timeout_sec = 72000`). That wall-clock is dominated by the agent loop re-prefilling a growing context every turn, which is exactly the `N`-axis work fak eliminates. This runbook is what an operator follows to prove the time-to-solution (TTS) win end to end, raw harness against the same harness routed through fak. Its companion authority page is [`FRONTIERSWE-RESULTS.md`](FRONTIERSWE-RESULTS.md).

Start here (offline, no assets):

```bash
fak frontierswe describe --tts
```

## The claim boundary (read before quoting anything)

You may quote no TTS number, no wall-clock ratio, no turn-count ratio, no "10x faster", until both of these hold:

1. the official FrontierSWE grader has produced both arms' `reward.json` (C13, [#1719](https://github.com/anthony-chaudhary/fak/issues/1719)); and
2. score-parity holds, so fak's `correctness`/`speedup` is at least the raw arm's (C11, [#1717](https://github.com/anthony-chaudhary/fak/issues/1717)).

A faster run that scored lower is a regression, not a win. The parity gate is what stops an ungoverned TTS number from leaking into the README or a Slack post.

Parent epic: [#1706](https://github.com/anthony-chaudhary/fak/issues/1706). This runbook is child C15 ([#1721](https://github.com/anthony-chaudhary/fak/issues/1721)), created gated so the recipe is written down before a witnessed run exists.

## 1. What runs today vs. what the run still owns

The offline spine (discover, score, TTS-model) is landed and green. The live wiring and the grader are the residual. Do not treat a step below as runnable until its column says so.

| Step | Piece | State | Entry point / issue |
|---|---|---|---|
| Discover | `fak frontierswe describe`, the 17 tasks, 20h budget, envelope, gates | RUNNABLE NOW (offline) | `fak frontierswe describe` (C2, [#1708](https://github.com/anthony-chaudhary/fak/issues/1708)) |
| Project | Deterministic TTS floor (turns to re-prefill work eliminated) | RUNNABLE NOW (offline; a projection, not a measurement) | `fak frontierswe describe --tts` (C4/C5, [#1710](https://github.com/anthony-chaudhary/fak/issues/1710)/[#1711](https://github.com/anthony-chaudhary/fak/issues/1711)) |
| Score | Go port of `scripts/score_from_reward.py` (`reward.json` to gated score) | RUNNABLE NOW (green: `go test ./internal/frontierswe`) | `internal/frontierswe` (C3, [#1709](https://github.com/anthony-chaudhary/fak/issues/1709)); parity: [`FRONTIERSWE-SCORING-PARITY.md`](FRONTIERSWE-SCORING-PARITY.md) |
| Witness reuse | Fold `fak serve` `/metrics` scrapes into the per-turn reused-prefill-token series | RUNNABLE NOW (offline fold of captured bodies) | `fak frontierswe cache-witness` (C8, [#1714](https://github.com/anthony-chaudhary/fak/issues/1714)) |
| Route | `harbor_ext`-compatible shim routing the harness's model traffic through `fak serve` | pending | C6 ([#1712](https://github.com/anthony-chaudhary/fak/issues/1712)) |
| Environment | Build/run a task's Docker/Modal image with `fak serve` co-resident | pending | C7 ([#1713](https://github.com/anthony-chaudhary/fak/issues/1713)) |
| Run one task | Drive one task end-to-end, capturing the submission and the per-turn TTS trace | pending | `fak frontierswe run` (C9, [#1715](https://github.com/anthony-chaudhary/fak/issues/1715)) |
| Grade | Official FrontierSWE verifier/grader (needs Docker/Modal) | pending | `fak frontierswe eval` (C13, [#1719](https://github.com/anthony-chaudhary/fak/issues/1719)) |
| Compare | Raw-vs-fak TTS and score table in FrontierSWE's own vocabulary | pending | `fak frontierswe compare` (C12, [#1718](https://github.com/anthony-chaudhary/fak/issues/1718)) |
| TTS metric | Wall-clock-to-`correctness`-1.0 and turn-count-to-first-correct, per trial | pending | C14 ([#1720](https://github.com/anthony-chaudhary/fak/issues/1720)) |

The `fak frontierswe` binary today exposes exactly two subcommands, `describe` and `cache-witness`. The `run` / `compare` / `eval` steps are named above as the gated pipeline they will become. They are not invocable yet.

## 2. Discover the tasks (offline, no assets)

```bash
# The 17 tasks, their 3 scoring families, the 20h per-task agent budget, and the
# [environment] envelope (cpus / memory_mb / gpus / allow_internet). Fully offline:
# no model, GPU, Modal account, or Docker image.
fak frontierswe describe

# The deterministic time-to-solution PROJECTION (a floor, NOT a measurement): per task,
# the projected TTS ratio T_fak/T_raw at the value-stack cross-turn reuse rate, plus the
# A/C re-prefill-work-elimination floor and a cross-trial sweep.
fak frontierswe describe --tts
```

`--tts` is a deterministic geometry floor from each task's turn/context shape. It is the hypothesis this runbook exists to test against a real grader. Never quote it as the measured TTS win.

## 3. Prerequisites for a live arm

1. A FrontierSWE checkout ([Proximal-Labs/frontier-swe](https://github.com/Proximal-Labs/frontier-swe)) and its Modal/Docker sandbox toolchain. The harness runs each task in its own sandbox.
2. A model and credentials for the CLI harness the raw arm uses (`harbor_ext.claude_code:ClaudeCodeApiKeyNoSearch`, codex, gemini-cli, and so on). Both arms must use the same model, budget, and retry policy.
3. For grading (step 7): Docker/Modal reachable and the FrontierSWE grader available.

## 4. Raw arm — the unmediated baseline

Run the FrontierSWE harness exactly as upstream does, with nothing in the model path. This produces the raw arm's submission and, after grading, its `reward.json`. Pin the task id, model, agent budget (`timeout_sec`), and trial count, and record them. The fak arm must match them field-for-field.

## 5. fak arm — the same harness, routed through fak

The only change from the raw arm is that the harness's model traffic is routed through a co-resident `fak serve` gateway (C6 shim plus C7 environment), so fak's value stack bites on every turn:

- KV persistence and RadixAttention prefix reuse: turn `k` reuses the resident KV of turns `1..k-1` instead of re-prefilling.
- In-process adjudication: the per-tool-call gate runs in-kernel rather than spawning a hook per call.
- vDSO call-elimination: served read-only calls fold into the response.

Everything else, task id, model, budget, retry policy, stays identical to step 4. That identity is the whole point. Any TTS difference must come from fak's mediation, not from a changed workload.

## 6. Witness the reuse per turn (offline fold)

Point the cache-witness at the `fak serve` `/metrics` bodies captured during the fak arm. It folds them into the per-turn reused-prefill-token series and the measured cross-turn reuse rate `r` that the TTS projection consumes:

```bash
fak frontierswe cache-witness --metrics-dir <captured-metrics-dir> \
  --out frontierswe-tts-reuse-trace.jsonl
```

fak's own KV-prefix reuse is WITNESSED (`kv_prefix.reused_tokens`). A provider's `cache_read` is OBSERVED. The two are kept separate and never summed.

## 7. Grade both arms with the official grader

Grade the raw and fak submissions with the official FrontierSWE grader (C13, [#1719](https://github.com/anthony-chaudhary/fak/issues/1719)). On a box without Docker/Modal, the eval step prints the exact remote command to run on a grading box. This is the only authority for `correctness`/`speedup`.

The Go scorer in `internal/frontierswe` is the runtime that re-derives the same number from a `reward.json`. It is pinned to the published script on committed fixtures (see [`FRONTIERSWE-SCORING-PARITY.md`](FRONTIERSWE-SCORING-PARITY.md)), and it does not replace the grader.

## 8. Gate, then compare — the order matters

1. Score-parity gate first (C11, [#1717](https://github.com/anthony-chaudhary/fak/issues/1717)): assert fak's `correctness`/`speedup` is at least the raw arm's. If parity fails, stop. There is no TTS win to claim, only a regression to fix.
2. Only then read the TTS metric (C14, [#1720](https://github.com/anthony-chaudhary/fak/issues/1720)): wall-clock-to-`correctness`-1.0 and turn-count-to-first-correct, per trial.
3. Render the comparison (C12, [#1718](https://github.com/anthony-chaudhary/fak/issues/1718)): the raw-vs-fak TTS and score table keyed to FrontierSWE's own vocabulary. That table, with both arms graded and parity green, is what a row in [`FRONTIERSWE-RESULTS.md`](FRONTIERSWE-RESULTS.md) records.

## 9. The smallest honest win

One task, one trial, both arms graded by the official grader, score-parity green, and a TTS ratio below 1.0 (the fak arm reached the same `reward.json` in less wall-clock). Even a single such trial is the first witnessed TTS point. The full 17-task, multi-trial number is the goal. Until that first point exists, this repo claims no FrontierSWE TTS number.

## 10. Provenance

- Task model, catalog, and scoring runtime: `internal/frontierswe` (C1–C5). The offline entry point `fak frontierswe describe [--tts]` and the reuse fold `fak frontierswe cache-witness` are in `cmd/fak/frontierswe.go`.
- Scoring parity to the published oracle: [`FRONTIERSWE-SCORING-PARITY.md`](FRONTIERSWE-SCORING-PARITY.md), pinned by `internal/frontierswe/score_test.go`.
- House benchmark discipline this runbook follows: [`BENCHMARK-AUTHORITY.md`](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md) and [`BENCHMARK-GOVERNANCE.md`](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-GOVERNANCE.md).
- Written on a host with no FrontierSWE sandbox and no grader. Every measured cell is gated by design. The offline `describe`, `--tts`, and `cache-witness` steps run here. The TTS number does not.
