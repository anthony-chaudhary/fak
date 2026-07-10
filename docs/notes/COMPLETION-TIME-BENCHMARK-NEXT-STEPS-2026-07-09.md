---
title: "Completion-time benchmarking next steps: from the modeled floor to a first witnessed wall-clock point (GLM-5.2 + DeepSeek)"
description: "A ranked, checkable plan to convert fak's MODELED time-to-solution floor into a MEASURED faster-completion-time win on top agentic benchmarks (FrontierSWE, DeepSWE, SWE-bench Verified, Terminal-Bench), run with GLM-5.2 and DeepSeek-V4 as the base models. States the honest attribution up front (harness efficiency, not raw model speed), separates the self-host arm where fak owns the KV from the closed-API arm where it does not, and splits the critical path into GPU-free-now steps and GPU/Docker-gated steps."
---

# Completion-time benchmarking next steps (GLM-5.2 + DeepSeek)

Date: 2026-07-09.

Status: planning note, not benchmark results. Every completion-time figure fak
holds today is a MODELED floor. This note is the critical path from that floor to
the first MEASURED wall-clock point. It claims no time-to-solution number and must
never be read as one.

## The one-line state

fak's claim is: **reach the same task score in less wall-clock, because the harness
eliminates the per-turn re-prefill work an unmediated agent loop repeats every
turn.** As of today that claim is witnessed only as a deterministic *floor* —
`fak frontierswe describe --tts` and the SWE-bench turn-tax A/B = 17.9× (see
[SWEBENCH-RESULTS.md](../benchmarks/SWEBENCH-RESULTS.md)). No grader-backed
wall-clock ratio exists. [FRONTIERSWE-RESULTS.md](../benchmarks/FRONTIERSWE-RESULTS.md)
is deliberately empty and gated. The inventory rule is explicit: *pass per
wall-clock hour — only when measured, never modeled*
([AGENTIC-BENCHMARK-INVENTORY-2026-06-25.md](AGENTIC-BENCHMARK-INVENTORY-2026-06-25.md)).

So "next steps to show faster completion time" is not "run a projection." It is
"produce the first witnessed point," and everything below is ordered to that.

## Honest attribution (read before designing any run)

The completion-time win does **not** come from GLM-5.2 or DeepSeek decoding faster.
On raw tokens/sec fak is at parity or slightly behind a tuned single-tenant server.
The wall-clock win comes from **harness efficiency**:

- **KV persistence / RadixAttention prefix reuse** — turn `k` reuses the resident
  KV of turns `1..k-1` instead of re-prefilling the whole growing context. This is
  the 17.9× turn-tax lever, and it is the *largest* lever.
- **Fused serving / cross-agent prefix sharing** — the system+tool preamble is
  prefilled once for all workers, not once per worker.
- **In-process adjudication** — the per-tool-call gate runs in-kernel instead of
  spawning a hook per call (measured ~2.4 µs vs ~5.8 ms spawn-per-hook).
- **Fewer turns** — tool-result batching and vDSO call-elimination cut the number
  of round trips to first-correct.

### The load-bearing caveat: does fak own the KV?

The magnitude of the win splits hard on one question, and the plan must keep the
two arms separate or it will over-claim:

| Arm | Who owns the KV | Which levers bite | Expected win |
|---|---|---|---|
| **Self-hosted** GLM-5.2 / DeepSeek behind `fak serve` (vLLM/SGLang) | **fak** | all four, incl. the 17.9× turn-tax re-prefill elimination | largest — this is where the floor is relevant |
| **Closed DeepSeek API** behind `fak serve` | the **provider** (it already prompt-caches) | only fewer-turns + in-process adjudication + vDSO + cache-preserving order | real but **small**; the 17.9× floor does **not** apply here |

Never quote the turn-tax floor for the closed-API arm. The provider owns the KV
there, so fak cannot claim the re-prefill elimination it can claim self-hosted.

## What each model surface measures *today* (and what it is not)

- **GLM-5.2** — `cmd/glmdsatput` / `internal/glm52prefillsweep` measure *native
  decode throughput* and prefill sweeps (GPU-gated). That is tokens/sec, not
  task-completion-time. GLM-5.2 is the best open-weight arm precisely because it is
  self-hostable behind `fak serve`, so fak owns the KV and the full turn-tax lever
  applies.
- **DeepSeek-V4** — `fak deepseekbench` (`internal/deepseekbench`) is a *serving
  latency* scorecard: TTFT / TPOT / context-scaling for V4 Pro/Flash, no-key
  dry-run by default, live doubly gated behind `DEEPSEEK_API_KEY` + `--spend`, and
  it carries a speedup-refusal gate. TTFT/TPOT is per-token speed, **not** agentic
  task time. A DeepSeek completion-time number is a *different, unbuilt*
  measurement than what `deepseekbench` reports today.

The gap is the same for both models: there is a serving/throughput surface, and a
modeled task-time floor, but **no measured agentic wall-clock-to-solution.**

## The critical path to the first witnessed point

Ranked by leverage. Split into what runs on this box (no GPU/Docker) and what needs
the GPU server or a Docker/Modal grader. The GPU-free steps unblock the gated ones,
so they come first.

### GPU-free now (do these to make the measured run possible and honest)

1. **`fak serve` `/metrics` Prometheus route** — the named integration gap in
   [SWEBENCH-RESULTS.md](../benchmarks/SWEBENCH-RESULTS.md). Today the kernel
   counters (`kernel.Counters()` — vDSO hits, quarantines, denies) and the
   prefix-reuse stats live only in-process; the gateway exposes no `/metrics`.
   Closing it (a) makes fak scrapable like SGLang so a harness monitor loop can
   read it, and (b) feeds `fak frontierswe cache-witness` a *live* per-turn
   reused-prefill series instead of a captured-body fold. Highest leverage: it is
   what turns the reuse claim from MODELED to WITNESSED per turn. GPU-free Go.
2. **C6 route shim (`harbor_ext`-compatible)** — route the harness's model traffic
   through a co-resident `fak serve` without changing task, model, budget, or retry
   policy. Buildable and unit-testable now against a fixture endpoint; it is the one
   piece both the FrontierSWE arm and the DeepSWE arm need. (FrontierSWE runbook C6,
   [#1712](https://github.com/anthony-chaudhary/fak/issues/1712).)
3. **C14 TTS metric + C12 compare (pure Go over artifacts)** — wall-clock-to-
   `correctness`-1.0 and turns-to-first-correct per trial, and the raw-vs-fak
   compare table in FrontierSWE's own vocabulary. These are pure functions over
   graded `reward.json` + the per-turn trajectory; write them against committed
   fixtures now so the pipeline is ready the moment a real run lands. (Runbook
   C14/C12, [#1720](https://github.com/anthony-chaudhary/fak/issues/1720) /
   [#1718](https://github.com/anthony-chaudhary/fak/issues/1718).)
4. **DeepSWE adapter** — drive the mini-SWE-agent scaffold through `fak serve` and
   emit the compare schema keyed by task id (the inventory's cleanest harness test:
   DeepSWE already publishes cost/output/step fields *and* includes GLM-5.2 rows
   under the same scaffold). Harness code is GPU-free; only the run needs endpoints.

### GPU / Docker gated (the actual measured run)

5. **Stand up the model endpoints behind `fak serve`.** GLM-5.2-FP8 on the GPU
   server via the vLLM recipe (8×H200/H20 or 8×B200-class), and DeepSeek-V4 via the
   API for the closed-API arm and/or self-hosted for the KV-owned arm. Same model,
   budget, and retry policy for both raw and fak arms — that identity is the whole
   point.
6. **Run raw vs fak on one task, then a 20-task slice.** Capture per-turn wall-clock
   and the reuse trace on the fak arm.
7. **Grade both arms with the official grader** (FrontierSWE grader / SWE-bench
   `run_evaluation` — Docker/Modal). This is the only authority for correctness.
8. **Parity gate, then ratio.** Assert `correctness`/`speedup`(fak) ≥ (raw). If
   parity fails, stop — a faster run that scored lower is a regression, not a win.
   Only then read the TTS ratio `T_fak / T_raw`.

## Run order (smallest honest win first)

1. **DeepSWE 20-task smoke, GLM-5.2 self-hosted, raw-through-vLLM vs fak-routed.**
   This is the cheapest first witnessed completion-time point: fak owns the KV, the
   full turn-tax lever applies, the scaffold already reports steps/cost, and GLM-5.2
   has a published row to sanity-check parity against. Target: parity green,
   `T_fak/T_raw < 1`, on even one task.
2. **DeepSeek-V4 arm, both flavors.** Self-hosted (KV-owned, large win expected) and
   closed-API (small win expected, turn-tax floor NOT claimed) — reported as two
   distinct rows so the caveat above stays visible.
3. **FrontierSWE one-task TTS trial** once C6/C12/C14 + `/metrics` are landed —
   the highest-signal surface because its 20h/task budget is dominated by exactly
   the re-prefill work fak eliminates.
4. **SWE-bench Verified control** (`fak swebench compare`) as the known-good
   compatibility check, and **Terminal-Bench 2.0 smoke** for the terminal surface.

## Honesty fences

- No completion-time number is claimed here. `--tts` and the 17.9× turn-tax are
  deterministic floors, never the measured win.
- Report WITNESSED (fak's own KV-prefix reuse, `kv_prefix.reused_tokens`), OBSERVED
  (the model's `correctness`/`speedup`, any provider `cache_read`), and MODELED (the
  `--tts` projection) side by side; never sum them.
- A raw-pass-@1 tie is a fak win only if wall-clock (or safe-pass / cost / evidence)
  improves at held-equal quality. The parity gate is what enforces "held-equal."
- Keep the self-host arm and the closed-API arm as separate rows. The turn-tax lever
  applies only where fak owns the KV.
- Every public row still has to land as JSON plus a
  [BENCHMARK-AUTHORITY.md](../../BENCHMARK-AUTHORITY.md) entry, per house discipline.

## Provenance / pointers

- Time-to-solution surface: `cmd/fak/frontierswe.go`, `internal/frontierswe`;
  gated authority page [FRONTIERSWE-RESULTS.md](../benchmarks/FRONTIERSWE-RESULTS.md)
  and recipe [FRONTIERSWE-TTS-RUNBOOK.md](../benchmarks/FRONTIERSWE-TTS-RUNBOOK.md).
- SWE-bench turn-tax floor + the `/metrics` integration gap:
  [SWEBENCH-RESULTS.md](../benchmarks/SWEBENCH-RESULTS.md).
- Model serving surfaces (throughput/latency, not task time): `cmd/glmdsatput`,
  `internal/glm52prefillsweep` (GLM-5.2); `cmd/fak/deepseekbench.go`,
  `internal/deepseekbench` (DeepSeek-V4).
- Benchmark selection + win map: [AGENTIC-BENCHMARK-INVENTORY-2026-06-25.md](AGENTIC-BENCHMARK-INVENTORY-2026-06-25.md).
- Discipline: [BENCHMARK-AUTHORITY.md](../../BENCHMARK-AUTHORITY.md),
  [BENCHMARK-GOVERNANCE.md](../../BENCHMARK-GOVERNANCE.md).
</content>
</invoke>
