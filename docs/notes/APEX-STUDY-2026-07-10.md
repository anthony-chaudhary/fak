# STUDY: AMD-AGI/Apex + EnvCommons/APEX-Agents — the grader/RSI harness, through the trust-floor lens

> Borrow-hunt pass, 2026-07-10. Uncommitted study note (scout-loop record). Two source repos, pinned:
> **AMD-AGI/Apex** @ `33857b1bc855b2dfc0dbc4cc6997ab9418cd645d` (`33857b1`) — an agentic GPU-kernel
> optimizer that writes code, grades it, and self-improves; and **EnvCommons/APEX-Agents** @
> `af080ff80b804c2134d0b6ab4ebc13d081cea6af` (`af080ff`) — an RL-environment + rubric-grader harness.
> Read via parallel deep-reader fan-out + a completeness critic per repo. Lens: **the grader is an
> attack surface** — what does each system do that fak's witness stack (`dos verify`, `dos
> commit-audit`, `internal/trajctl`, `internal/antipattern`) does not yet do to stop a worker gaming
> the check that grades it.

## License gate
- **AMD-AGI/Apex** — **MIT** (© 2025 Advanced Micro Devices). Apache-compatible; a snippet *could* be
  vendored with attribution. But every candidate is Python/C++ → Go, so **all `inspire`** (clean-room,
  source-cited). No bytes vendored.
- **EnvCommons/APEX-Agents** — **no LICENSE file** → all-rights-reserved → **`inspire`-ONLY**. Pattern
  reimplemented clean-room from the described behavior; no code copied.

## Why Apex is the highest-signal borrow source fak has studied
Every prior study repo (ktransformers #3900, LMCache #3366, ZML #3236) sits one level *below* fak — a
caching/serving layer. Apex sits at fak's *own* level: an agent whose output is graded and fed back
into self-improvement. Its grader is built on the assumption fak's witness stack does not yet encode —
**a solving agent will try to fake the checker green** — and defends every channel:

| Apex mechanism | witness (pinned `33857b1`) | fak today |
|---|---|---|
| Pin the checker's own bytes | `graders/kernel_grader.py:1071-1081` (`protected_file_hashes` → sha256 recheck → `TAMPER DETECTED`) | **ABSENT** — fak hashes cache blobs (`l3kv`/`blobfs`) and grades diff-vs-claim, but never the check's own bytes → **#3924** |
| Scan artifact for harness bypass | `graders/kernel_grader.py:99-221` (`_detect_benchmark_tampering`: `sys.exit`/`os._exit` short-circuit, hardcoded `PASS`/`BENCHMARK_MS`) | **PARTIAL** — `internal/antipattern` spine exists; no grader-gaming family → **#3923** |
| Static hardcoded-output + import allowlist | `graders/reward_backends.py:143-177` (`run_triton_static_check`: `return torch.tensor([literals])`, `ALLOWED_IMPORTS`) | folded into **#3923** |
| Runtime degeneracy detector | `graders/reward_fn.py:228-283` (`detect_runtime_hacking`: NaN/Inf, identity-kernel, `suspiciously_fast`) | follow-up rung of **#3923** |
| Re-measure + plausibility bound | `graders/kernel_grader.py:428-463` (reject speedup ∉ `[0.01,100]`, independently re-benchmark) | noted-not-filed (needs scorecard-side witness) |
| Synonym + fuzzy tool lookup | (mcp/skills read) | **PARTIAL** — `internal/devindex` is pure `strings.Contains` (`devindex.go:466-537`); `internal/trigram` exists to wire in → **#3925** |

## What fak already owns (dropped as PRESENT, witnessed)
- **Artifact-as-oracle** (success = artifact appeared, not exit code) — `dos verify` / `dos commit-audit`.
- **Auth-vs-transient retry** — fak does it *better*: `internal/agent/account_failover_test.go` classifies the
  deceptive org-OAuth-disabled 403 (reads like re-login helps; it doesn't) as permanent → fail-fast + account-swap.
- **Deterministic pinned-schema judge** — `internal/trajctl/{judgescorer,rubric}.go`: forced-tool-choice pinned
  JSON schema, `Temperature:0`, per-call token cap enforced on request AND return, fail-closed everywhere.
  Stronger than EnvCommons' substring parse of judge prose.
- **Per-criterion rubric attribution** — `trajctl` `RubricCriterion.ID` (`rubric.go:48`), #2544.

## EnvCommons (APEX-Agents) — one surviving gap
fak's `internal/trajctl` rubric+judge already implements the strong version of everything transferable,
*and stricter*. The single gap: fak's rubric findings are **soft `[0,1]` progress** (`RubricFinding.Progress`);
EnvCommons grades **conjunctively** (`reward = 1.0 iff every criterion passes`). → **#3926** (binary AND-gate
rubric mode, opt-in, extends #2544). Container-substrate ideas (read-only input mount, single-submission latch,
type-gated channel) don't transfer cleanly to fak's local-worktree isolation — noted, not filed.

## Issues filed this pass (6)
- **#3921** `epic(apex-study)` — mine AMD-AGI/Apex for fak — witnessed borrows.
- **#3922** `epic(apexagents-study)` — mine EnvCommons/APEX-Agents for fak — witnessed borrows.
- **#3923** `feat(antipattern)` SOLUTION_GAMES_CHECKER detector (parent #3921).
- **#3924** `feat(safecommit)` pin-the-checker-bytes witness (parent #3921).
- **#3925** `feat(devindex)` synonym + fuzzy fallback for feature/leaf/doc lookup (parent #3921).
- **#3926** `feat(trajctl)` binary conjunctive rubric mode (parent #3922, extends #2544).

Discipline held to the #3900 bar: witness hard, drop most candidates as PRESENT with citations, file
only the few genuinely PARTIAL/ABSENT leaves. Every source line citation above was re-verified against
the pinned checkouts before filing (no recalled/unverified pins).
