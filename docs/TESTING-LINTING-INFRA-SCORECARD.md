---
title: "fak testing and linting infrastructure scorecard"
description: "A current-state score and top-20 improvement spine for fak's testing, linting, CI, and agentic developer-experience infrastructure."
---

# Testing and linting infrastructure scorecard

<!-- testing-linting-infra-scorecard: 2026-06-30 - process: evidence-backed manual snapshot -->

This is the scorecard for the developer loop itself: tests, linting, CI, scorecard
gates, local wrappers, and the agent-facing feedback path. It answers two questions:

- How good is the testing/linting infrastructure today?
- What are the top 20 items that would make agentic development faster, better typed,
  more modular, and easier to trust?

This is a current-worktree snapshot, not a generated baseline. The repo is high churn,
so the numeric debt rows should be refreshed before turning any item into a claim. The
spine order is intended to survive churn.

## Evidence snapshot

Collected from the current tree on 2026-06-30:

| Evidence | Current read-back |
|---|---|
| Go package count | `go list ./...` -> 328 packages |
| Go test file count | `rg --files -g "*_test.go"` -> 1327 files |
| Python tool test count | `rg --files tools -g "*_test.py"` -> 237 files |
| Workflow count | `.github/workflows/*.{yml,yaml}` -> 34 workflows |
| `fak test --list` | tiers: `fast`, `full`, `race`, `affected`, `durations`, `shards`, `<pkg>` |
| `fak test --json -n race` / `fak test --json ruff tools` | emits `fak.test_repair_packet.v1` with the resolved command, normalized finding class, diagnostics, exit code, and next action; command-result packets classify Go test, go build, go vet, gofmt, codelint, ruff, setup, and spawn failures, while scorecard-specific parsers are still pending |
| `fak test durations --help` | runs `go test -json` for a selected tier/package via `--run`, writes `fak.test_duration_ledger.v1` via `--out`, or folds an existing stream via `--input`; CI automatic emission still pending |
| `fak test shards --help` | reads `fak.test_duration_ledger.v1` via `--input` and emits deterministic balanced `go test` commands as `fak.test_shard_plan.v1`; CI job consumption and Python tool-test sharding still pending |
| `make test-durations` | local fast-tier duration ledger target writes `.fak/test-duration-ledger.json` with ranked package/test budget findings |
| `fak affected --help` | supports changed-file/package selection, JSON/list dry-runs, budgeted runs, `-run`, `-short`, `-count`, and report output |
| `tools/gated_tool_tests.py --check` | 237 tests = 69 gated elsewhere + 20 quarantined + 148 hermetic; red-debt 6 |
| Go code-quality scorecard | score 78.4/100, grade C, code-debt 23; build/vet/honesty/ship-integrity all 100; tests 326/327 non-trivial packages; architecture debt 18 |
| Python tooling-quality scorecard, static mode | score 70.2/100, grade C, py-debt 65; tests debt 43, architecture debt 22; ruff skipped under `--no-toolchain` |
| Stability scorecard | 98.0/100, grade A, stability-debt 0 |
| Agent-readiness scorecard | 100.0/100, grade A, friction-debt 0 |
| Bench-DX scorecard | 100.0/100, grade A, bench-DX-debt 0 |
| Observability scorecard | 95.9/100, grade A, observability-debt 1 |

Primary files behind the snapshot:

- `Makefile`: local green gate, fast/full/affected/race targets, hygiene, scorecard,
  demo, gated-tool-test, and CUDA header gates.
- `.github/workflows/ci.yml`: Linux authoritative CI, Go build/vet/test with coverage,
  race job, per-commit snapshot build, scorecard regression suites, leak scan,
  DOS review, and no-blackhole tool-test runner.
- `scripts/ci.ps1` and `test.ps1`: Windows bridge, with known differences from the
  Linux/WSL gate.
- `cmd/fak/test.go`, `cmd/fak/testduration.go`, and `cmd/fak/testshards.go`:
  host-aware `fak test` wrapper plus duration-ledger and shard-plan subcommands.
- `cmd/fak/affected.go` and `internal/affectedtests`: affected-package selection.
- `tools/code_quality_scorecard.py`, `tools/tooling_quality_scorecard.py`,
  `tools/gated_tool_tests.py`: current mechanical debt lenses.

## Score

Composite: **78.5/100, grade C**.

The floor is strong: the repo has a real full Go suite, race detector, coverage profile,
claim/hygiene/scorecard gates, per-commit build snapshots, agent-facing wrappers, and a
no-blackhole runner for Python tool tests. The drag is also concrete: local feedback is
still heavy, Python tooling is weakly typed and partly ungated, current code/tooling
architecture debt is high, ruff is optional/skipped on this snapshot, Windows parity is
not exact, and several important checks are advisory rather than ratcheted.

| Dimension | Score | Why |
|---|---:|---|
| Correctness coverage | 86 | Full `go test ./...`, race CI, architest, and many domain witnesses are wired; current generated score still shows one non-trivial Go package without tests and broader model/backend coverage debt outside this loop. |
| Inner-loop velocity | 72 | `fak test`, `test-fast`, `test-affected`, Go caches, local duration ledgers, and a Go shard planner exist; the default full gate is still large, Windows requires WSL routing, and CI has not yet consumed measured shards. |
| Static analysis and typing | 74 | Go build/vet/gofmt/architest are strong, and codelint exists for agent writes; Python tools remain dynamically typed, ruff can skip, and new-tool policy still coexists with a large grandfathered Python surface. |
| CI/local parity | 82 | `make ci` deliberately mirrors many hard CI steps and CI adds race/per-commit snapshots; `scripts/ci.ps1` still differs from Linux/WSL on gofmt and full tool-test execution. |
| Agentic usability | 85 | Agent-readiness is A, `fak test` is discoverable, and `fak test --json` now emits repair-packet envelopes with next actions for Go test plus build/vet/gofmt/codelint/ruff gates; raw scorecard output still lacks the full unified ranked parser set. |
| Modular test architecture | 69 | Architest/lane boundaries exist; current code-quality and tooling-quality scorecards show large god-file/god-function debt that slows targeted tests and safe refactors. |
| Witness and honesty quality | 90 | Claims lint, salience, DOS review, proof artifacts, and rendered/demo witnesses are unusually strong; several coverage and scorecard checks remain advisory. |
| Python tool-test discipline | 70 | The no-blackhole runner accounts for 237 tests and runs 148 hermetic tests; there are 43 untested non-trivial modules, 20 quarantined tests, and 6 red quarantines. |
| CI speed/caching | 76 | Go build/test cache and separate race cache are wired, `make test-durations` emits a local budgeted package/test ledger, and `fak test shards` plans balanced Go shards; CI ledger emission, Python tool-test sharding, CI consumption of the plan, and hard slow-test trends are still pending. |
| Coverage observability | 73 | CI emits a coverage profile and total statement coverage; there is no changed-package coverage floor or trend gate tied to risk. |

## Top-20 Improvement Spine

The ordering favors changes that improve agentic development first: fast feedback, typed
repair signals, modular boundaries, and witnessed claims. Each item has a concrete
definition of done and a witness.

| Rank | Item | Why it matters | Definition of done | Witness |
|---:|---|---|---|---|
| 1 | Make affected testing the default agent loop | Agents should test the packages they can affect before paying for the full suite. | `fak test affected` or equivalent wraps `fak affected` with dry-run, JSON, budget, and pass-through flags; docs point agents there before `make test`. | Unit tests for planner plus a run over representative file sets. |
| 2 | Add a test-duration ledger and slow-test budget | Speed regressions are invisible until people feel them. | CI and local runs emit per-package/test durations into a stable JSON ledger; packages over budget get a ranked action. | A deterministic parser test plus a sample `go test -json` fixture. |
| 3 | Shard CI by package duration, not hand order | The full gate is large enough that fixed ordering wastes parallelism. | A Go planner reads the duration ledger and produces balanced shards for Go tests and Python tool tests. | Planner golden tests plus CI job using the produced shards. |
| 4 | Turn raw failures into an agent repair packet | Agents need the shortest path from red to next edit, not pages of logs. | `fak test --json` emits normalized findings for build, vet, gofmt, Go tests, ruff/codelint, and scorecard failures. Current slice shipped: resolved-command, usage, spawn, Go test, Go build, Go vet, gofmt, codelint, ruff (including explicit unavailable SKIP), and setup failure packets with diagnostics/output tails and next action. | Fixture tests for each failure class and a docs example. |
| 5 | Ratchet Python tooling quality from C toward A | `tools/` is load-bearing CI/dispatch code but carries 65 static py-debt units. | Commit `docs/TOOLING-QUALITY-SCORECARD.md`, retire the 6 red quarantines, and set a non-regression baseline for py-debt. | `tools/tooling_quality_scorecard_test.py` plus scorecard control-pane baseline. |
| 6 | Make ruff availability explicit | A skipped lint is not the same signal as a clean lint. | CI installs or vendors the ruff path used by tooling-quality, or the scorecard labels lint as unmeasured and fails only on hosts that declare ruff support. | CI step proving `ruff check` ran, or a typed SKIP reason in the payload. |
| 7 | Retire current Go code-quality hard debt | Existing code-debt is directly slowing test targeting and refactors. | Drive code-debt from 23 to <=11: split the worst god-functions/files first, remove the gofmt defect, and resolve the lone untested package. | `python tools/code_quality_scorecard.py --json` showing the reduced debt. |
| 8 | Restore or explicitly re-budget the zero-dependency invariant | External deps and `go.sum` change build, cache, and supply-chain assumptions. | Either remove `golang.org/x/term`/`x/sys` and `go.sum`, or update the invariant and gates to accept the declared deps. | `go list -m all` plus code-quality `deps` KPI clean or intentionally rebaselined. |
| 9 | Close the tool-test quarantine red-debt | Quarantine is useful only when it trends down. | The six currently red quarantined tool tests are fixed or reclassified with a stronger reason and issue. | `python tools/gated_tool_tests.py --check` with red-debt 0. |
| 10 | Add changed-package coverage floors | Total coverage is too broad to protect a risky edit. | For changed Go packages, report statement coverage and fail only on a declared regression or below-package floor. | Fixture test for coverage parser plus one CI advisory run before hardening. |
| 11 | Add flake classification without retries as proof | Retrying can hide a real failure; classifying preserves signal. | A flake ledger records test, seed, package, failure signature, and reproduction command; no automatic retry converts red to green. | A synthetic flaky fixture proves classification and non-green exit semantics. |
| 12 | Expand negative-control tests for gates | A gate that never proves it can fail is weak evidence. | Each major gate has at least one fixture that fails for the intended reason: claims, file admission, public leak, architest, doc links, tool-test manifest, scorecard ratchet. | Unit tests named after the failure token. |
| 13 | Make codelint a first-class write feedback path | Agent-written code should be parsed before the next turn drifts. | `fak codelint --json` findings are used by agent write loops and documented beside `fak test`; Go/JSON/Python/CUDA pack coverage is listed. | `cmd/fak/codelint` tests plus a SWE-bench lint-writes fixture. |
| 14 | Add a contract-test registry for CLI examples | Docs and examples rot faster than code. | All fenced `fak ...`, `go run ./cmd/...`, and demo commands that are meant to run are registered with expected mode: hermetic, network, GPU, or docs-only. | Existing demo command audit extended to general CLI docs. |
| 15 | Make Windows parity a single command | Agents on Windows should not memorize which gates are Linux-only. | Add `fak ci` or `fak test ci` that routes to WSL for Linux-authoritative gates and labels native-only skips. | `cmd/fak/test` or new command tests for Windows/non-Windows planning. |
| 16 | Move new test/lint orchestration into Go leaves | New infra should strengthen the typed module, not expand Python debt. | New runners and planners live under `internal/<leaf>` plus thin `cmd/fak` verbs; Python remains only for grandfathered tools until ported. | `internal/pythongate` stays green and new command tests cover the runner. |
| 17 | Add fixture registry and seed discipline | Deterministic tests need fixtures that are easy to find and reuse. | A registry lists shared corpora, golden files, seeds, and owner packages; new randomized tests print/replay seeds. | Registry validation test plus a failing-seed replay fixture. |
| 18 | Broaden visual/render witnesses where UI bugs land | Visual defects need captured render proof, not only logic tests. | Terminal/TUI surfaces and demo renderers use captured-byte or screenshot/golden witnesses for regressions reported visually. | Render-witness tests similar to the watchdog pane proof. |
| 19 | Add agent-workflow E2E tests around edit-test-repair | The product is an agent loop; unit tests miss loop ergonomics. | A hermetic harness simulates an agent writing a broken change, receiving lint/test findings, fixing it, and producing a witnessed green state. | Go or existing harness test that asserts the repair packet drove the second edit. |
| 20 | Automate this scorecard | This manual snapshot should become a maintained gate only after the KPIs settle. | A Go `fak testing-scorecard --json` folds the dimensions above, emits debt and trend, and writes this doc on demand. | Command tests plus scorecard control-pane integration. |

## First five packets

If this is picked up by a fleet, split the first wave like this:

1. Velocity packet: ranks 1-4. Outcome: affected default, duration ledger, shard plan,
   and repair packet.
2. Python tooling packet: ranks 5-6 and 9. Outcome: tooling-quality baseline, ruff
   truth, zero red quarantine.
3. Go debt packet: ranks 7-8 and 16. Outcome: lower code-debt without adding Python.
4. Coverage/witness packet: ranks 10-12 and 18. Outcome: changed-package coverage,
   flake ledger, and stronger negative/render witnesses.
5. Agent-loop packet: ranks 13-15, 17, and 19-20. Outcome: codelint in the write loop,
   CLI contract registry, Windows single command, fixture registry, E2E edit-repair,
   and generated scorecard.
