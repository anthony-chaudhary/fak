# Benchmark Contract Map for Mediated Agent Evals

> **Why this exists.** fak mediates an agent's tool calls before they run. To say a
> mediated eval "scored X on benchmark Y" without drift, every mediated benchmark must
> be compared to the *right* public benchmark and report the *right* evidence. This is
> the single map that binds each mediated fak eval to its official public benchmark, its
> official oracle, the fak mediation surface, the artifacts that must be checked in, and
> — explicitly — where the local mediated task is **not** comparable to the public
> leaderboard. Research deliverable for [#908](https://github.com/anthony-chaudhary/fak/issues/908).

> **One shared artifact contract.** Every mediated eval above the structural-safety floor
> gates any result claim through a single provenance packet, `fak.agentic-benchmark-result-packet.v1`
> (defined in `internal/agenticbench/result_packet.go`, scanned by the [#868](https://github.com/anthony-chaudhary/fak/issues/868)
> epic rollup `fak.agentic-benchmark-epic-rollup.v1`). The [#870](https://github.com/anthony-chaudhary/fak/issues/870)
> GLM-5.2/vLLM agentic battery and the agenticbench epic rollup reference the **same**
> artifact contract: a packet missing provenance or oracle fields is rejected, not
> promoted. That keeps GLM/vLLM and the rollup honest on one schema rather than two.

## The matrix

| # | Mediated benchmark | Official source | Task surface | Official oracle | fak mediation surface | Required artifacts (per arm) | Non-comparable caveat | fak contract schema |
|---|---|---|---|---|---|---|---|---|
| [#869](https://github.com/anthony-chaudhary/fak/issues/869) | AgentDojo structural safety floor | AgentDojo corpus | tool-call red-team (injection into tool args) | attack-success-rate (ASR) + benign completion | IFC provenance taint + sink-gate; detector stack | full-stack ASR, benign controls, corpus hash | **Local structural floor only.** Model-free, deterministic, and explicitly not an official AgentDojo leaderboard result or a raw-model arm. | `gate` field in `experiments/agent-live/agentdojo-fak-fullstack-*.json` |
| [#1064](https://github.com/anthony-chaudhary/fak/issues/1064) | AgentDojo external entry — `fak_gateway` registered non-model defense | <https://github.com/ethz-spylab/agentdojo> (PR-into-fork, **not** a leaderboard) | 629-case security cross-product (Workspace 240 + Slack 105 + Travel 140 + Banking 144) + 97-case utility, under a named attack | three coupled columns: `targeted ASR` + `benign utility` + `utility-under-attack` | `FakGatewayDefense(BasePipelineElement)` tool-call admission gate (capability floor + IFC) in the `ToolsExecutionLoop` | module + load/intercept unit test; entry artifact with the three columns provenance-labeled; lineage fields (#9) | **Module BUILT + WITNESSED; public row operator-gated.** `targeted ASR`=WITNESSED (fak floor); `benign/under-attack utility`=OBSERVED, property of the fronted model, NEEDS_KEY. PLACE in the ~0-ASR tier (co-equal CaMeL/MELON), never a win. `result_claim_allowed=false`. | `agentdojo-external-entry.v1` in `experiments/agent-live/agentdojo-fak-gateway-defense-entry-20260627.json` |
| [#870](https://github.com/anthony-chaudhary/fak/issues/870) | GLM-5.2/vLLM agentic battery | vLLM serving + GLM-5.2 | multi-harness agentic suite served through vLLM | serving-readiness witness (the battery gates on a live served model, not a public score) | fak gateway fronting a vLLM-served model | preflight + serving-witness artifacts | **No public agentic-leaderboard comparability.** BLOCKED on H200/vLLM readiness; no GLM-5.2 benchmark number is quotable yet. | `experiments/vllm/glm52-agentic-battery/final-check.json` |
| [#871](https://github.com/anthony-chaudhary/fak/issues/871) | SWE-bench Verified (Opus smoke) | <https://www.swebench.com/> | repo-resolve coding tasks (patch generation) | official SWE-bench harness: `FAIL_TO_PASS` + `PASS_TO_PASS` tests per instance (`report.json`) | tool/agent calls through the fak gateway | raw + fak `predictions.json`, official `report.json` for both arms | **Pre-run contract only.** No solve-rate until the official harness grades both arms over the *same* task ids. | `fak.swebench-opus-smoke-contract.v1` |
| [#872](https://github.com/anthony-chaudhary/fak/issues/872) | SWE-bench Verified via DeepSWE/R2E-Gym | <https://www.swebench.com/> | repo-resolve coding tasks (DeepSWE adapter) | official SWE-bench harness (`report.json`) | DeepSWE/R2E-Gym adapter routed raw vs through-fak | raw + fak `predictions.json`, official `eval.json`, adapter metadata | **Pre-run contract only.** Not a result until both arms produce predictions and the official harness grades them. | `fak.swebench-deepswe-raw-fak-contract.v1` |
| [#873](https://github.com/anthony-chaudhary/fak/issues/873) | ToolSandbox / tau3 | Apple ToolSandbox <https://github.com/apple/ToolSandbox> · tau-bench <https://github.com/sierra-research/tau-bench> | tool-agent policy-state tasks (retail/airline) | benchmark-native success/pass^k + policy compliance | mediated tool calls (fak verdict/evidence per call) | `result_summary.json`, `trajectories.jsonl` for both arms | Local adapter smoke is `SIMULATED_LOCAL_FIXTURE`; an official row needs benchmark-native tau3/ToolSandbox task ids + grader. | `fak.toolsandbox-official-run-contract.v1` |
| [#874](https://github.com/anthony-chaudhary/fak/issues/874) | Terminal-Bench | <https://www.tbench.ai/> | terminal command tasks (bounded shell) | benchmark-native test output per task (`tb-results.json` pass/fail) | mediated terminal commands (fak per-command verdict/evidence) | run dir + `command-log.jsonl` + official test output, both arms | **External-run contract only.** Not official until benchmark-native Terminal-Bench run logs + test output are checked in. | `fak.terminalbench-official-run-contract.v1` |
| [#875](https://github.com/anthony-chaudhary/fak/issues/875) | Browser / computer-use action | BrowserGym <https://github.com/ServiceNow/BrowserGym> · WebArena <https://webarena.dev/> · OSWorld <https://os-world.github.io/> | browser/desktop action tasks | benchmark-native task success/score (WebArena/WorkArena); OSWorld is desktop-state, not yet bridged | mediated browser actions (fak action verdict/evidence) | benchmark-native trace/study dir + `benchmark-score.json`, both arms | **External-run contract only.** OSWorld (desktop) and BrowseComp (answer-scored) are not selected targets yet — desktop/answer bridges need a separate adapter. | `fak.browseraction-official-run-contract.v1` |

### Benchmarks learned but not yet mediated

The issue's SOTA list also names two public benchmarks fak does **not** yet mediate. They are
recorded here so a future eval does not silently claim parity with a benchmark it has not run:

- **AgentBench** (<https://github.com/THUDM/AgentBench>): multi-environment agent evaluation
  (OS, DB, web, KG, etc.). No fak adapter; not comparable until a benchmark-native arm runs.
- **tau-bench** (<https://github.com/sierra-research/tau-bench>): the tau3 line is mediated
  under #873; the original tau-bench (retail/airline tool-agent) is the same family and shares
  the ToolSandbox/tau3 contract, not a separate mediated row.

## What each benchmark scores (so the right oracle is reported)

- **Patch correctness** — SWE-bench/Verified (#871, #872): the official harness runs the repo's
  test suite against the produced patch. fak must preserve the `predictions.json` + `report.json`
  per instance, never a local "looks right".
- **Terminal state** — Terminal-Bench (#874): benchmark-native per-task test pass/fail. fak
  reports `tb-results.json`, not a mediated pass count.
- **Browser/task state** — Browser/computer-use (#875): benchmark-native task success/score.
  OSWorld grades desktop state; BrowseComp grades a final answer — different oracles, do not
  conflate.
- **Tool-agent behavior + policy state** — ToolSandbox/tau3 (#873): success/pass^k **and** policy
  compliance. A "safe pass" that violates policy is not a pass.
- **Safety / injection resistance** — AgentDojo (#869): ASR against a fixed corpus + benign
  controls. fak's value here is the safety gate, not model quality.
- **Serving readiness** — GLM/vLLM (#870): a live served model is the precondition for any
  agentic number; until the serving witness passes, no GLM-5.2 agentic score is quotable.

## What fak must preserve to be comparable

The shared result packet (`fak.agentic-benchmark-result-packet.v1`) enforces this. A packet
is **rejected** unless it carries: `schema`, a live `issue` lane, `status=PASS_RESULT`,
`result_claim_allowed=true`, the parity gates `benchmark_native` + `same_task_ids` +
`same_model` + `same_budget`, `official_grader.available=true`, both a `raw` and a `fak` arm,
the six metric categories (`task_success`, `safe_success`, `cost_or_token_budget`, `latency`,
`policy_events`, `evidence_completeness`), and checked-in `artifacts` that exist on disk.
This is witnessed by `TestBuildRejectsIncompleteResultPacket`
(`internal/agenticbench/rollup_test.go`): a packet missing the official grader, the fak arm,
or a checked-in artifact fails the gate and cannot graduate.

## Which fak value each row measures

A mediated compare must separate the **external benchmark score** (model + harness fidelity)
from **fak-specific added value** (command/action mediation safety, cache economics,
redaction/quarantine, evidence completeness). Each contract's `CompareMetrics` does this
explicitly — e.g. Terminal-Bench reports `benchmark_native_test_success` separately from
`safe_resolve`, `blocked_dangerous_actions`, `unnecessary_blocks`, and
`fak_verdict_evidence_completeness`. Never fold a fak mediation metric into the public
benchmark's headline score.

## Non-comparable caveats (the comparability boundary)

The discipline this map enforces: **do not claim public benchmark comparability when the local
mediated task differs from the official benchmark.**

- Every row above with `result_claim_allowed=false` is a contract or local fixture, **not** a
  public-leaderboard result. Cite it only as contract/adapter evidence.
- A local smoke (e.g. the ToolSandbox `SIMULATED_LOCAL_FIXTURE`, the AgentDojo structural floor)
  may be quoted only as `[SIMULATED]` / local evidence. It must not be promoted into a
  leaderboard, a README headline, or an external benchmark claim.
- Raw and fak arms must share the same task ids, model, budget, image/browser-state, and retry
  policy before any delta is attributable to fak mediation. Without that parity, the delta is
  confounded, not measured.
- fak mediation metrics (denies, quarantines, cache savings) are **additive** value on top of
  the benchmark score; they are never a substitute for the official oracle.

## Cross-links

- Epic rollup and shared packet: [#868](https://github.com/anthony-chaudhary/fak/issues/868)
  (`internal/agenticbench/rollup.go`, schema `fak.agentic-benchmark-epic-rollup.v1`) and the
  [#870](https://github.com/anthony-chaudhary/fak/issues/870) GLM-5.2/vLLM battery, both gated
  by `fak.agentic-benchmark-result-packet.v1` (`internal/agenticbench/result_packet.go`).
- Per-family contracts: `internal/terminalbench/contract.go`,
  `internal/browseraction/contract.go`, `internal/toolsandbox/contract.go`,
  `internal/swebench/deepswe_contract.go`.
- The authoritative numbers (the *what*) live in [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md);
  this map is the *which benchmark, which oracle, which caveat*.
