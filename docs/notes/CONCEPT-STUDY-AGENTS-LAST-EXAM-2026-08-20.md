---
title: "Study: rdi-berkeley/agents-last-exam — official artifact-scored work as a fak benchmark lane"
date: 2026-08-20
status: "FILED as #8223 and #8273-#8285; evidence folded into #6629/#6802/#5662/#6572/#5686"
repo: "https://github.com/rdi-berkeley/agents-last-exam"
pinned_sha: 75a3f866535946b67f9a57e4f158eb30ad50be8a
source_license: "Apache-2.0 software; CC-BY-4.0 benchmark data"
borrow_posture: "ADAPT contracts and INSPIRE analysis; no source or task data copied"
parent: "#1063"
---

# Study: Agents' Last Exam `@75a3f866`

## Verdict

Enter ALE as an **official-harness benchmark lane**, not as task code to absorb into
fak. The smallest useful spine runs one public ALE task through ALE-Claw in clean,
counterbalanced direct/fak pairs—with the same source revision, task variant, snapshot,
model, reasoning effort, retry policy, provider image, and wall-clock budget. ALE's
own hidden references and evaluator remain the sole task-score authority; fak owns
only the comparison of cost, latency, context behavior, policy events that actually
cross its seam, and evidence completeness.

This is **Enabling** work. It tests fak on professional, artifact-scored, terminal-plus-
GUI work without turning an external benchmark's model score into a fak claim.

- **For:** operators deciding whether fak improves long-running professional agents.
- **Problem:** fak's benchmark portfolio has coding, terminal, browser-action, and
  tool-policy lanes, but no broad artifact-scored economic-work lane.
- **Today:** `internal/agenticbench` has the right official-grader and raw/fak fences,
  but its result-packet gate is hard-coded to issues #870-#875
  (`internal/agenticbench/result_packet.go:90-105`).
- **Better because:** one official ALE task makes cost per scored outcome and timeout
  residue inspectable under the same task/model/budget instead of comparing unrelated
  leaderboard rows.
- **Witness:** checked-in direct/fak ALE run directories, upstream evaluator JSON,
  trajectory/log artifacts, exact arm identities, and a FAK comparison packet.

Problem checklist: **P1 managed context** is measured through the gateway arm; **P2
net-true efficiency** reports cost/latency per official score and never relabels gross
token reduction as a gain; **P3 bounded adaptation** fixes task/model/effort/budget and
records every difference; **P4 integrated operations** keeps source pin, environment,
grader output, logs, and hashes in one reproducible packet.

## Source and acquisition

- Full git history cloned into the allocated ignored scratch directory
  `_scratch/study-agents-last-exam/repo`; clean `main` pinned at
  `75a3f866535946b67f9a57e4f158eb30ad50be8a` (2026-08-14).
- Observed 2026-08-20: 955 stars, 60 forks, 15 issues (7 open), 56 pull
  requests (11 open, 39 merged), no GitHub releases, and one infrastructure tag.
  These are discovery facts, not quality evidence.
- The paper was read as arXiv:2606.05405v2 alongside the repository. README, docs,
  framework code, representative tasks and graders, tests, full history, tags,
  issues, pull requests, roadmap, licenses, and submodule pins were inspected.
- Software is Apache-2.0 and benchmark content is CC-BY-4.0
  (`README.md:165-170@75a3f866`). Five agent integrations are pinned git submodules;
  their upstream provenance remains separate. This study copies no code or task data.

## Worldview

ALE evaluates the **agent system**, not an isolated model call. It preserves each
harness's loop, tools, memory, and subagents, gives it only the task description, and
compares the artifacts left behind (`README.md:86-111@75a3f866`). Agent, sandbox, and
task are separate dimensions; a run provisions the sandbox, stages inputs, runs the
agent, stages the hidden reference only after the agent stops, evaluates, then records
raw logs and a uniform trajectory (`README.md:113-135@75a3f866`).

Its optimization target is successful economic work across heterogeneous professional
domains. That drives four defaults worth preserving:

1. **Outcome equivalence over harness equivalence.** Different agent implementations
   are comparable only at the official scored artifact boundary.
2. **Deterministic-first grading.** The paper reports 93.2% code-based graders; the
   remainder use narrow LLM judgments composed with code checks. Gate-and-score makes
   a validity or safety precondition capable of forcing zero before quality credit.
3. **Timeouts still leave work.** An agent wall timeout becomes a typed result, but the
   lifecycle still gathers outputs, stages the hidden reference, and evaluates them
   (`ale_run/orchestration/lifecycle.py:326-447@75a3f866`). Partial official scores are
   therefore evidence, not missing runs.
4. **Diagnosis separates knowing, choosing, and doing.** The paper's post-run analysis
   attributes classifiable failures to understanding, approach, or execution rather
   than treating every low score as a tool/runtime failure (Appendix D.3).

The main difference from fak is ownership: ALE owns benchmark tasks, sandboxes, and
graders; fak owns the model/tool mediation seam and the truthfulness of any efficiency
or safety claim made about that mediation. The integration must compose those worlds,
not merge them.

## Fan-out and social-context coverage

The repository pass covered four bounded surfaces: tasks/evaluators, orchestration and
providers, agent integrations/trajectories, and project rationale/history/social
signals. A final critic was biased to find a missed material technique and to disprove
the proposed FAK gaps. Its five declared source claims were independently read back at
the pinned revision: **5/5 confirmed, 0 refuted, 0 unwitnessed**.

The most consequential issue/PR signals were:

- **Issue #7:** reasoning effort is not normalized across harnesses. Any FAK arm
  comparison must record and hold it constant rather than inherit defaults.
- **Issues #26/#27:** per-task disk admission and provider-native artifact transfer are
  real ALE operational gaps; they belong in ALE's provider layer, not fak.
- **Issue #39:** Harbor can supply agents to ALE, but translating full ALE VM/GUI tasks
  into Harbor's container task model is lossy. Run ALE natively.
- **PR #48 (draft):** per-task network isolation is promising but unmerged; treat it as
  WATCH evidence, never as a shipped ALE property.
- **PRs #55/#67:** deterministic source archives and parsed config substitution improve
  reproducibility, but are pending and must not be assumed at the pinned revision.

## Candidate matrix

| Candidate | Source evidence | FAK witness | Disposition |
|---|---|---|---|
| Official ALE direct-vs-fak benchmark spine | ALE-Claw accepts a per-run OpenAI-compatible `base_url` without global environment mutation (`ale_run/agents/ale_claw/config.py:44-58@75a3f866`); its runner composes agent × task × variant (`ale_run/orchestration/experiment_spec.py:15-45,175-200@75a3f866`) | **ABSENT on-axis.** No ALE entry exists in `internal/benchcatalog`; `internal/agenticbench/result_packet.go:90-105` accepts only the closed #870-#875 campaign | **ADAPT / DEFAULT — filed as #8223 under #1063.** Use upstream runner and grader; add only FAK contract, launch/ingest, and comparison surfaces |
| Preserve honest timeout outcome and optional score | ALE evaluates after an agent timeout (`lifecycle.py:326-447@75a3f866`) and stores reward plus timeout status in trajectory final metrics (`ale_run/base_interface/trajectory.py:129-142,316-343@75a3f866`); an evaluation timeout may yield no extracted score | **PARTIAL.** `internal/timeoutphase` classifies timeout phase and AgenticBench gates official artifacts, but no ALE packet proves `(timeout, score|null)` survives ingestion without status promotion | **DEFAULT requirement in #8223**, not a second grader or retry mechanism |
| Import ALE's cross-harness trajectory with explicit loss accounting | ALE-v1.0 records agent identity, tool calls/results, usage, screenshots, nested subagents, continuations, reward, and status (`trajectory.py:1-15,77-142,180-224@75a3f866`) | **PARTIAL.** `internal/trajectory@r21+g394e195591` has provenance-preserving adapters and fidelity receipts; open #6629 already owns standards-shaped ATIF v1.7 conformance, while ArmBench usage still lacks per-field authority/completeness | **SPLIT.** Attach ALE as an external fixture to #6629 for schema loss; #8282 owns authoritative-vs-fallback token/cache/cost reconciliation. ALE-v1.0 is ATIF-inspired, not Harbor ATIF v1.7 |
| Evidence-cited understanding/approach/execution analysis card | Paper Appendix D.3 builds a structured card from run artifacts, requires field-level citations and confirmed-vs-inferred separation, then applies a closed failure taxonomy | **ABSENT above execution categories.** `internal/terminalbench@r20+gcad27c07b7` classifies execution/policy failures but has no claim-level diagnostic artifact | **ADAPT — filed #8278.** Keep the card advisory and model/prompt-provenanced; deterministic evidence resolution, not the generated diagnosis, is the gate |
| Hidden-reference staging and gate-and-score evaluators | `agent_finished` precedes output gather, reference staging, and evaluation (`lifecycle.py:363-460@75a3f866`); representative graders score observable files component-wise | **PARTIAL.** `internal/agenticbench@r12+gdd43207fa9` requires an available official grader and artifact paths, but proves neither chronology nor evaluator evidence rung | **ADAPT receipts, EXCLUDE grader implementation.** #8274 owns the event-order receipt; #8281 distinguishes normalized scalar from raw grader evidence. ALE remains the sole evaluator authority |
| Hybrid CLI+GUI CUA bridge | Unified cross-OS CUA MCP lifts CLI harnesses to desktop work; ALE normalizes tool aliases and preserves tool-call IDs (`trajectory.py:77-109,152-177@75a3f866`) | **DIVERGENT ownership, absent denominator.** A gateway arm sees provider requests but does not automatically own ALE's local desktop effects | **ADAPT claim coverage — filed #8283.** Join trajectory call IDs to FAK events and report mediated/unmediated/unknown by class; build no desktop bridge |
| Provider admission, licensed images, bulk artifact transfer, network policy | Task cards declare snapshot/CPU/RAM/disk/network needs; loader drops disk/network on pinned main; provider image identity ranges from mutable labels to content-addressed receipts; upstream issues #24/#26/#27 and draft PR #48 remain open | **PARTIAL.** FAK has compute/egress controls but no benchmark task-envelope matcher; ALE still owns provisioning | **SPLIT.** #8277 owns FAK's pre-spend admission/observed-environment receipt. Provider implementation and bulk transfer remain upstream/EXCLUDED |
| Rolling public/private benchmark and contamination controls | Paper §3.1 publishes a rotating minority while retaining private tasks; pinned named lists and paper counts already differ | FAK consumes benchmarks but lacks a content-addressed observed task universe in AgenticBench | **EXTEND #6572.** Bind manifest digest/license/QC class and refuse public-to-private absolute-score or treatment-effect transfer; do not author ALE's corpus |
| Translate ALE tasks into Harbor/container packets | Upstream issue #39 documents the full-VM/GUI mismatch | FAK can run external harnesses without task transcoding | **EXCLUDE.** Native ALE execution is smaller and preserves benchmark validity |

## Second-pass issue portfolio

The deeper pass converted the remaining concrete gaps into small, independently
witnessed leaves. #8223 remains the live one-task spine; the rows below are
offline-first contracts or validators, not permission to defer that run.

| Issue | Missing invariant | Cheapest falsifier |
|---|---|---|
| #8273 | evaluator infrastructure/protocol failure, candidate zero, and successful finite `[0,1]` score are distinct | accept `success/0`; refuse `success/null`, `success/NaN`, and judge outage as a score |
| #8274 | hidden-reference isolation needs chronological evidence, not `official_grader.available` | stage reference before `agent_finished`; the packet must refuse |
| #8275 | authoritative events and fallible terminal/transfer artifacts must reconcile | event score `.7` plus missing or `.5` terminal JSON must not yield a complete claim |
| #8276 | queue, agent execution, evaluator, cleanup/total clocks have different owners | `10s + 100s + 30s` must not report `100s` as end-to-end |
| #8277 | task requirements must match an immutable observed environment before spend | declared disk/network-off plus an unsatisfied provider receipt refuses before launch |
| #8278 | diagnostic claims need field pointers, input coverage, confidence, and confirmed/inferred status | remove one cited field; the claim becomes unsupported, not plausible prose |
| #8279 | final returned result must not hide failed attempts or their spend | `failed -> completed` imports two attempts and one declared selection rule |
| #8280 | ALE's native resume key can alias direct and fak arms before execution | a prior direct completion must not skip a same-task fak arm |
| #8281 | stock `eval_result.json` is a normalized scalar, not raw grader evidence | structured reasons disappear in the stock file; FAK must label the higher rung unavailable |
| #8282 | token/cache/cost fields need authority, coverage, and missing-vs-zero semantics | authoritative aggregate `300/$0.005` must not equal fallback `100/$0.001` |
| #8283 | policy claims need a measured mediated-effect denominator | one mediated Bash call plus one unmediated GUI action cannot become “FAK governed the run” |
| #8285 | planned/admitted/attempted/evaluated/scored attrition must survive aggregation | equal completed means with half the hard tasks missing must not compare as equal cohorts |

One non-benchmark Core defect also survived deduplication: #8284 follows closed
#2093 because `internal/idempotency.Store.Do` records only after a nil apply
return and the CLI calls every apply error safe to retry. ALE's post-launch
no-blind-retry/read-back pattern exposes the counterexample: an effect can land,
its response can fail, and a same-key retry can duplicate it. #8284 owns durable
`PENDING` / `UNKNOWN_APPLIED` plus operation-specific read-back under #2063.

Existing owners received the remaining evidence instead of duplicate issues:

- #6802: strict parsed agent config and redacted effective-config provenance;
- #5662: 160 ALE tier memberships versus 152 distinct overall tasks, plus sparse
  model×harness support that cannot justify a causal range ratio;
- #6572: content-addressed task-universe/license identity and public/private claim scope;
- #5686: failure-taxonomy excluded/unclassifiable denominators and classifier reliability.

Independent GitHub read-back confirmed #8273-#8285 open with their declared
labels, milestones, parents, and stable keys. `fak-dev issue contract` over the
read-back reports **13/13 dispatchable, 0 triage-only, 0 refused** under strict
born-routed, project-work, scale, and witness gates.

## Smallest spine and hard fences

The filed work should start with one public, unlicensed task that exercises a real
artifact evaluator and can run on ALE's supported environment. Use ALE-Claw because
its `base_url` is an explicit, run-local gateway seam. Capture counterbalanced direct/
fak pairs from clean environments, then compare:

- official ALE run/evaluation status plus `score|null`; preserve a score returned after
  agent timeout, accept null after evaluation timeout, and never promote timeout to
  completed;
- task/source revision, task ID, model, harness, reasoning effort, retry policy,
  provider image, wall budget, and all arm differences;
- cost/tokens, wall time, cache/context events, and only the policy events fak
  actually observed;
- upstream `eval_result.json`, `run.json`, `trajectory.json`, `events.jsonl`, origin
  logs, output manifest, and hashes.

Do not call one exact pair a benchmark result or infer a population effect from the
small crossover. Do not run the full public suite before this path is reproducible. Do not change a
task, timeout, VM resource, hidden reference, evaluator, or score. Do not publish an
ALE number as a model-only or fak-owned result. Do not equate a gateway-only arm with
CUA effect enforcement.

## Completeness and honest limits

- **Source coverage:** framework, tasks/graders, agent adapters, tests, docs, paper,
  full git history, tag/release surfaces, licenses/submodules, and all issue/PR titles
  were inspected. Representative graders were sampled across exact, structured,
  geometric/visual, behavioral, and free-text modes; 150 task implementations were
  not read line by line because the shared evaluator framework and paper taxonomy
  define the integration seam.
- **Live-run limit:** this study did not spend model or cloud budget and therefore
  makes no ALE performance claim. The filed spine's captured official run is the
  missing witness, not an optional follow-up.
- **Upstream freshness:** open issues and PRs are observations as of 2026-08-20.
  Pending network isolation, archive, config, and agent-adapter work is WATCH-only.
- **Provenance:** no ALE code, task data, reference output, or submodule content was
  copied. All implementation proposals are Go-side adapters/contracts around the
  public upstream interface.
