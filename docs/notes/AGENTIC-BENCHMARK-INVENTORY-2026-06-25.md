---
title: "Fresh agentic benchmark inventory and fak win map"
description: "A dated inventory of current agentic benchmark surfaces and a falsifiable plan for where fak, used as its own harness around Opus-class and GLM-5.2-class models, can win on solve rate, safe solve rate, cost, and evidence."
---

# Fresh agentic benchmark inventory and fak win map

Date: 2026-06-25.

Status: planning note, not benchmark results. Any number here that comes from an
external leaderboard is a routing signal for what to run next, not a fak claim.
Any fak claim still has to land as JSON plus a `BENCHMARK-AUTHORITY.md` entry.

## Verdict

The benchmark field has split into two different games:

1. **Model-ceiling benchmarks** - the leaderboard is mostly the base model plus
   a standard agent scaffold. fak can win only if it improves the scaffold
   without hurting resolve rate. Examples: SWE-bench Verified, SWE-bench Pro,
   DeepSWE, Terminal-Bench.
2. **Agent-contract benchmarks** - the score depends on policy, state,
   tool-result trust, replay, prompt-injection resistance, or long-run workflow
   control. fak can win as a harness, because the kernel changes the task
   contract. Examples: AgentDojo, AgentHarm, tau-bench, ToolSandbox, OSWorld,
   WorkArena, WebArena, BrowseComp.

The right headline is therefore not "fak beats Opus" or "fak beats GLM-5.2."
The falsifiable headline is:

> With the same base model, the fak harness increases **safe completion per
> dollar** and preserves or improves solve rate, while producing a witnessed
> trace that a raw agent does not produce.

The useful model arms are:

- **Opus-class closed frontier** - highest near-term pass@1 ceiling for coding,
  browser, and computer-use tasks. Claude Opus 4.8 is the current concrete arm:
  Anthropic documents it as an Opus-tier model for complex reasoning,
  long-horizon agentic coding, and high-autonomy work, with 1M context, 128k
  max output, mid-conversation system messages, fast mode, and a 1,024-token
  prompt-cache minimum.
- **GLM-5.2 open-weight frontier** - best open/local-control arm to test
  fak-over-vLLM or fak-over-SGLang. The model card lists `glm_moe_dsa`,
  MIT license, vLLM/SGLang use paths, and provider-local deployment. The vLLM
  recipe targets GLM-5.2-FP8 on 8xH200/H20 and 8xB200-class nodes, including
  tool-call parser and 5-token MTP settings.
- **Mixed Opus + GLM harness** - Opus for high-risk planner/verifier steps,
  GLM-5.2 for open-weight long-context worker, summarizer, replay, and
  self-hostable cost arms. This is where fak's per-aspect routing and witness
  gates matter most.

## Inventory

| benchmark family | current shape | why it matters | fak win thesis | next fak artifact |
|---|---|---|---|---|
| **SWE-bench Verified / Full / Multilingual / Multimodal** | Official SWE-bench reports resolved percentage across Full, Verified, Lite, Multilingual, and Multimodal; Verified is 500 human-filtered instances and Multimodal has 517 visual issues. | Still the lingua franca for coding agents, but increasingly saturated at the frontier. | Use as compatibility control, not the main headline. fak should show raw-model vs fak-gateway on identical selected instances, with no unsafe tool execution and lower token/step waste. | `fak swebench compare` 20-task and 500-task arms: `raw-opus`, `fak-opus`, `raw-glm52-vllm`, `fak-glm52-vllm`. |
| **SWE-bench Pro** | Scale's public set has 731 tasks inside a 1,865-task benchmark across 41 professional repos; scoring requires fail-to-pass and pass-to-pass tests. It was explicitly designed for contamination, diversity, underspecification, and reproducibility gaps. | Better discriminator than Verified; realistic multi-file professional work. | fak should compete on safe resolve, cost per resolved task, and trace quality. The harness win is strongest if tool/result gating prevents bad commits or wasted retries without lowering pass@1. | Pro public 50-task slice first, then full public set. Record patch, tests, tool events, deny/quarantine events, cost, steps. |
| **DeepSWE** | 113 original long-horizon tasks across 91 repos and 5 languages, updated 2026-06-24. Its public table includes Opus 4.8 and GLM-5.2 rows under the same mini-SWE-agent scaffold. | Fresh long-horizon coding signal; public table already separates cost, output tokens, and steps. | This is the cleanest "fak as harness" test: same mini-SWE-agent task set, same model, raw vs fak. fak wins if safe pass@1 stays equal or improves and cost/steps fall through caching, canonicalization, refusal, or better routing. | Add a DeepSWE adapter that drives the agent through `fak serve` and emits a compare JSON keyed by task id. |
| **SWE-bench-Live / SWE-EVO** | SWE-bench-Live is auto-updating, multi-language, multi-OS, and has a Windows track. SWE-EVO targets release-sized long-horizon evolution tasks. | These directly address contamination and repo-scale sustained edits. | fak's witness ledger and replayable tool boundary should matter more than on one-shot bugfixes. GLM-5.2 is attractive for local long-context, but Opus should be the ceiling arm. | Start with Live Windows/Go subsets and one SWE-EVO smoke slice; report freshness date and selected issue ids. |
| **Terminal-Bench** | Terminal-Bench 2.0 is a hard CLI benchmark with 89 tasks, unique environments, human-written solutions, and comprehensive tests; Terminal-Bench 3.0 and Science are in development. | Measures the developer-terminal surface fak already gates. | fak can win on "same resolve, safer terminal" by enforcing command/file/network policy, redacting secrets, preserving evidence, and reducing repeated setup. Raw pass@1 still belongs mostly to the base model. | Harbor/Terminal-Bench runner using `fak guard` or `fak serve`, with per-command verdict and test outcome logs. |
| **WebArena / VisualWebArena / VideoWebArena** | WebArena-x is the umbrella for WebArena, WebArena-Infinity, VisualWebArena, and TheAgentCompany. VisualWebArena adds 910 visually grounded tasks; VideoWebArena adds 2,021 tasks tied to 74 tutorial videos. | Browser agents fail on state, visual grounding, task continuation, and reproducibility. | fak wins if it records browser state, gates mutating actions, quarantines poisoned page/tool content, and makes cache invalidation explicit. It will not fix weak visual grounding by itself. | BrowserGym/WebArena adapter with action policy, state checkpoint, and "safe success" score. |
| **OSWorld / OSWorld-Verified** | OSWorld uses 369 real desktop tasks; OSWorld-Verified permits either 369 tasks with manual Google Drive setup or 361 tasks excluding those cases. Official results are run under unified settings. | Desktop/computer-use benchmark with real apps and execution checks. | Opus-class models likely set the ceiling. fak's win is auditability and safe action control across file, shell, and GUI operations. | OSWorld 50-task verified slice under raw Opus vs fak Opus, reporting action denials, reversibility, and success. |
| **WorkArena / WorkArena++ / BrowserGym** | WorkArena uses ServiceNow knowledge-work tasks and BrowserGym. WorkArena++ expands to 682 compositional enterprise tasks. | Enterprise workflow tasks expose planning, memory, form state, and UI complexity. | fak has a direct story: policy-following, durable state, scope labels, form/tool canonicalization, and safe mutation gates. | BrowserGym runner for WorkArena with enterprise "safe-complete" metric: task success plus no policy breach. |
| **BrowseComp / GAIA** | BrowseComp has 1,266 hard-to-find browsing questions; OpenAI's launch notes show browsing alone is insufficient and persistent reasoning/search matters. GAIA tests tool-use, browsing, file, and multimodal assistant work. | Good for research agents and evidence gathering. | fak wins if it reduces ungrounded answers by requiring cited evidence, replayable search traces, and answer witnesses. It should also route cheap search/summarize vs expensive final verification. | BrowseComp adapter with evidence-required answer schema, raw vs fak routing, and answer-with-source witness. |
| **tau-bench / tau3-bench** | tau-bench emulates dynamic conversations between simulated users and tool-using agents under domain policy; the old repo now warns to use tau3-bench for fixed airline/retail plus banking and voice. | Measures policy-following under tool calls and simulated users. | This is a high-probability fak win: tool calls are syscalls, policy is explicit, and pass^k reliability should improve when unsafe or off-policy actions are structurally denied. | tau3 airline/retail/banking compare: raw tool-calling vs fak-adjudicated tools, report pass^1/pass^k and policy breach rate. |
| **ToolSandbox** | Apple's benchmark adds stateful tool execution, implicit state dependencies, on-policy user simulation, and milestone/minefield evaluation. | Directly targets the gap between stateless REST tool calls and real agent state. | fak's canonicalization, result admission, state ledger, and quarantine are aligned with state dependencies and minefields. | ToolSandbox adapter mapping tools to fak tool registry, with milestone score and minefield avoidance score. |
| **AgentDojo** | Dynamic prompt-injection benchmark for agents over untrusted data; includes task suites, attacks, defenses, and a benchmark script. | Security plus utility, not just refusal. | This is the strongest immediate fak-owned win. The kernel's result-side quarantine and capability floor should reduce attack success while preserving utility on benign tasks. | Extend existing `internal/agentdojo` into an external AgentDojo-compatible runner; publish utility, attack success rate, and quarantine reasons. |
| **AgentHarm** | 110 explicitly malicious agent tasks, 440 with augmentations, across 11 harm categories; scores harmful multi-step agent behavior, not chat refusal only. | Measures whether tool-using agents stay dangerous after jailbreaks. | fak can win structurally: even if the model is jailbroken, the tool boundary can deny harmful actions. The key metric is blocked harmful completion with useful benign completion intact. | AgentHarm-safe runner with closed reason codes and "model attempted vs kernel executed" split. |
| **Cybench / CyberSecEval 4 AutoPatchBench** | Cybench has 40 professional CTF tasks with subtasks; CyberSecEval 4 adds AutoPatchBench for automatically patching native-code vulnerabilities. | Dual-use cyber and security-patching signal. | Split offensive and defensive. fak should avoid boosting offensive capability without policy; for AutoPatchBench, fak can help with witnessed patch/test loops and safe command policy. | Defensive-only AutoPatchBench arm first; Cybench only under a bounded safety policy and clear dual-use caveat. |

## Where fak can win first

### 1. AgentDojo and AgentHarm: structural safety wins

These benchmarks ask whether an agent can be hijacked by untrusted content or
misused for harmful multi-step actions. A raw model has to remember the policy
inside the same context that the attack manipulates. fak moves the policy to the
tool/result boundary. That creates an actual harness advantage.

Target result shape:

```text
same base model:
  utility pass rate: raw ~= fak or fak better
  attack success rate: fak << raw
  harmful tool executions: fak == 0 for denied classes
  evidence: every deny/quarantine has a closed reason code
```

Use Opus first for the ceiling and GLM-5.2 second for the self-hostable open arm.

### 2. tau3-bench and ToolSandbox: policy/state wins

Customer-support and tool-use benchmarks reward following domain policy across
multi-turn state. fak's tool registry, canonicalization, and result admission are
native to this shape. This should produce better pass^k reliability than a raw
tool-calling loop, especially on insufficient information, state dependency, and
minefield tasks.

Target result shape:

```text
pass^1 and pass^k:
  raw model/tool-calling scaffold
  same scaffold through fak adjudication

secondary metrics:
  off-policy tool calls denied
  args repaired/canonicalized
  minefields avoided
  user simulator turns saved or added
```

### 3. DeepSWE and Terminal-Bench: harness efficiency wins

These are less likely to show a giant resolve-rate jump, because the model's
engineering skill dominates. But they can show a credible production win:
equal or better pass@1 with fewer unsafe commands, fewer wasted steps, better
cache behavior, and better trace evidence.

Target result shape:

```text
raw-opus vs fak-opus:
  pass@1 non-regression
  cost/task down or safe-pass/task up
  command/file policy violations blocked

raw-glm52-vllm vs fak-glm52-vllm:
  open-weight cost curve
  local/private task trace
  long-context worker viability
```

### 4. SWE-bench Pro and SWE-bench-Live: contamination-resistant coding wins

Verified is useful as a known-good control, but Pro and Live are better
headlines. fak should report:

- resolved percentage;
- safe-resolved percentage;
- average cost and steps per resolved task;
- test pass/fail plus exact command log;
- denied tool calls and quarantined tool results;
- task freshness or contamination guard.

Do not quote a Pro/Live win until the same task ids, same model, and same budget
are run through both raw and fak arms.

### 5. WebArena/OSWorld/WorkArena/BrowseComp: evidence wins

On browser and desktop tasks, fak is not a visual grounding model. It can still
win as the control plane:

- mutating actions require authority;
- page/tool content can be quarantined before it enters model-visible context;
- state checkpoints make failures inspectable;
- final answers can require independent evidence;
- cache resets and stale-state decisions become explicit events.

This is a "safe success and debuggability" win first, not a raw success-rate win.

## Benchmark harness design

Every external benchmark adapter should produce the same compare schema:

```json
{
  "benchmark": "deepswe|terminal-bench|swe-pro|agentdojo|agentharm|tau3|toolsandbox|osworld|webarena|browsecomp",
  "date": "2026-06-25",
  "task_selection": {
    "source": "official|random|fixed-slice",
    "task_ids": [],
    "freshness_cutoff": ""
  },
  "arms": [
    {
      "name": "raw-opus-4.8",
      "model": "claude-opus-4-8",
      "harness": "benchmark-native",
      "tasks": 0,
      "success": 0,
      "safe_success": 0,
      "cost_usd": 0,
      "steps": 0
    },
    {
      "name": "fak-opus-4.8",
      "model": "claude-opus-4-8",
      "harness": "fak-gateway",
      "tasks": 0,
      "success": 0,
      "safe_success": 0,
      "cost_usd": 0,
      "steps": 0,
      "denies": {},
      "quarantines": {},
      "witnesses": {}
    }
  ],
  "invariants": {
    "same_task_ids": true,
    "same_model_budget": true,
    "official_grader": true,
    "raw_and_fak_both_submitted": true
  }
}
```

The additional fak columns are the point. If a third-party leaderboard reports
only pass@1, keep pass@1. For fak's own harness, also report:

- **safe pass@1** - task resolved and no forbidden tool/result event executed;
- **pass per dollar** - resolved tasks per dollar;
- **pass per wall-clock hour** - only when measured, never modeled;
- **attack success rate** - AgentDojo/AgentHarm only;
- **policy breach rate** - tau3/ToolSandbox/WorkArena;
- **witness coverage** - percent of claimed successes closed by external tests,
  graders, or read-back;
- **work eliminated** - tokens, tool calls, or setup steps skipped by cache,
  vDSO, session reuse, fan-out, or route choice.

## Run order

1. **AgentDojo ceiling run** - Opus raw vs fak, then GLM-5.2 raw-vLLM vs
   fak-vLLM. This is the highest-confidence fak harness win.
2. **tau3 or ToolSandbox run** - policy/state task surface; report pass^k and
   minefield/policy breach rate.
3. **DeepSWE 20-task smoke** - same mini-SWE-agent scaffold, raw vs fak, because
   DeepSWE already publishes cost/output/step fields and includes GLM-5.2.
4. **Terminal-Bench 2.0 smoke** - terminal policy and command evidence.
5. **SWE-bench Pro public 50-task slice** - harder coding headline, but only
   after the harness proves non-regression on DeepSWE/Terminal-Bench.
6. **OSWorld or WorkArena slice** - desktop/browser safe-action evidence.
7. **BrowseComp evidence-required run** - use fak routing and answer witnesses;
   treat as research-agent evidence, not browsing UI control.

## Source map

- SWE-bench official leaderboards:
  <https://www.swebench.com/>
- SWE-bench Pro public leaderboard and methodology:
  <https://labs.scale.com/leaderboard/swe_bench_pro_public>
- DeepSWE:
  <https://deepswe.datacurve.ai/>
- SWE-bench-Live:
  <https://swe-bench-live.github.io/>
- Terminal-Bench:
  <https://www.tbench.ai/> and <https://arxiv.org/abs/2601.11868>
- BrowseComp:
  <https://openai.com/index/browsecomp/>
- WebArena-x:
  <https://webarena.dev/>
- VisualWebArena:
  <https://jykoh.com/vwa>
- VideoWebArena:
  <https://videowebarena.github.io/>
- OSWorld:
  <https://os-world.github.io/>
- WorkArena:
  <https://servicenow.github.io/WorkArena/>
- tau-bench / tau3-bench pointer:
  <https://github.com/sierra-research/tau-bench>
- ToolSandbox:
  <https://machinelearning.apple.com/research/toolsandbox-stateful-conversational-llm-benchmark>
- AgentDojo:
  <https://agentdojo.spylab.ai/>
- AgentHarm:
  <https://arxiv.org/abs/2410.09024>
- Cybench:
  <https://cybench.github.io/>
- CyberSecEval 4:
  <https://meta-llama.github.io/PurpleLlama/CyberSecEval/>
- Claude Opus 4.8:
  <https://www.anthropic.com/news/claude-opus-4-8> and
  <https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-8>
- GLM-5.2 model card and vLLM recipe:
  <https://huggingface.co/zai-org/GLM-5.2> and
  <https://recipes.vllm.ai/zai-org/GLM-5.2>

## Review fences

- Do not compare fak synthetic GLM kernel throughput to full GLM-5.2 serving.
  Use `docs/notes/GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN-2026-06-25.md`
  for the category boundary.
- Do not call a raw pass@1 tie a fak win unless safe-pass, cost, latency, or
  evidence coverage improves.
- Do not report a safety win if benign utility collapses. AgentDojo and
  AgentHarm need utility plus refusal/deny.
- Do not mix model improvements with harness improvements. Every public claim
  needs same model, same selected task ids, same budget, raw and fak arms.
- Do not let "same score" hide trace differences. fak's own harness should make
  model attempts, kernel executions, denials, quarantines, and witnesses
  inspectable per task.
