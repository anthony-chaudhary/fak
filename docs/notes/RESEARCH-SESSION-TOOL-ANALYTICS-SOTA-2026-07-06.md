---
title: "SOTA survey: agent session / tool-call analytics & observability"
description: "Filed to scope the session-analytics epic (rollups & reports over agent TOOL CALLS: tool-mix, timing, input/output SHAPE, cross-session trends)."
---

# SOTA survey: agent session / tool-call analytics & observability (2026-07-06)

Filed to scope the session-analytics epic (rollups & reports over agent TOOL CALLS:
tool-mix, timing, input/output SHAPE, cross-session trends). Confidence flags are the
researcher's; verify volatile items (esp. OTel `gen_ai.tool.*` keys) against the live
source before coding against them.

## 1. Observability / tracing platforms (per-tool-call schema + rollups)
- **OpenTelemetry GenAI semantic conventions** — `gen_ai.*` span attributes. VERIFIED real
  tool-call keys (from the live registry): `gen_ai.tool.name`, `gen_ai.tool.type`
  (`function`|`extension`|`datastore`), `gen_ai.tool.description`, `gen_ai.tool.call.id`,
  `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`, `gen_ai.tool.definitions`,
  operation `execute_tool`. NOTE: these were MOVED to `open-telemetry/semantic-conventions-genai`
  and marked deprecated in the old registry — align to the live repo.
- **OpenInference (Arize)** — parallel convention, more settled on tools: `span_kind=TOOL`,
  `tool.name`/`tool.parameters`, `input.value`/`output.value`/`mime_type`. Repo
  `Arize-ai/openinference`. Best off-the-shelf schema to copy for a `TOOL` span.
- **LangSmith** — Run tree typed `chain|llm|tool|retriever|...`; inputs/outputs arbitrary JSON;
  latency/token/error rollups; explicit trajectory/tool-sequence evaluators.
- **Langfuse** (`langfuse/langfuse`) — Trace→Observation(`SPAN|GENERATION|EVENT`); GENERATION
  carries usage/cost; sessions group traces; scores.
- **Arize Phoenix** (`Arize-ai/phoenix`) — OTel-native via OpenInference; `TOOL` spans.
- **Helicone** (`Helicone/helicone`) — proxy request logging + Sessions.
- **Braintrust / W&B Weave / Traceloop-OpenLLMetry** (`traceloop/openllmetry`) — span/op trees with
  input/output/metrics; OpenLLMetry emits the OTel GenAI attributes above.
- Two competing conventions: OTel `gen_ai.*` (neutral, volatile) vs OpenInference (Arize, settled).

## 2. Trajectory eval / analysis (per-step schemas)
- **ToolBench/ToolLLM** (`OpenBMB/ToolBench`) — steps = thought → action(name) → action_input(JSON) → observation.
- **SWE-agent** (`princeton-nlp/SWE-agent`) — `.traj` logs of thought/action/observation over a small
  typed command set (bash/edit/search) — a real corpus of typed tool-call sequences.
- **tau-bench** (`sierra-research/tau-bench`) — OpenAI function-call shape; pass^k consistency metric.
- **AgentBoard** (`hkust-nlp/AgentBoard`) — progress-rate + per-step board (trajectory analytics, not pass/fail).
- **BFCL** (in `ShishirPatil/gorilla`) — AST checker parses (fn name, arg names, arg types, values) —
  directly relevant to decomposing tool-arg SHAPE.
- Also: AgentBench, WebArena, SWE-bench, GAIA, API-Bank.

## 3. Tool input/output SHAPE analysis (thin area)
- **BFCL AST arg/type checking** (as above) — closest real prior art for arg-shape decomposition.
- JSON-schema inference: `genson` (`wolverdude/GenSON`), `quicktype` (`glideapps/quicktype`),
  Snowplow `schema-guru`; Avro Parsing-Canonical-Form + Rabin fingerprint for schema fingerprints.
- Data profiling: `ydata-profiling`, Great Expectations, Soda (tabular-first, imperfect fit for nested JSON).
- **GAP:** no named project histograms tool-OUTPUT token/byte size or truncation rate as a first-class feature.

## 4. Sequence / phase analysis of action logs
- **Process mining — PM4Py** (`process-intelligence-research/pm4py`): directly-follows graphs (== tool-transition
  graphs), variant analysis, XES `(case, activity, timestamp)`. Map session→case, tool-call→activity.
- N-gram/Markov over action sequences (clickstream lit; R `clickstream`) — established technique, agent
  application mostly DIY.
- Phase segmentation (recon/edit/verify): no turnkey agent tool; neighbors are `ruptures` (change-point),
  `hmmlearn`. **Open problem.**

## 5. Open-source repos worth reading
`langfuse/langfuse`, `Arize-ai/phoenix`, `Arize-ai/openinference`, `traceloop/openllmetry`,
`open-telemetry/semantic-conventions(-genai)`, `Helicone/helicone`, `AgentOps-AI/agentops`
(agent event taxonomy: `ToolEvent`/`ActionEvent`/`LLMEvent`), `princeton-nlp/SWE-agent`,
`OpenBMB/ToolBench`, `process-intelligence-research/pm4py`, `ShishirPatil/gorilla` (BFCL),
`lunary-ai/lunary` (lower confidence).

## Gaps the field hasn't solved (this epic's target)
Capture + per-call schema is mature (adopt OpenInference / OTel `gen_ai.*`; PM4Py for transition graphs;
BFCL/genson for arg-shape). Genuinely open: (1) tool input/output SHAPE-distribution analytics
(arg cardinality, key-set/type fingerprints, output size/token/truncation distributions per tool);
(2) automated recon/edit/verify PHASE segmentation of trajectories; (3) cross-session longitudinal
tool-mix / arg-shape / output-size TREND reporting keyed on a stable shape fingerprint. fak already has
the data plane (`internal/trajectory` Turn rows, `fak traj export`, `trajhook` scorer registry) — the
shape-distribution + cross-session-trend layer is what's missing and what the epic builds.
