---
title: "vLLM/SGLang pre-agentic assumptions, and the agent-era repairs"
description: "A source-backed map of the serving assumptions vLLM and SGLang inherited from chat-completion era workloads, plus the concrete fak-shaped repairs for long-running tool-using agents."
---

# vLLM/SGLang pre-agentic assumptions, and the agent-era repairs

Date: 2026-06-25.

Status: synthesis and planning note. This is not a benchmark and not an attack on
vLLM or SGLang. They made rational choices for high-throughput model serving:
short-lived requests, token economics, prefix reuse, GPU utilization, and
replica availability. Agent workloads add a different contract: long-running
program state, tool effects, mutable world reads, fleet fan-out, human
interrupts, and evidence-backed completion.

Verdict:

> The next agent-era serving layer is not "a faster vLLM/SGLang." It is a
> control plane that treats a model request as one syscall inside a long-running
> agent program. Cache hits, placement, streaming, tool calls, and completion are
> verdicts over identity, authority, state, and evidence, not just throughput
> events.

## Current engine contracts checked

Primary-source facts used in this pass:

- vLLM Automatic Prefix Caching caches full KV blocks by prefix-derived block
  hashes, includes extra axes such as LoRA, multimodal hashes, and cache salts,
  and uses LRU-style block eviction:
  <https://docs.vllm.ai/en/stable/design/prefix_caching/>
- vLLM chunked prefill frames scheduling around TTFT, ITL, throughput, and a
  token budget; V1 prioritizes pending decode work before prefill work when
  chunked prefill is enabled:
  <https://docs.vllm.ai/en/stable/configuration/optimization/>
- vLLM tool calling makes the application responsible for handling tool calls;
  in `tool_choice="auto"` mode, arguments are parser-extracted from raw text and
  may be malformed or off-schema:
  <https://docs.vllm.ai/en/stable/features/tool_calling/>
- vLLM metrics expose sampled KV block lifetime, idle-before-eviction, and reuse
  gaps:
  <https://docs.vllm.ai/en/stable/design/metrics/>
- SGLang exposes request scheduling policies (`fcfs`, `lpm`, `dfs-weight`,
  `priority`, `routing-key`) and radix eviction policies (`lru`, `lfu`, `slru`,
  `priority`):
  <https://docs.sglang.ai/advanced_features/server_arguments.html>
- SGLang HiCache organizes KV as GPU L1, host L2, and shared L3 storage; it
  tracks exact local storage metadata but queries L3 metadata on access rather
  than continuously synchronizing it:
  <https://docs.sglang.ai/advanced_features/hicache_design.html>
- SGLang Model Gateway documents a real cache-efficiency tradeoff: with multiple
  replicas, each replica builds its own radix tree and the trees are not
  synchronized:
  <https://docs.sglang.ai/advanced_features/sgl_model_gateway.html>
- SGLang documents that dynamic batching and prefix caching can make repeated
  temperature-0 requests slightly different:
  <https://docs.sglang.ai/references/faq.html>
- SGLang tool choice uses EBNF grammar through the Xgrammar backend for required
  or specific tool calls:
  <https://docs.sglang.ai/advanced_features/tool_parser.html>

## Assumption map

| Pre-agentic assumption | Why it was reasonable | Where agents break it | Agent-era repair | fak surface |
|---|---|---|---|---|
| **The request is the unit.** | Chat and completion APIs naturally start and end around one HTTP request. | An agent run is a trace with turns, tools, waits, retries, resets, and witnesses. The model request is only one step. | Introduce a durable agent-run envelope: `trace_id`, `agent_id`, `goal_id`, `world_epoch`, tool schema digest, authority labels, and witness requirements. | `internal/loopmgr`, `internal/taskmgr`, gateway session state, `fak loop` |
| **A cache hit is a performance event.** | KV reuse is valid when model/token/prefix identity matches; the primary outcome is saved prefill. | A reused span can be byte-correct but illegal, stale, tainted, or incoherent with a later tool result. | Make every cache hit a typed verdict: hit, miss, revalidate, transform, quarantine, or fault, with scope, taint, freshness, and consumer graph. | `internal/cachemeta`, vDSO cache events, provider telemetry, radix KV lowering |
| **Eviction is an economics policy.** | LRU/LFU/priority maximize reuse under memory pressure. | Agent cache entries also die because a witness was refuted, a tool result was quarantined, a policy changed, or a tenant boundary changed. | Split placement from invalidation: LRU chooses what is cold; policy/refutation chooses what is forbidden. | `cachemeta.InvalidationMode`, `PlanExternalInvalidations`, `internal/enginecache` |
| **Whole-prefix reset is enough for remote engines.** | vLLM/SGLang expose practical control-plane resets; exact external middle-span deletion is not the common API. | A poisoned tool result can contaminate only a span. Full reset is safe but expensive; silent partial survival is unsafe. | Prefer exact-span directives when an adapter can prove them; fail closed or whole-prefix reset when it cannot. | `--engine-cache-require-exact-span`, `internal/enginecache`, `internal/xenginekv` seam |
| **Tool calling is an output format.** | Serving stacks should parse or constrain tool-call syntax and leave side effects to the app. | The dangerous part is not just valid JSON. It is whether this principal may call this tool, whether args are safe, and whether the result may enter context. | Treat tool call as a syscall: parse, canonicalize, adjudicate, execute, admit/quarantine result, then update context/cache. | `internal/adjudicator`, `internal/ctxmmu`, `internal/vdso`, gateway tool buffering |
| **Schedulers optimize token latency and GPU utilization.** | TTFT, ITL, TPOT, goodput, and KV utilization are the right model-server KPIs. | Agent latency includes tool wait, policy wait, result admission, witness checks, session reset, fan-out barrier, and human interrupt. | Add a 2-D scheduler: which agent turn runs next, and which compute/tool/cache tier should serve that step. | `fak loop`, `taskmgr`, route/aspect model routing, `cachemeta` placement |
| **Replica routing is locality plus availability.** | Cache-aware routers can prefer the replica with the prefix while preserving HA. | An agent route also carries authority: tenant, trace, taint, tool-result provenance, and which world snapshot the route may see. | Route on `(KV locality, load, health, authority, taint, world_epoch)`, not locality alone. | gateway `ReplicaRouter` today is static round-robin; residency/health index is the gap |
| **L3 cache cells are opaque bytes addressed by hashes.** | Disaggregated KV needs a fast data path; per-read semantic verification would destroy latency. | Agents need provenance, deletion proof, access control, and state validity on shared cache artifacts. | Keep the data path fast, but add a control-path sidecar: signed manifests, return-digest verification, scope labels, and deletion certificates. | `L3-DISAGGREGATED-CACHE-REIMAGINED.md`, `deletioncert`, `cachemeta.KVManifest` |
| **Determinism is a quality detail.** | Dynamic batching and prefix caching may perturb exact repeatability while keeping output quality acceptable. | Agents need replayable accountability: "same enough" is fine for prose, not for witness, cache admission, or policy proof. | Separate model nondeterminism from kernel determinism. Record the exact prompt/materialization tuple, adjudication verdict, cache verdict, and independent evidence. | hash-chained journals, `taskmgr.WitnessRecord`, `loopmgr` witness rows |
| **Observability ends at token serving.** | Model operators need latency, throughput, KV residency, queue depth, and errors. | Agent operators ask: did the loop actually run, what did it prove, what got refused, what cache verdicts hid work, and where budget burned without evidence? | Add agentic KPIs beside serving KPIs: witness rate, claim/refusal split, budget burn without evidence, cache verdict by plane, tool wait, reset count. | `AGENTIC-LOOP-KPIS-2026-06-25.md`, `fak loop status`, cache stream metrics |
| **The model is selected for the request.** | A router normally chooses one backend for the whole completion. | Agents have heterogeneous steps: cheap parse, local preflight, frontier reasoning, tool result summarization, verification. | Route per aspect or step, with ensembles as first-class reductions, while every routed call still crosses residency/adjudication. | `cmd/fak/route.go`, `docs/model-routing.md`, gateway route dispatch |
| **Inbound context is the application's job.** | The server sees serialized messages and tool definitions after the app has composed them. | Tool definitions, skills, memory, and system text are themselves attack and cache-break surfaces. | Add an ingress prompt-MMU that prunes/compacts tool definitions and records serializer/cache-break identity. | `INBOUND-PROMPT-MMU-2026-06-25.md`, `internal/promptmmu` |

## The better split

Keep vLLM/SGLang excellent at what they are excellent at:

- paged/block KV management;
- continuous batching and chunked prefill;
- tensor/pipeline/data-parallel serving;
- prefix/radix/HiCache reuse;
- structured output and tool-call grammar;
- prefill/decode disaggregation;
- engine-level metrics and GPU efficiency.

Put the agent-era contract around and below those calls:

- identity and authority travel with the request;
- cache reuse is allowed by a policy/witness verdict;
- tool calls are syscalls, not just parsed strings;
- result admission happens before context visibility;
- placement respects taint, tenant, world state, and cache locality;
- long-running loops have durable ledgers and witnessed completion;
- metrics report agent progress and evidence, not just tokens.

This preserves the engine value while refusing the category error: an agentic run
is not "just a longer prompt."

## Build sequence

1. **Normalize an agent-run envelope.** Add a small shared wire schema for
   `trace_id`, `agent_id`, `goal_id`, `turn_id`, authority labels, world epoch,
   tool schema digest, and prompt serializer hash. Use it in gateway logs,
   cachemeta events, and benchmark records before changing routing behavior.

2. **Make engine-cache actions emit cachemeta verdicts.** Today remote cache
   resets are witnessed through `internal/enginecache`; the next step is to
   record the reset as `PlaneProvider` / `PlaneKVPrefix` invalidation events,
   including whether the adapter performed exact-span eviction, whole-prefix
   reset, miss, or fault.

3. **Add a residency and authority index beside `ReplicaRouter`.** Keep the
   current static round-robin as the safe floor. Add an optional index that can
   answer: which replica has the prefix, which tenant/taint scopes may route
   there, and whether a world epoch mismatch forces revalidation.

4. **Wire cache verdicts into benchmarks.** Every vLLM/SGLang/fak comparison
   should report local KV hit, provider prompt-cache hit, remote reset count,
   revalidate/quarantine/fault, and tool wait separately. A single blended hit
   rate is no longer a meaningful agent metric.

5. **Promote L3 semantics sidecar as the external-engine bridge.** Do not put
   fak on the hot KV data path. Put it on admission, manifest validation,
   witness/refutation, scope, and deletion certificate generation.

6. **Teach the loop scheduler cache locality.** Once `fak loop` can see which
   logical loops are armed and which cache prefixes are warm, schedule green
   agent threads toward warm gateways unless authority or witness health says
   no.

## Review rules

Refuse these claims during review:

- "vLLM/SGLang already handles agents" when the evidence is only tool-call
  syntax or `/v1/responses` compatibility.
- "Cache-aware routing is enough" when the route ignores authority, taint, and
  world state.
- "Prefix cache hit proves safety." It proves reuse eligibility on engine axes,
  not policy or freshness.
- "Whole cache reset is exact deletion." It is a safe fallback, not a precise
  span proof.
- "Temperature 0 means replayable." Dynamic batching and prefix caching can
  still perturb outputs; replay claims need recorded inputs and independent
  witnesses.
- "Tool-call JSON is a syscall." A valid schema is only a candidate syscall; the
  syscall starts at adjudication.

## Related in-tree artifacts

- [`AGENTIC-CACHING-SOTA-2026-06-19.md`](AGENTIC-CACHING-SOTA-2026-06-19.md)
- [`THROUGHPUT-TRUST-SHARED-SPINE-2026-06-24.md`](THROUGHPUT-TRUST-SHARED-SPINE-2026-06-24.md)
- [`L3-DISAGGREGATED-CACHE-REIMAGINED.md`](L3-DISAGGREGATED-CACHE-REIMAGINED.md)
- [`AGENTIC-LOOP-KPIS-2026-06-25.md`](AGENTIC-LOOP-KPIS-2026-06-25.md)
- [`INBOUND-PROMPT-MMU-2026-06-25.md`](INBOUND-PROMPT-MMU-2026-06-25.md)
- [`LONG-RUNNING-AGENT-LOOPS-2026-06-25.md`](LONG-RUNNING-AGENT-LOOPS-2026-06-25.md)
