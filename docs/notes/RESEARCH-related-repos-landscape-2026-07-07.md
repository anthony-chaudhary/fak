---
title: "Related-repos & SOTA landscape for fak/DOS (2026-07-07)"
description: "A verified prior-art map of repos, papers, and tools adjacent to fak/DOS across three intents — inspiration, related/neighbors, and directly-usable-for-dev — with the durable searches that keep it fresh. Complements the automated idea-scout and the docs/explainers/sota-optimizations prior-art docs."
---

# Related-repos & SOTA landscape (2026-07-07)

A fresh landscape sweep for work adjacent to **fak/DOS**, organized by three intents:
**inspiration** (ideas/architectures to learn from), **related** (neighbors & competitors),
and **usable** (libraries/tools/datasets to integrate, vendor, or benchmark against).

- **Method:** 15 topic clusters swept via parallel web research, then every low-confidence
  find (2026-dated arXiv, obscure repos) adversarially existence-checked. Result: **~200
  finds, zero refuted** — all `needs_verify` items resolved to real papers/repos (a few URL
  corrections only). Star counts are search-snippet-approximate, not repo-read.
- **Complements, does not duplicate:** the automated [idea-scout](../idea-scout.md) (6 baked-in
  topics) and the prior-art docs ([sota-optimizations](../explainers/sota-optimizations.md),
  [pd-disaggregation-kv-routing-sota](../serving/pd-disaggregation-kv-routing-sota.md)). Already
  covered there and intentionally not re-listed: vLLM, SGLang/RadixAttention, LMCache, llama.cpp,
  Ollama, TensorRT-LLM, DeepSpeed, AWQ, GPTQ, FlashAttention, TGI, MLX, Mooncake (paper).
- **Operational half:** the durable search queries are shipped as a drop-in scout config,
  [`tools/idea_scout_topics_fresh.json`](https://github.com/anthony-chaudhary/fak/blob/main/tools/idea_scout_topics_fresh.json)
  (6 defaults + 13 fresh topics). Run `python tools/idea_scout.py --config
  tools/idea_scout_topics_fresh.json --json` (dry-run) to preview candidates.

## Bucket 1 — Inspiration (learn from)

The strongest external analog to the whole DOS thesis is a cluster of 2026 "agent OS / kernel"
papers converging independently on fak's exact design — mediate every tool call as a syscall,
verify actions from evidence, refuse structurally:

| Work | URL | Maps to |
|---|---|---|
| **ActPlane** — OS-level agent policy engine, IFC DSL enforced via eBPF, *semantic structured refusals*, cross-event gates | arxiv.org/abs/2606.25189 · github.com/eunomia-bpf/ActPlane | capability floor + structured refusal (closest external twin) |
| **Agent Control Protocol (ACP)** — admission check over the execution *trace* (capability scope, delegation, cooldown), Ed25519 identity | arxiv.org/abs/2603.18829 | `dos_arbitrate` admission kernel + trace-level (not per-call) policy |
| **AgenticOS** (Tencent) — OS as an "intent filter": least-capability env synthesis + mandatory mediation + IFC | arxiv.org/abs/2606.21129 | default-deny floor + result quarantine + closed refusal vocabulary |
| **Securing AI Agents Like Operating Systems** (Holz/Rieck) — Tools↔Syscalls via a "system mediator" | arxiv.org/abs/2605.14932 | fak's "tool call is a syscall" thesis, stated as a security architecture |
| **Right to History** — Rust "sovereignty kernel" emitting RFC-6962 Merkle audit logs of every agent action; critiques AIOS for trusting the LLM | arxiv.org/abs/2602.20214 | `dos_commit_audit`/`dos_verify` — verify actions from external evidence |
| **CaMeL** (DeepMind) → **FIDES** → **Progent** → **MVAR** | 2503.18813 · 2505.23643 · 2504.11703 · github.com/mvar-security/mvar | the IFC-by-design lineage that formalizes fak's provenance denies |
| **AIOS** + "LLM as OS, Agents as Apps" | github.com/agiresearch/AIOS · arxiv.org/abs/2312.03815 | the founding LLM-OS analogy fak's "agent kernel" framing descends from |
| **CoAgent** — DB concurrency-control theory applied to multi-agent systems | arxiv.org/abs/2606.15376 | `dos_arbitrate` (named academic anchor) |
| **MINJA** — query-only memory-poisoning; "agents treat retrieved memories as ground truth, no provenance checks" | arxiv.org/abs/2503.03704 | `dos_recall`'s exact motivating threat |
| **KVzip** (NeurIPS'25 Oral) — evict a KV span only if the model can reconstruct context from it | arxiv.org/abs/2505.23416 | policy-driven / bit-exact KV eviction (`kvmmu`) |
| **Defeating Nondeterminism in LLM Inference** (Thinking Machines) — batch-invariant kernels → 1000 identical runs | thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference · github.com/thinking-machines-lab/batch_invariant_ops | fak's bit-exact-vs-HF-oracle + backend-conformance thesis |
| **Reproducible Builds** | reproducible-builds.org | "determinism itself is the evidence; a diverging byte is a refusal" |
| **BenchJack** (Berkeley RDI) — an agent scored ~100% on 8 benchmarks *without solving tasks* | arxiv.org/abs/2605.12673 | third-party proof of the claim-vs-evidence gap `dos_review`/`dos_commit_audit` close |
| **EAGLE-3** — draft-head speculative decoding (the gap fak lacks) | arxiv.org/abs/2503.01840 | in-kernel decode roadmap |

## Bucket 2 — Related (neighbors & competitors)

**LLM gateways** (fak serve's direct competitor set): [LiteLLM](https://github.com/BerriAI/litellm),
[Portkey](https://github.com/Portkey-AI/gateway), **[Bifrost](https://github.com/maximhq/bifrost)** (Go,
closest posture peer), [Arch/archgw](https://github.com/katanemo/archgw),
[TrueFoundry](https://www.truefoundry.com/ai-gateway) (its `mcp_pre_tool`/`mcp_post_tool` hooks are fak's
exact seam), [TensorZero](https://github.com/tensorzero/tensorzero),
[Envoy AI Gateway](https://github.com/envoyproxy/ai-gateway), [Higress](https://github.com/alibaba/higress),
[vLLM Semantic Router](https://github.com/vllm-project/semantic-router). **The distinction fak keeps:
~15 gateways do content-guardrails + routing; none do evidence-verifying adjudication + quarantine +
closed structured refusal.**

**Coordination / worktree / fleets** (`dos_arbitrate` neighbors):
**[Forge Orchestrator](https://github.com/nxtg-ai/forge-orchestrator)** (closest whole-system twin — a
deterministic policy kernel + file-locking), **[wit](https://github.com/amaar-mc/wit)** (finer-grained
symbol-level AST locks), [swarm-protocol](https://github.com/phuryn/swarm-protocol),
[ccswarm](https://github.com/nwiizo/ccswarm), [Claude Squad](https://github.com/smtg-ai/claude-squad),
[Vibe Kanban](https://github.com/BloopAI/vibe-kanban), [OpenHands](https://github.com/All-Hands-AI/OpenHands),
[A2A](https://a2a-protocol.org)/[ACP](https://github.com/agentclientprotocol/agent-client-protocol).
**[AgenticFlict](https://arxiv.org/abs/2604.03551)** is a merge-conflict dataset usable to *benchmark*
whether fak's tree-disjointness actually cuts collision rate.

**Agent memory** (`dos_recall` neighbors): [Mem0](https://github.com/mem0ai/mem0) (openly concedes the
staleness gap DOS closes), **[Zep/Graphiti](https://github.com/getzep/graphiti)** (bitemporal KG — the
principled upgrade path), [cognee](https://github.com/topoteretes/cognee).

**Guardrail / red-team** (the "detector fak positions as NOT-the-floor"):
[garak](https://github.com/NVIDIA/garak), [PyRIT](https://github.com/microsoft/PyRIT),
[NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails),
[Llama Guard / Prompt Guard](https://huggingface.co/meta-llama/Llama-Guard-3-8B),
[LlamaFirewall](https://arxiv.org/abs/2505.03574). The load-bearing citation for the "detector is
~100% evadable by design" thesis: **[Bypassing LLM Guardrails](https://arxiv.org/abs/2504.11168)**
(character-injection → up to 100% evasion vs Azure Prompt Shield + Meta Prompt Guard). Third-party
support for the architectural floor: **[AgentDojo](https://arxiv.org/abs/2406.13352)** finds
system-level defenses reach near-zero ASR where detectors only hit ~8%; **[OWASP Agentic Security
Initiative](https://genai.owasp.org)** makes "least agency" the headline principle.

## Bucket 3 — Directly usable for dev (Go-native flagged)

| Need | Vendor / integrate |
|---|---|
| Policy engine for the capability floor | **[cedar-go](https://github.com/cedar-policy/cedar-go)** (formally verified, default-deny) · **[OPA/Rego](https://github.com/open-policy-agent/opa)** (compiles to Wasm) · **[cel-go](https://github.com/google/cel-go)** |
| Run untrusted tool/policy logic in-process | **[wazero](https://github.com/tetratelabs/wazero)** (pure-Go Wasm) · **[Extism](https://github.com/extism/go-sdk)** (obj-capability broker) |
| Sandbox tool execution | **[go-landlock](https://github.com/landlock-lsm/go-landlock)** · **[firecracker-go-sdk](https://github.com/firecracker-microvm/firecracker-go-sdk)** |
| MCP server/client + defense | **[official MCP go-sdk](https://github.com/modelcontextprotocol/go-sdk)** · **[mcpproxy-go](https://github.com/smart-mcp-proxy/mcpproxy-go)** (single-binary Go MCP proxy that auto-*quarantines* new servers + human-approval-before-exec — the closest external analog to fak's tool defense) · **[mcp-scan](https://github.com/invariantlabs-ai/mcp-scan)** |
| Witness / provenance / signing (DOS ledger) | **[sigstore-go](https://github.com/sigstore/sigstore-go)** (returns a result only if integrity holds; forces artifact-digest binding) · **[Trillian](https://github.com/google/trillian)** (verifiable Merkle log) · **[gittuf](https://github.com/gittuf/gittuf)** (in-repo signed policy + RSL, verify without trusting the forge — the strongest analog to DOS's git-evidence-not-forge-trust thesis) · **[in-toto witness](https://github.com/in-toto/witness)** · **[DSSE / go-securesystemslib](https://github.com/secure-systems-lab/go-securesystemslib)** |
| Determinism / bit-exact witness | **[batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops)** |
| Go agent/gateway peers to front or interop with | **[cloudwego/eino](https://github.com/cloudwego/eino)** · **[google/adk-go](https://github.com/google/adk-go)** · **[firebase/genkit](https://github.com/firebase/genkit)** (tool-interrupt HITL ≈ fak's capability gate) · **[mudler/LocalAI](https://github.com/mudler/LocalAI)** (single-Go-binary OpenAI+Anthropic gateway) · **[ollama native Go engine](https://github.com/ollama/ollama)** (ml/·kvcache/ — closest Go peer to the in-kernel transformer) |
| Context / tool-schema compression | **[LLMLingua](https://github.com/microsoft/LLMLingua)** · [Anthropic Tool Search Tool `defer_loading`](https://www.anthropic.com/engineering/advanced-tool-use) · [langgraph-bigtool](https://github.com/langchain-ai/langgraph-bigtool) — the productized twins of `fak_tools_search` |
| KV eviction / disaggregation to benchmark against | **[KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory)** (unifies the eviction family) · **[SafeKV](https://openreview.net/pdf?id=jhDsbd5eXL)** (per-block privacy labels ≈ L3 `ShareScope`) · [NVIDIA Dynamo/KVBM](https://developer.nvidia.com/dynamo) · [InfiniStore](https://github.com/bytedance/InfiniStore) |
| Agent-eval harnesses / datasets | **[Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai)** (closest OSS analog to fak's scorer + trajectory-audit) · [tau2-bench](https://github.com/sierra-research/tau2-bench) · [AgentDojo](https://arxiv.org/abs/2406.13352) · [MCP-Bench](https://github.com/Accenture/mcp-bench) · [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) |

## Sharpest signals for fak/DOS

1. **fak/DOS is not alone — a 2026 "agent OS" wave is converging on its exact design** (ActPlane,
   Agent Control Protocol, AgenticOS, Holz/Rieck, Right to History, AIOS). fak's differentiators
   within it: pure-Go single binary, in-process (no eBPF/sidecar), *and* the claim-verification
   substrate (dos_commit_audit/verify/review) most of these lack. Cite ActPlane + ACP in prior-art.
2. **The IFC-by-design lineage (CaMeL→FIDES→Progent→MVAR) is the academic backbone** for fak's
   provenance denies — currently asserted, not formalized. A label-lattice adoption is a real option.
3. **Provenance/attestation (Sigstore/in-toto/gittuf/Trillian) is DOS's witness model, already built
   and Go-native** — the highest-leverage vendorable cluster.
4. **The gateway market validates fak's position but not its mechanism** — none adjudicate+quarantine+
   refuse-structurally. `mcpproxy-go` is the nearest external twin of the tool-defense floor.
5. **Honest gaps:** no speculative decoding (EAGLE-3 is standard); the fused cross-session KV pool
   (a labeled stub) has many serving-side neighbors (Dynamo/KVBM, Mooncake Store, InfiniStore, SafeKV)
   to borrow transport + cross-tenant-admission designs from.

## Keeping it fresh (durable searches)

The operational half is [`tools/idea_scout_topics_fresh.json`](https://github.com/anthony-chaudhary/fak/blob/main/tools/idea_scout_topics_fresh.json)
— 13 fresh topics (claim-verification, provenance-attestation, agent-coordination-locking,
agent-memory-verification, kv-cache-eviction, capability-ifc, tool-sandboxing, deterministic-inference,
llm-os-agent-kernel, mcp-tooling-defense, context-tool-disclosure, agent-eval-benchmarks,
kv-disaggregation) on top of the 6 defaults. Meta-lists worth re-scraping monthly:
`ai-boost/awesome-harness-engineering`, `andyrewlee/awesome-agent-orchestrators`, `korchasa/awesome-mcp`,
`cuihuan/awesome-ai-gateway`.
