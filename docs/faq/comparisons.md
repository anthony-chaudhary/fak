---
title: "fak FAQ — Comparisons with other tools"
description: "How fak differs from inference servers, agent frameworks, sandboxes, policy engines, observability tools, and provider-native controls."
---

# Comparisons with other tools

Part of the [fak FAQ](../FAQ.md) — the essentials and every other theme are
indexed there.

You gain a top-level `fak` object on `/v1/chat/completions` and `/v1/messages` responses, present only on turns with tool activity, and a policy refusal arrives as a successful `200` carried as a value rather than an HTTP error. That `fak` extension has an `adjudications` array (one entry per proposed call, with `repaired_arguments` only when the verdict kind is `TRANSFORM`) and a `result_admissions` array (one per inbound result screened, where `QUARANTINE` means the bytes were paged out). HTTP error statuses are reserved for malformed requests, auth failures, and upstream faults, so your client never treats "the kernel said no" as an exception.

## Comparisons with other tools

Where `fak` sits next to inference engines, agent frameworks, sandboxes, and hand-rolled middleware. The recurring theme is layer, not rival.

## Does fak replace vLLM, SGLang, or llama.cpp?

No, `fak` sits in front of them; they are inference engines that turn prompts into tokens, and `fak` is the governance and gateway layer that decides which tool calls run and which results enter context. Point `fak serve --base-url` at a running OpenAI-compatible engine (vLLM, SGLang, or `llama-server`) and your clients move their base URL to `fak`; prompts, tool defs, and the agent loop stay unchanged. `fak` buffers each upstream turn, adjudicates the whole set of proposed tool calls, then re-serializes well-formed SSE, so raw pre-adjudication deltas never pass through. The engines win raw throughput and front-of-prompt prefix caching; `fak` owns capability, quarantine, and audit.

```bash
fak serve --addr 127.0.0.1:8080 --base-url http://localhost:8000/v1 --model qwen2.5-7b
```

## How is fak's gate different from LangChain's tool-calling guards?

A LangChain agent decides which tools to call inside the model loop, so a guard there is advisory; `fak` adds a structural deny floor underneath that the model cannot talk past. `fak serve` speaks the OpenAI and Anthropic wires, so you keep your chains, `@tool`/`StructuredTool` definitions, and `AgentExecutor`/LangGraph loop and change only the chat-model base URL. Every proposed tool call is adjudicated against a reviewable allow-list before it reaches your loop: a tool you never allow-listed is refused regardless of context or injection, and denied calls simply never appear in the model's tool-call list. Your process still runs the surviving tools; `fak` does not execute them for you.

## How does fak compare to an E2B-style sandbox for agent safety?

A sandbox like E2B limits the blast radius of a tool once it runs, while `fak` decides whether the irreversible tool runs at all, before any effect. `fak`'s capability lock is default-deny: a tool that was never allow-listed is refused at the kernel floor, so the dangerous lever is never pulled rather than pulled inside a container. It also gates the result side, holding poisoned or secret-shaped tool outputs out of the model's context entirely (paged to a stub pointer). The two compose: sandbox what does run, and let `fak` decide what is allowed to run and what may enter memory.

## Why use fak instead of a proprietary built-in agent guard from a platform like Replit?

A platform's built-in guard is tied to that platform; `fak` is an open, self-hostable Apache-2.0 Go binary you run yourself in front of any model. Because it speaks the OpenAI, Anthropic, and MCP wires on one port, you point your existing agent's base URL at it and gain a reviewable capability floor, result quarantine, and a trace-correlated audit log without adopting a closed runtime. The policy is a manifest you author and version: `fak policy --dump` emits the default floor to edit, `--check` validates it against a closed refusal vocabulary, and a bad manifest is a hard error rather than a silent fall-back to permissive. You can inspect the code, run the offline proofs, and host it on a laptop CPU with no key, model, or GPU.

## What does fak give me that hand-rolled middleware around my model API does not?

Custom middleware can log and block calls, but `fak` ships the hard parts as a kernel: deny-as-value, a closed refusal vocabulary, result quarantine, and a tamper-evident audit journal. A refusal is a successful HTTP 200 carried as a verdict value, not an exception, so your client never treats "the kernel said no" as a transport error; error statuses are reserved for malformed requests, auth failures, and upstream faults. Refusals draw from a fixed 17-code vocabulary (`DEFAULT_DENY`, `POLICY_BLOCK`, `SELF_MODIFY`, `SECRET_EXFIL`, and so on) rather than free text, and each verdict carries a bounded witness naming only the offending rule. The opt-in decision journal hash-chains each event and records content digests, never the arguments or result bytes.

## Isn't fak just a WAF or API gateway for LLM traffic?

No, a WAF or API gateway screens traffic from the outside and typically fails open on a crash or timeout, whereas `fak` puts the permission check on the same in-process call path as the tool call and fails closed. There is no spawned hook and no inter-process round-trip on the decide path: a proposed call folds an in-process adjudicator chain to the most-restrictive verdict, and a tool that was never allow-listed cannot run no matter what the model was talked into. It also reaches places a network gateway cannot: it holds poisoned tool results out of the model's context and can evict a single span from the KV cache. The audit log records tool names, verdicts, and timings keyed by `trace_id`, never request bodies or arguments.

## Can a rate limiter or quota gateway do what fak's capability floor does?

No, a rate limiter caps how often a tool is called, while `fak`'s capability floor decides whether a given effect is permitted at all. The floor is by tool name and is default-deny: an unlisted irreversible tool is refused structurally, and the refusal does not depend on catching an attack. `fak` does have a rate-limit reason code (`RATE_LIMITED`) in its closed vocabulary, but that is one verdict among 17, not the model. The honest scope is that the floor bounds tool names, not the resolved arguments of an allow-listed coarse tool, so you keep exfil-shaped tools off the allow-list and lean on the result-side quarantine for the rest.

## How does fak's result quarantine differ from a guardrails output-content filter?

A typical output filter classifies text and blocks it when a classifier fires, so its protection is only as good as the classifier; `fak`'s guarantee is structural and does not depend on the detector firing. At the moment a tool result would enter context, `fak`'s gate either admits it, pages an oversized-but-benign result out to a sub-2KB pointer, or quarantines a secret/injection/pollution result so its bytes are physically absent from the model's context. The byte-pattern detector that flags suspicious results is treated as roughly 100% evadable by design and false-positive-prone; it is a bonus, never the floor. The load-bearing protection is the quarantine policy plus the default-deny capability lock, two independent gates an attacker must beat at once.

## When should I keep my serving engine and just add fak, versus using fak's in-kernel model?

Keep your serving engine and front it with `fak serve --base-url` for any production workload; `fak`'s in-kernel model is a correctness reference, not a hardened production server. The recommended path with llama.cpp, vLLM, or SGLang is to keep the engine running and point `fak` at its OpenAI-compatible endpoint, moving clients from the engine's URL to `fak`'s. The in-kernel path (`--gguf`, no `--base-url`) loads a checkpoint directly and is bit-exact against a HuggingFace reference on a small llama model, but it has no continuous batching, paged attention, or multi-tenant scheduling, and several of its GPU backends are slower than llama.cpp. Use it to prove the math or for offline correctness work, not to serve a fleet.

## Does fak give me anything an inference engine's prompt cache doesn't?

Yes, `fak`'s KV cache is addressable, so policy can evict a single span from the middle of a kept run; every shipped engine cache (vLLM APC, SGLang RadixAttention, the OpenAI/Anthropic prompt caches) only reuses contiguously from the front. Change context at position N in a front-of-prompt cache and everything after N is recomputed. `fak` owns the cache as a kernel object and keeps the pre-RoPE keys, so it can remove a poisoned result or expired secret from the middle and leave the cache bit-for-bit identical to a run that never saw it, witnessed at `max|Δ| = 0`. The honest fence: this provable mid-run eviction is proven on a synthetic model in `internal/kvmmu` and is not yet wired into the live agent HTTP loop; the front-of-prompt prefix-reuse path is shipped.

## If vLLM already has an --api-key, why front it with fak?

vLLM's `--api-key` is a single bearer token over its routes; `fak` adds a capability floor, result quarantine, and an audit surface on top of auth. Beyond auth, `fak` adjudicates each proposed tool call against a reviewable allow-list, quarantines poisoned tool results out of context, and emits a trace-correlated audit log and Prometheus metrics, none of which a bare API key provides. Its own auth is off by default for loopback but hardens with one flag, `--require-key-env VAR`, which gates every route except `/healthz` and accepts a bearer token or `x-api-key` compared in constant time over SHA-256 digests. You add flags, not new components.

## I already run an API gateway for auth and routing; where does fak fit alongside it?

