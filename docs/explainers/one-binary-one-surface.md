---
title: "One binary is the whole agent-serving surface"
description: "fak delivers the entire governed agent-serving stack (API surface, capability gate, result containment, audit, and auth) as one Go binary, laptop to fleet."
---

# One binary is the whole surface — laptop to fleet

> The other two explainers ([policy in the kernel](policy-in-the-kernel.md),
> [addressable KV cache](addressable-kv-cache.md)) are about *what* `fak` does. This one
> is about *what you deploy and operate*. It is the answer to a question the throughput
> benchmarks never ask: **when you actually go to serve an agent safely, how many moving
> parts is that, and who owns them?**

## Serving an agent safely is a stack, not a component

A model server turns prompts into tokens. That is one band of the problem. Engines
like **vLLM** and **SGLang** are superb at it: fast, with paged/radix KV caches and
continuous batching. They are production-proven at enormous scale (SGLang has been
reported across 400,000+ GPUs). `fak` does **not** compete with them on tokens per
second, and never claims to. They win that, and they should.

But *serving an agent* is more than serving tokens. The moment a tool-using agent is in
the loop, you also need a longer list of parts:

- An **API surface** your agents speak (OpenAI wire? Anthropic wire? MCP?).
- A **capability gate** that decides which tool calls are even allowed.
- **Result containment** so a poisoned tool result can't walk into the model's context.
- An **audit trail** that says what was allowed, denied, repaired, or quarantined.
- **Auth** in front of all of it.
- **Observability** for the governance decisions, rather than just the token throughput.

A serving engine gives you the *first* band. By design, it does not give you the rest.
vLLM's and SGLang's tool-calling support **parses** tool-call syntax out of the model's
output and hands it to your client. The docs are explicit that validating and executing
those calls is "the caller's responsibility." There is no built-in capability gating, no
tool-result quarantine, and no audit-by-default in the core serving engine. (Their
ecosystems do add routers, load balancers, and production-stack components that are real
and useful. But those exist to *scale throughput* rather than *govern effects*.)

So to actually run a governed agent fleet, the conventional answer is to **assemble** the
rest of the stack around the engine. You bolt on:

- a reverse proxy for auth and endpoint allow-listing (vLLM's own security docs tell you to do exactly this)
- a policy/authorization service
- a result-screening layer
- a logging/audit pipeline
- an MCP bridge

That is four-to-six components. Most of them are separate processes, and most of them are
something you deploy, version, monitor, and secure on their own.

**`fak` is the other half of that stack collapsed into one static Go binary.** It does not
replace the token engine; it fronts it. You `go install` (or `curl | sh`) and run **one
process**, and that process *is* the gateway and the capability gate. It is the quarantine
and the audit trail. It is the auth and the governance observability.

## The two halves

```
            ┌─────────────────────────────────────────────┐
            │            governed agent serving            │
            ├──────────────────────┬──────────────────────┤
            │   the GOVERNANCE +    │     the TOKEN         │
            │   GATEWAY surface     │     engine            │
            │                       │                       │
            │  • OpenAI/Anthropic/  │  • prefill + decode   │
            │    MCP wires          │  • paged/radix KV     │
            │  • capability floor   │  • continuous batch   │
            │  • result quarantine  │  • tensor/pipe/data   │
            │  • audit + tracing    │    parallelism        │
            │  • auth, metrics      │                       │
            ├──────────────────────┼──────────────────────┤
            │   ONE static Go       │  vLLM / SGLang /      │
            │   binary: `fak`       │  llama.cpp / Ollama / │
            │                       │  a cloud provider     │
            └──────────────────────┴──────────────────────┘
              fak owns this half      you keep this half
                                      (or fak fronts it)
```

The split is the point. `fak` doesn't try to be your fast token engine; that's a band
where the incumbents already win and `fak` says so plainly. It owns the band they leave
empty, and it owns it in a single deployable artifact.

## The honest contrast (operational surface, not throughput)

This table is about **what you deploy and operate**, not about who decodes faster. On raw
tokens-per-second, vLLM and SGLang win. That is their job, and they are excellent at it.
The comparison below is confined to operational surface area and governed-agent serving,
where a single Go binary has a real, structural advantage.

| Dimension | vLLM / SGLang (the token engine) | `fak` (the governed-serving surface) |
|---|---|---|
| **What it is** | A token-serving inference *engine* — prompts → tokens, as fast as possible. | A governed-serving *control surface* — an OpenAI/Anthropic/MCP gateway that adjudicates the tool calls a model proposes. Explicitly **not** a faster token engine; it fronts one. |
| **Implementation / runtime** | Python (SGLang's router adds Rust), on a PyTorch + CUDA/ROCm stack with compiled GPU kernels. | A single static Go binary — no Python, no PyTorch, no CUDA toolchain. **Zero external dependencies** (standard library only; there is no `go.sum`). |
| **Process topology** | Multi-process by design: API server + engine-core(s) + per-GPU worker(s) over ZMQ, Ray for multi-node (vLLM); FastAPI server + runtime + a separate Rust router, plus optional prefill/decode-disaggregation processes (SGLang). | One process. The gateway *is* the adjudication kernel. The token engine it fronts is a separate, swappable process (or it owns a small reference model in-binary). |
| **Install / stand-up** | `pip`/`uv` into a fresh CUDA-matched PyTorch env, or a multi-GB Docker image (~8–12 GB compressed in current tags, bundling CUDA + PyTorch by design). Multi-node adds Ray or a router + RDMA transfer engine. | `go install …/cmd/fak@latest`, a single signed binary download, or a `distroless/static` image that is the base **plus one ~13 MB binary** — no shell, no package manager, no libc, runs nonroot. |
| **Hardware** | Built for GPUs (CUDA by default; CPU / ROCm / XPU / TPU backends exist as alternative paths). | No GPU required to run the kernel or gateway — it runs on a laptop CPU. GPU compute for its in-binary reference model is an opt-in build tag, off by default. |
| **Tool calls** | *Parse* tool-call syntax out of model output and hand it to the client; per-model parser only. Validating/executing is the caller's responsibility. | *Adjudicates* each proposed call at the boundary: capability allow-list (fail-closed `DEFAULT_DENY`), argument repair, and result quarantine — returns only the survivors with a per-decision verdict. (Like the engines, `fak` never executes the tool; your client does, on the admitted calls.) |
| **Capability gating** | None built into the engine; `--api-key` protects only `/v1`, and operators are told to add a reverse proxy. | A reviewable, editable capability floor (`fak policy --dump`/`--check`, `--policy floor.json`) enforced fail-closed, with a closed 12-reason refusal vocabulary. |
| **Result quarantine** | Not an engine concern; untrusted tool output is not contained. | First-class: a write-time gate holds secret-shaped / injection / poison results out of context entirely, and tracks taint. |
| **Audit trail** | No built-in audit logging; security docs direct you to log at the reverse proxy. | Per-request JSON access log + per-operation verdict log, correlated by a minted/propagated `X-Trace-Id` — without exposing request bodies, arguments, or result content. |
| **MCP** | Not in the serving engine (MCP is a client/agent concern). | Built in: MCP over HTTP (`POST /mcp`) and over stdio (`fak serve --stdio`), same adjudication applied. |
| **Observability** | Engine-level Prometheus for throughput / latency / KV usage. | Prometheus `/metrics` (HTTP latency/status, verdict counters, kernel counters, vDSO hit ratio) + an authenticated `/debug/vars` snapshot — aimed at the *governance* decisions. |

**The fair reading:** these are top-tier token engines, and the contrast is no knock on
them. The thing they're great at, moving tokens fast, is simply a different job. An agent
platform team spends its nights on a different set of questions: which effects are allowed,
which results may enter memory, what gets logged, and how many components that takes.

## Same binary, two scales

The part that's easy to miss: **the laptop story and the enterprise story are the same
binary.** You don't graduate from a dev tool to a different production system. You add
flags.

| | A developer, locally | A platform team, in a fleet |
|---|---|---|
| **Command** | `fak serve --base-url … --model …` | the same `fak serve`, plus the flags on the right → |
| **Policy** | the compiled-in default floor | `--policy floor.json` — a reviewable allow-list in version control (GitOps-friendly; it's a file, not a Go edit) |
| **Auth** | none (loopback) | `--require-key-env FAK_TOKEN` — bearer or `x-api-key`, constant-time compare |
| **Observability** | `curl /healthz`, glance at `/metrics` | scrape `/metrics` into Prometheus; ship the JSON access logs + `X-Trace-Id` to your SIEM; `/debug/vars` for break-glass |
| **Wires** | point one OpenAI client at it | point Claude Code, Cursor, OpenAI/Anthropic SDKs, or an MCP client at it — no agent-side changes |
| **Footprint** | one binary on your `PATH` | one `~13 MB` container per replica behind your load balancer |

Nothing new gets installed between those two columns. There is no Python environment that
drifts, no CUDA/PyTorch pin to match, no sidecar to keep in lockstep, no second service to
authenticate. The supply-chain surface is one statically-linked Go binary with no
third-party dependency tree: trivial to audit, trivial to pin, trivial to ship into a
locked-down environment. That is what "scales to enterprise without changing shape" means
here: the artifact a developer runs on a laptop is, byte-for-byte the same kind of thing,
the artifact a platform team runs at fleet scale.

## The honest fences (so this stays inside the ledger)

The single-surface story is real, but it is **operational**, and it does not quietly
smuggle in claims the rest of the repo is careful not to make:

- **`fak` is not a faster (or production) token engine.** It owns the governance +
  gateway surface and *fronts* a real engine (Tier 1). Its own in-binary model (Tier 2)
  is a correctness *reference* forward pass (proven bit-exact against HuggingFace), not a
  production serving engine: no continuous batching, paged attention, or multi-tenant
  scheduling. For chat-quality serving, front vLLM / SGLang / llama.cpp / Ollama / a cloud
  provider. See [`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) and the
  [getting-started caveat](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md#4-tier-2--run-the-fused-in-kernel-model).
- **The cache-reuse win is self-host only**, and a few-fold vs a tuned warm-cache stack
  (the eye-catching multiples are vs the naive re-send-everything pattern). An app that
  merely *calls* a frontier API gets the safety floor but none of the reuse savings.
- **Power/energy numbers are simulated**; zero-copy KV co-residence with an *external*
  engine and the fine-tuned adjudication model are labeled stubs; the result *detector* is
  ~100% evadable by design (the floor is the capability lock + containment rather than detection).
- **Respect the incumbents.** vLLM and SGLang are excellent and production-proven; their
  ecosystems (routers, production-stack, load balancers) add real operational features.
  The claim here is narrow and structural: the *core serving engine* has no built-in
  capability gating, tool-result quarantine, or audit-by-default. Those are external
  layers you assemble, and `fak` is that layer as one binary.

→ Every operational fact above is verifiable: [`go.mod`](https://github.com/anthony-chaudhary/fak/blob/main/go.mod) (zero deps),
[`INSTALL.md`](https://github.com/anthony-chaudhary/fak/blob/main/INSTALL.md) (static targets, distroless image), the gateway routes in
[`GETTING-STARTED.md`](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md#3-tier-1--put-fak-in-front-of-a-real-model-the-practical-serving-path),
and the claim tags in [`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md).

*Last updated: 2026-06-21*
