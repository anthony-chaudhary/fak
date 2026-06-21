# Frequently Asked Questions (FAQ)

Direct answers to the most common questions about `fak`, the agent kernel. Each
answer is written to stand on its own. For the full story, start with the
[README](../README.md); for runnable proof, see the [2-minute repro](repro-packet.md).

---

## What is fak?

`fak` is an **agent kernel** — an in-process, default-deny **permission gate** for AI
agents, fused with an **addressable, bit-exact KV cache**, written in Go. It treats the
language model like an untrusted program and every tool call like a syscall that must
pass through a kernel the model cannot control. The same boundary enforces security
(which effects are allowed, which tool results may enter the model's context) and drives
performance (do shared work once instead of every turn). It is also described as an
**agent tool firewall**.

## What problem does fak solve?

It closes the gap between agent **safety** and agent **cost** at the same boundary:

1. **Prompt injection and tool poisoning** reach the model through tool results. `fak`
   quarantines suspicious results so they never enter the model's context.
2. **Irreversible actions** (refunds, deletes, sends) are gated by a reviewable
   allow-list that is checked inside the kernel — default-deny, fail-closed.
3. **Agent fleets waste tokens** re-processing the same shared context every turn. `fak`
   makes the KV cache a kernel object so shared work is computed once and reused.

## How is fak different from a normal firewall or API gateway?

A normal firewall or gateway screens traffic from the *outside* and typically fails
**open** when it crashes or times out. `fak` puts the permission check on the *same call
path* as the tool call — one address space, no inter-process call — so it is something
the call passes *through*, like `read()` through an OS kernel. It is **default-deny**: an
action that was never allow-listed cannot run, no matter what the model was talked into.

## How does fak prevent prompt injection?

With two independent gates, not one classifier:

- **The capability lock.** A dangerous tool is simply not on the allow-list, so no
  amount of injected text changes the answer — the lever was never wired up.
- **Result quarantine.** Suspicious tool *results* are held out of the model's context
  entirely, so a booby-trapped document never reaches the model to influence it.

The detector that *flags* suspicious results is deliberately treated as evadable (~100%
evadable by design) — it is a bonus, never the floor. An attacker has to beat two
structural gates, not fool one screener. In live tests, prompt injection reached the
unprotected baseline 5/5 and `fak` walled it off 5/5.

## Does fak address the OWASP Agentic Top-10 and the MCP Top-10?

Yes — structurally. It targets **Tool Poisoning (MCP03)** and **Memory Poisoning (T1)**
by keeping untrusted tool results out of the model's context (containment) and by gating
which effects are even possible (the capability floor). It does not rely on recognizing
each attack; it relies on the dangerous lever not existing and the poisoned bytes not
arriving.

## What is an addressable KV cache?

A **KV cache** is the scratchpad a model builds as it reads, so it doesn't re-read from
scratch each turn. Every shipped engine (vLLM, SGLang, the OpenAI/Anthropic prompt
caches) only reuses it *from the front*: change anything in the middle and everything
after is recomputed. An **addressable** KV cache lets policy reach into the *middle* of a
kept run, evict a single span (a poisoned result, an expired secret), and leave the cache
**bit-for-bit identical** to a run that never saw it — verified at `max|Δ| = 0`. `fak`
can do this because it owns the cache as a kernel object instead of renting it from a
serving engine. See [Addressable KV cache](explainers/addressable-kv-cache.md).

## Is fak a faster model server? How does it compare to vLLM, SGLang, or llama.cpp?

No. `fak` is **not** a faster model server, and it does not try to beat vLLM, SGLang, or
llama.cpp at raw throughput or front-of-prompt prefix caching — those engines win that,
and `fak` measures itself against them honestly, not against a strawman. `fak` owns the
*orthogonal* questions they don't: which effects are allowed, which results may enter
memory, when reuse is still legal, and what survives a session boundary. You can even run
`fak serve` *in front of* one of those engines and keep using it.

## How much faster is fak for agent fleets?

The win is in **reread-rate**, not raw GPU speed. On a 50-turn × 5-agent run it is about
**4× fewer tokens than a tuned warm-cache stack** (the honest few-fold vs. state of the
art; ~60× vs. the naive re-send-everything pattern, which is easy to beat and not the
headline). On real WebVoyager web-agent workloads (643 tasks) it eliminates **8.8–9.7×**
of prefill, measured. The reuse win is **self-host only** — an app that merely *calls* a
frontier API gets the safety floor but not the savings. Every number is traced to a
commit and artifact in the [benchmark authority](../BENCHMARK-AUTHORITY.md).

## Is fak novel? What did the prior-art audit find?

A 29-claim prior-art audit scored **0/29 novel**. Every individual primitive
(capability security, quarantine, KV caching, content-addressed storage) is established
prior art. The contribution is the **assembly**: putting them together as one in-process
gate where the tool call is the checkpoint, so the security boundary and the reuse
boundary become the same boundary. `fak` is built to survive a skeptic reading the code —
see the [claims ledger](../CLAIMS.md), where every capability carries one
machine-checked tag.

## How do I install fak?

One static binary, no clone or Go toolchain required:

```bash
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh
```

Or download a [prebuilt archive](https://github.com/anthony-chaudhary/fak/releases/latest)
(`linux_amd64`, `darwin_amd64`, `darwin_arm64`, `windows_amd64`), or run it in a
container. Full guide: [Getting Started](../GETTING-STARTED.md).

## Can I try fak without a model, API key, or GPU?

Yes. With just [Go 1.26+](https://go.dev/dl/):

```bash
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"
go run ./cmd/fak agent --offline
```

`refund_payment` returns `DENY (POLICY_BLOCK)`; `search_kb` returns `ALLOW`; and
`agent --offline` runs the same task twice (tools wired directly vs. behind `fak`) and
prints the before/after. Full walkthrough: [repro packet](repro-packet.md).

## What language and license is fak?

`fak` is written in **Go** (requires Go 1.26+ to build from source) and licensed under
**Apache-2.0**.

## How do I put fak in front of my existing model?

`fak serve` fronts any OpenAI-compatible server (Ollama, vLLM, a cloud provider). You
keep your model and stack and gain a reviewable allow-list, result quarantine, and an
audit trail:

```bash
fak policy --dump > floor.json   # a starter allow-list you can edit and review
fak serve --addr 127.0.0.1:8080 --base-url http://localhost:11434/v1 --model qwen2.5:1.5b
```

This is where most people should start; it is a complete product by itself. See the
[getting started guide](../GETTING-STARTED.md).

## Who is fak for?

Teams running **self-hosted LLM agent fleets** who need three things at once:
prompt-injection containment, reviewable capability security, and cache-efficient
inference. It is useful at every rung — front your existing model for the safety floor,
or go all-in on the fused kernel for the reuse wins on a self-hosted model.

## Where do I report a security vulnerability?

See [SECURITY.md](../SECURITY.md) for the disclosure process. Please do not open a public
issue for an undisclosed vulnerability.

## Where can I learn more?

- [Guided tutorial](fak/tutorial.md) — zero to first adjudicated call.
- [Policy in the kernel](explainers/policy-in-the-kernel.md) and [Addressable KV cache](explainers/addressable-kv-cache.md) — the two core ideas.
- [Benchmark authority](../BENCHMARK-AUTHORITY.md) — every number.
- [llms.txt](../llms.txt) — a machine-readable map for LLMs and answer engines.
