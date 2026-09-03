---
title: "fak FAQ — Limitations and honest scope"
description: "What fak does not guarantee, where its security and performance boundaries end, and which deployments still require external controls."
---

# Limitations and honest scope

Part of the [fak FAQ](../FAQ.md) — the essentials and every other theme are
indexed there.

`fak doctor` runs the same answer-shape witness AND cross-checks the real kernel admit verdict on the same bytes, then turns each finding into an operator recommendation. It calls `ctxmmu.ScreenBytes` — the exact predicate the kernel's write-time gate uses — so its `KernelAdmit` field reports the gate's actual decision (for example `SECRET_EXFIL`, `TRUST_VIOLATION`, or `OVERSIZE`), not a parallel re-implementation. Note that the kernel's repeat gate is a conservative binary seal (it quarantines only a 16-byte chunk repeated more than 50 times in a body of at least 512 bytes), so `doctor` is most useful for catching the softer loops the binary gate deliberately admits, where the graded answer-shape signal still warns. It exits 0 healthy, 1 when there is at least one finding, and 2 on a usage error, so it drops into CI as a gate over a captured answer — the `fak` analogue of `dos doctor`.

## Limitations and honest scope

The fences, stated plainly — `fak` is built to survive a skeptic reading the code, so the boundaries of what it does are part of the documentation.

## Is the in-kernel model engine ready to serve production traffic?

No, the in-kernel model engine is a bit-exact correctness reference, not a tuned production serving engine, and the README and claims ledger say so plainly. It is a from-scratch pure-Go forward pass whose load-bearing claim is oracle correctness versus a HuggingFace reference, not throughput, and it has no continuous batching, no paged attention, and no multi-tenant scheduler. Forward-pass parity is proven for the llama family (SmolLM2-135M, argmax-exact at every position, final-logit `max|Δ|` about 6e-5); non-llama family parity is open, real-GGUF end-to-end parity is open, and a Qwen3.6-27B multi-token greedy decode was refuted because it diverges from llama.cpp at token index 2. For real serving, run `fak serve` in front of vLLM, SGLang, or llama.cpp instead. The [What fak is not](../explainers/what-fak-is-not.md) explainer states this serving-engine boundary — and the 0/29-novel prior-art audit — plainly.

## Why do the cache-reuse savings only apply to self-hosted models?

Because the reuse win comes from owning the KV cache as a kernel object, and an app that merely calls a frontier API never holds that cache, so it gets the safety floor but none of the savings. The roughly 4x figure (versus a tuned warm-cache stack) and the 8.8x to 9.7x figure (modeled prefill elimination vs the naive floor over the real 643-task WebVoyager dataset, swept across worker counts) are reread-rate reductions over a cache `fak` controls. When you proxy to OpenAI or Anthropic, the provider owns prefix caching upstream, so `fak` is governing the wire rather than eliminating prefill. Front your existing API for the capability floor and result quarantine; go all-in on the fused kernel with a self-hosted model to also get the reuse wins. Every benchmark traces to a commit and artifact in the benchmark authority.

## What does the max|Δ|=0 bit-exactness proof actually guarantee, and what does it not?

It guarantees that when policy evicts a tool-result span from the KV cache, the model's next-token logits are byte-identical to a run that never saw that span, proven at `max|Δ|` of exactly zero with a non-vacuity control that confirms keeping the poison genuinely moves the distribution. That is a strong but narrow claim: it shows reuse and eviction are a faithful shortcut, not a numerical approximation. It does not prove the model is correct, does not prove the detector caught the poison, and for the quarantine-drives-KV-eviction bridge specifically it is witnessed on a synthetic model in `internal/kvmmu` and is not yet wired into the live `fak agent` HTTP loop. The deletion certificate that binds such an eviction to an audit journal is also self-attesting in v1 (integrity, not third-party independence) and proves removal only from the inference working set, not from weights, embeddings, backups, or replicas.

## What is Claude usage?

Claude usage is the share of an account's available model capacity consumed by requests.
It is not just a message count: plan, model, conversation length, files, tools, and other
compute-intensive features can change how much a request consumes. See the
[Claude usage guide](../claude-usage-limits.md) for the full answer.

## How do Claude usage limits work?

Claude usage limits are provider-enforced, time-windowed capacity controls. There is no one
fixed prompt allowance for every user because the effective limit depends on the account,
plan, model, workload, and Anthropic's current policy. The authenticated Claude usage screen
is the source of truth for remaining capacity and reset time.

## When does the Claude usage limit reset?

Use the reset timestamp shown by Claude for the exhausted limit. Session and longer-period
limits can reset on different schedules, so one window reopening does not prove that every
limit has reset. Fak records observed cooldown evidence but does not invent or change the
provider's reset time.

## Is the Claude usage limit the same as the context-window limit?

No. A usage limit caps account capacity over time; a context-window or length limit caps how
much a request or conversation can carry. Shorten context for a length error. For an exhausted
usage allowance, follow Claude's displayed reset or an available account capacity option.

## Does fak increase or bypass the Claude usage limit?

No. Fak cannot increase, reset, or bypass Anthropic's quota. It can adjudicate tool calls,
stop destructive retry loops, preserve recoverable long-run state, and route work from
observed capacity. Those controls can reduce avoidable waste, but Claude still enforces the
account's usage limit. See [Claude usage and usage limits](../claude-usage-limits.md).

## Does Claude Code have a separate usage limit from Claude on the web?

No, not when Claude Code uses the same Pro or Max subscription. Claude Code and Claude on
web, desktop, and mobile draw from shared usage, so activity on one surface can reduce what
remains on another. Direct API billing is a separate capacity path. See
[Claude Code usage limits](../claude-code-usage-limits.md).

## How do I check my Claude Code usage?

Run `/status` in Claude Code to confirm client account and plan information, then use Claude's
authenticated usage view for remaining capacity and reset timestamps. A fak gateway summary
shows requests and tool decisions, not Anthropic's remaining quota.

## Does Claude Code have a weekly usage limit?

Claude can apply longer weekly limits as well as shorter session windows. The limits that
apply depend on the account, plan, and current provider policy. Use the authenticated usage
view: a shorter window reopening does not prove that a weekly model-specific or aggregate
limit has reset.

## What is Claude extra usage?

Claude extra usage is optional paid capacity for an eligible account after included plan
usage is exhausted. It uses purchased usage credits and remains bounded by spending controls
and provider limits. It is not unlimited and does not reset the included allowance. See
[Claude extra usage and spending limits](../claude-extra-usage.md).

## Does Claude extra usage reset or bypass the usage limit?

No. Extra usage supplies a separately billed continuation path when enabled and funded; it
does not move the included plan's reset time or remove usage controls. Saying work continued
is not evidence that the original limit was bypassed.

## Why is Claude extra usage not working?

Verify account eligibility, that extra usage is enabled, that credits remain, that the
spending limit is not exhausted, and that an administrator does not control the setting.
