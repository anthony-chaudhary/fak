# Tool-call control spine — 2026-08-17

## Value frame

- **Centrality:** Core. FAK handles every tool call; avoiding calls before execution is on its primary performance seam.
- **For:** long-running coding and research agents whose prompt is already large.
- **Problem:** a low-value tool call pays not only tool latency, but another model continuation over the accumulated context.
- **Today:** prompts can ask the model to be frugal, while existing runtime telemetry mostly explains calls after they happen.
- **Better because:** a deterministic prefilter can reuse exact fresh observations, coalesce independent reads, and defer weak speculative reads before execution.
- **Witness:** `go run ./cmd/toolcallcontroldemo -selfcheck -pretty` emits per-call decisions and a same-trace control/prefilter ablation.

All four problem checks apply: this reduces managed context replay (P1), reports net-true call and context-token effects without calling a proxy measured cost (P2), admits state changes and avoids suppressing mutations (P3), and emits machine-readable decisions plus arm metrics (P4).

## Minimal working spine

The model-side instruction is deliberately short: before calling, name the missing evidence and the decision it can change; reuse fresh results; batch independent reads; stop when evidence is sufficient. At 64k tokens and above it explicitly identifies the long-context replay penalty.

The prefilter then checks claims the kernel can witness:

1. **Exact fresh repeat:** `(tool, normalized args, state epoch)` matches a prior observation → `reuse`.
2. **Batchable read:** read-only proposals share a batch key → allow one deterministic leader and merge followers into it.
3. **Speculative read:** no evidence gap, no changed decision, and low stated information gain → `defer`.
4. **Safety boundary:** mutations are not deferred on weak model rationale; novel and required calls remain allowed.

This first spine intentionally avoids semantic similarity and learned gating. False-positive suppression is more expensive than a redundant call, so broader gates need trace-backed calibration first.

## Ablation contract

Every arm runs over the same independently labeled proposals. Reports keep these quantities separate:

- calls executed;
- unneeded calls avoided;
- needed calls suppressed (the false-positive guardrail);
- context tokens whose replay was avoided;
- `attention_proxy_avoided = Σ context_tokens²`, explicitly a quadratic exposure proxy—not provider FLOPs, latency, dollars, or measured savings.

The captured 128k-context demo executes 5 calls in control versus 2 continuations in the prefilter arm, avoids 2 independently labeled unnecessary calls, and suppresses 0 needed calls. Batching merges one needed read but does not count it as suppression.

## Recent GitHub field scan (2026-08-03 through 2026-08-17)

GitHub search was run for recently created/popular agent tool-calling and context-engineering repositories, plus recent commits/issues in established agent projects. Search results were noisy; popularity alone did not identify a shipped, general unnecessary-call gate. The strongest transferable work was in actively developed OpenAI Codex:

- [`openai/codex#38978`](https://github.com/openai/codex/pull/38978), merged 2026-08-17: caps skill-catalog context at a configurable budget (default remains 2% of window, hard cap 10k). **Borrow:** bound model-visible catalogs instead of assuming retrieval is free. **FAK disposition:** follow-on, not this gate.
- [`openai/codex#39008`](https://github.com/openai/codex/pull/39008), merged 2026-08-17: task-context fusion for short continuation requests, bounded to two prior substantive requests and recent relevant skills, evaluated as a shadow selector. **Borrow:** use bounded task state rather than full transcript, and shadow new selectors before enforcement. **FAK disposition:** default methodology for learned/semantic call gating.
- [`openai/codex#38993`](https://github.com/openai/codex/pull/38993), merged 2026-08-17: recent-skill and character-routing variants evaluated in shadow mode with deterministic ranking and truncation metadata. **Borrow:** compare multiple selectors on identical traffic and preserve truncation metadata. **FAK disposition:** follow-on ablation arm design.
- [`openai/codex#38980`](https://github.com/openai/codex/pull/38980), merged 2026-08-17: bounds parent-compaction context and fails closed when oversized. **Borrow:** context admission needs explicit byte/token limits and an observable oversized outcome. **FAK disposition:** follow-on for evidence/result reuse payloads.
- [`openai/codex#38921`](https://github.com/openai/codex/pull/38921), merged 2026-08-17: groups successful command activity while preserving the full transcript and flushes at failures/boundaries. **Borrow:** compact observability without deleting drill-down evidence. **FAK disposition:** report/UI follow-on.

New repositories with the largest stars in the exact two-week creation window were mostly application wrappers (for example `ysr666/dsh-vision-router`, 615 stars when observed) rather than evidence about unnecessary-call prevention. That negative result matters: this spine borrows tested mechanisms from active upstream changes, not popularity theater.

## Reproduction

```bash
go test ./internal/toolcallcontrol ./cmd/toolcallcontroldemo
go run ./cmd/toolcallcontroldemo -selfcheck -pretty
```

The demo is offline, deterministic, and performs no real tool execution.
