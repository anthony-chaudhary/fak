---
title: "More Fable 5 (and every model) out of your Claude seat — with fak guard"
description: "How one command — fak guard -- claude — stretches your existing Claude seat: the pure-fak, native-cache-excluded slice (context shedding + KV-prefix reuse), plus the one genuinely Fable-5-specific lever (the capped-Opus → Fable fallover). Every number is from a live fak-guarded fleet and labeled by provenance, with the double-count fence carried in full."
date: 2026-07-09
---

# More Fable 5 (and every model) out of your Claude seat — with `fak guard`

*This page reports **only fak's own authored slice** of the savings, on purpose:
Anthropic's native prompt-cache discount — the larger number `fak` preserves
byte-for-byte but does **not** author — is deliberately **excluded** here. Every
figure is labeled `WITNESSED` (fak measured it directly), `OBSERVED`
(provider-relayed), or `ESTIMATED`, and cites where it comes from. The number
authority is [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md); the
mechanism deep-dive and its honesty fences are
[`context-shedding.md`](../explainers/context-shedding.md).*

## The claim, stated honestly

`fak guard -- claude` is the **easiest way to get more work out of the Claude
seat you already pay for** — one command, on by default, nothing to configure,
nothing to clean up. It does that two ways, and it's worth being precise about
which is which:

1. **It keeps your session small and alive (model-agnostic).** Guard's gateway
   sheds the stale middle of your transcript each turn and keeps the cached head
   byte-identical, so the session runs longer under whatever usage window your
   seat has. This works the same for Opus, Sonnet, Haiku, **and Fable 5** — so
   "more Fable 5" from this lever is really "more of *whatever model you're
   running*."
2. **It reaches Fable 5's separate allocation when Opus caps out (Fable-specific).**
   When your default Opus 4.8 window hits a weekly / 5-hour / usage cap, guard's
   launcher fails the session over to **Fable 5's separate allocation bucket**
   automatically — genuinely "more Fable 5 usage" you could not otherwise reach.

Read "**for free**" precisely: it means *your existing subscription seat, one
command, default-on, no native prompt cache, and no manual model-switching* — it
does **not** mean the tokens are unbilled. On API billing Fable 5 is a paid tier
(and, under this repo's own apex cost model, [a pricey one](#the-honest-fences)).
"Free" is about effort and your existing seat, not about zero cost.

---

## The one command

```bash
fak guard -- claude    # your normal Claude Code, on your subscription, kernel-adjudicated
```

That's the whole setup. Specifically:

- **On your subscription by default** — no API key. Guard sources your logged-in
  Pro/Max OAuth token and authenticates upstream; the wrapped `claude` never sees
  a real credential. (Details and the end-to-end proof are in
  [`claude.md`](claude.md).)
- **Context shedding is on by default** at a 48,000-token resident budget
  (`gateway.DefaultCompactHistoryBudget`, `internal/gateway/gateway.go`; `0`
  disables). A short session never trips it; a long one starts shedding once it
  sprawls past the budget.
- **Nothing persistent is installed.** Guard writes *ephemeral, session-scoped*
  temp files (`--settings`, `--mcp-config`) and injects them into the child's
  argv only — your `~/.claude/settings.json`, your shell, and any other `claude`
  in another terminal are untouched (`cmd/fak/guard_precompact.go:246-253`,
  atomic temp+rename at `:264-283`). When Claude exits, there's nothing to undo.

If you want to see the lever move, run a long session and watch the per-turn
`fak-turn …` line and the exit summary; both print what compaction shed.

---

## Lever 1 — keep the session small and alive (the pure-fak slice)

A coding agent's transcript is append-only: every turn re-sends the whole
history. The provider discounts the byte-identical *front* via its prompt cache,
but the growing *middle* keeps missing cache and re-billing at full price. The
industry fix — summarize-and-recompact — rewrites the prompt body near the
context wall, which busts the cached prefix and forces a large cold re-prefill.

`fak guard`'s lever is different: past the budget it **drops whole stale middle
turns and splices the original bytes back together**, proves the protected
`cache_control` prefix is byte-for-byte identical, and forwards the body
unchanged on any doubt (`internal/gateway/messages.go:697` onward). The cached
head survives — so the provider's discount keeps paying — while the un-cacheable
middle is shed and a small restore handle is left where it was (recoverable later
via `fak_context_restore`). Full mechanism:
[`context-shedding.md`](../explainers/context-shedding.md).

### What we measured (live fak-guarded fleet, pure-fak slice only)

Source: `fak cachevalue report --json`, run against the live dogfood fleet.
Native prompt-cache dollars are **excluded** — these are fak's authored slice.

| Metric | Value | Provenance | Field |
|---|---|---|---|
| Realized KV-prefix reuse (latest multi-turn week; 0.69 all-time) | **69.7%** | WITNESSED | `track1_witnessed_kernel.latest_reuse_ratio` |
| Compaction fires across the fleet | 2,964 fires over 2,046 sessions | WITNESSED | `fleet_benefit.compaction_fired` / `exit_sessions` |
| Shed **per fire** (the honest unit — see fence below) | **≈37,000 tokens/fire** | WITNESSED | `compaction_shed_tokens ÷ compaction_fired` |
| fak's authored share of total token-equivalent value | **≈7.0%** | WITNESSED | `fleet_benefit.fak_share_pct` |
| fak-authored API cost attributed over the sampled window | ≈$541 (≈$63/day) | OBSERVED | `fleet_benefit.fak_api_cost_avoided_usd` |

Read these the honest way:

- **KV-prefix reuse (69.7% latest week, 0.69 all-time)** is a lossless, WITNESSED
  reuse of already-computed prefix — no output token changed. Two fences, both
  stated plainly: (i) the WITNESSED KV track is a **small sample** (11 sessions,
  6 multi-turn) — a direct measurement, not a fleet-wide average; and (ii) the
  all-time 0.69 currently sits **below the repo's 0.751 CI floor** (`fak nightrun
  score` exits non-zero on it right now), so cite it as a witnessed measurement
  *under active improvement*, **not** as a passing benchmark. Also note the
  vs-naive re-prefill multiple `1/(1-reuse)` is deliberately **excluded** — the
  honest single-session cache value is ~1.0× marginal over a tuned warm-KV server
  (the #1066 fence; `internal/cachevalueledger/ledger.go`).
- **Shed is cited per fire (~37k tokens), never as a sum.** The cumulative
  `compaction_shed_tokens` (≈110M fleet-wide) **re-counts the same aged middle
  every fire** as the client re-sends full history and fak re-trims from scratch.
  Turning that sum into "N context windows saved" or a "share of cache reads"
  is exactly the double-count this repo has an **on-record retraction** for (a
  since-corrected "75% share" claim; see
  [`context-shedding.md`](../explainers/context-shedding.md) and
  [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md)). Per fire it's "about a
  third of a turn's worth of context, trimmed each time compaction fires."
- **fak's authored slice is ~7% here, and that's the honest shape** — fleet-wide
  it runs roughly 0.3–16% of total tokens saved on the Claude route. fak's *bigger*
  job is keeping the provider's much larger discount **alive** as the session
  grows; that larger discount is the part we're excluding on this page by design.

### Dogfood witness: this very page

This guide was written **inside a `fak guard -- claude` session** (gateway trace
`guard`). At the time of writing, `fak_context_value` reported the session had
stayed alive across **54 context events over 80 turns** while its resident window
was held near the 48,000-token budget (`resident_tokens` 117,441, `peak` 153,273)
— against a raw transcript weight of **7,302,545 resident-token-turns**
(`total_resident_token_turns`, WITNESSED). Without the lever, that session would
long since have hit the context wall and forced a lossy native recompact. The
session extension isn't a projection here; it's the thing running this document.

---

## Lever 2 — reach Fable 5's separate allocation when Opus caps (Fable-specific)

This is the one lever that is *specifically* about getting **more Fable 5**.

Every seat starts on Opus 4.8 (`defaultLaunchModel = "claude-opus-4-8"`,
`cmd/fak/accounts_launch.go:65`). When that Opus start is refused **before the
session begins** by a usage/weekly/5-hour cap — matched by
`launchModelUsageLimitSignals` (`accounts_launch.go:511-522`) — guard's launcher
retries on the cheaper **Fable 5** alias (`defaultLaunchFallbackModel = "fable"`,
`:72`), which draws on a **separate allocation bucket**. The Opus 4.8 → Fable 5
chain is built at `:416-457` and fires via `shouldRetryLaunchWithFallback`
(`:471-481`):

```bash
# The launcher path (fak accounts launch / the dispatch fleet): capped Opus
# automatically fails the seat over to Fable 5's separate bucket.
fak accounts launch --fallback-model fable          # Opus weekly cap → Fable (accounts.go:143)
```

There's also a cost-pressure gate that lets you *choose* Fable 5 to satisfy a
high-pressure session cleanly, where an Opus route would be refused without an
explicit justification:

```bash
fak guard --session-pressure-gate high --model claude-fable-5 -- claude
# an explicit Fable route SATISFIES current high cost/context-pressure actions
# (cmd/fak/guard_session_pressure.go:86-136; docs/cli-reference.md:550-554)
```

**Honest scope of this lever:** (i) the automatic fallover lives in the
**launcher** (`fak accounts launch` / dispatch worker), adjacent to `fak guard`,
not inside plain `fak guard -- claude`; (ii) it fires only on a **fast startup
refusal**, not mid-session; (iii) it engages only when Opus is walled and
`--model` is left implicit. It genuinely gets you onto Fable 5 when Opus is
capped — it does not continuously "top up" Fable usage.

---

## The honest fences

Keep these in view — they are why the framing above is worded the way it is:

1. **The budget-stretch (Lever 1) is model-agnostic.** Context shedding and
   KV reuse behave identically for Opus, Sonnet, Haiku, and Fable 5. "More Fable
   5" from Lever 1 is really "more of whatever model you're on." Only Lever 2 is
   Fable-specific.
2. **"For free" = effort and your existing seat, not zero cost.** Fable 5 tokens
   are billed on API billing. Under this repo's own apex cost model Fable 5 is the
   **priciest** tier (`internal/modelroute/cost.go:62` prices it 2× frontier;
   `internal/fleetaccounts/apextier.go` calls it the restricted, explicit-only
   apex) — so "more Fable 5" can mean "more of the most expensive model." The
   subscription seat is where "free" is true: no extra spend beyond what you
   already pay.
3. **The repo disagrees with itself on Fable's price.** The session-audit
   subsystem prices Fable at $3/$15 (same as Sonnet — `internal/sessionaudit/
   sessionaudit.go:21`), while `modelroute`/`apextier` price it at $6/$30. Any
   concrete per-token savings math would be wrong under one of the two — this
   page reports token-equivalents and the report's own attributed dollars, and
   cites the field, rather than doing fresh Fable price math.
4. **Native prompt cache is excluded here on purpose.** The single largest
   "free savings" lever on the Claude route is Anthropic's own prompt-cache
   discount (and the 1h-TTL `--managed-cache` upgrade of it). fak *preserves* it
   byte-for-byte but does not author it, and it's passive on subscription seats —
   so this page leaves it out and reports only fak's authored slice, as asked.
5. **Shed per fire, never the sum.** Repeated because it's the easiest number to
   get wrong: never present cumulative shed as distinct savings or as a
   share-of-cache ratio.

---

## Prove it yourself

```bash
# 1. Your own fak-authored slice, native cache excluded (the numbers above):
fak cachevalue report --json        # track1_witnessed_kernel + fleet_benefit fields

# 2. Price your real Claude Code transcripts (~/.claude/projects/**/*.jsonl):
fak cachevalue report --dev-sessions

# 3. Watch the lever move on a live session (compaction on by default; 0 disables):
fak guard --compact-history-budget 48000 -- claude
#   → the per-turn `fak-turn …` line and the exit summary print what was shed.

# 4. Inspect the live session's context budget (what THIS page witnessed):
#   the fak_context_value MCP tool, exposed to a guarded child by default.
```

---

## Related

- [Claude Code / Anthropic API](claude.md) — the wiring, the subscription-by-default proof, and the four end-to-end checks.
- [Context shedding](../explainers/context-shedding.md) — the canonical mechanism and the full honesty framing (**primary source for Lever 1**).
- [Long-session value: 100 / 200 / 300 turns](../long-session-value.md) — the modeled savings table, every number cited and labeled SIMULATED/OBSERVED.
- [Cache-value roll-up](../cache-value-rollup.md) — the ongoing WITNESSED-vs-OBSERVED P&L (`fak cachevalue report`) and the #1066 marginal-over-warm-KV fence.
- [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md) — the number authority and the compaction double-count row.
