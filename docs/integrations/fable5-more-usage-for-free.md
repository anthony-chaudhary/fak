---
title: "More Fable 5 (and every model) out of your Claude seat — with fak manage"
description: "How one command — fak manage -- claude — stretches your existing Claude seat: the pure-fak, native-cache-excluded slice (context shedding + KV-prefix reuse), plus the one genuinely Fable-5-specific lever (the capped-Opus → Fable fallover). Every number is recent — the latest fleet week, plus this repo's own last-3-day Claude Code dev sessions priced end to end — labeled by provenance, with the double-count fence carried in full and the few deliberately-cited all-time baselines flagged as such."
date: 2026-07-09
---

# More Fable 5 (and every model) out of your Claude seat — with `fak manage`

*This page reports **only fak's own authored slice** of the savings, on purpose:
Anthropic's native prompt-cache discount — the larger number `fak` preserves
byte-for-byte but does **not** author — is deliberately **excluded** here. Every
figure is labeled `WITNESSED` (fak measured it directly), `OBSERVED`
(provider-relayed), or `ESTIMATED`, and cites where it comes from. The number
authority is [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md); the
mechanism deep-dive and its honesty fences are
[`context-shedding.md`](../explainers/context-shedding.md).*

## The claim, stated honestly

`fak manage -- claude` is the **easiest way to get more work out of the Claude
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
   When your default Opus window hits a weekly / 5-hour / usage cap, guard's
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
fak manage claude      # your normal Claude Code, on your subscription, kernel-adjudicated
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

`fak manage`'s lever is different: past the budget it **drops whole stale middle
turns and splices the original bytes back together**, proves the protected
`cache_control` prefix is byte-for-byte identical, and forwards the body
unchanged on any doubt (`internal/gateway/messages.go:697` onward). The cached
head survives — so the provider's discount keeps paying — while the un-cacheable
middle is shed and a small restore handle is left where it was (recoverable later
via `fak_context_restore`). Full mechanism:
[`context-shedding.md`](../explainers/context-shedding.md).

### What we measured (latest week, live fak-guarded fleet, pure-fak slice only)

Source: `fak cachevalue report --since 2026-07-06 --json` — the **latest week
only** (W28: 2026-07-06 → 2026-07-09, a 3.8-day window), snapshot 2026-07-09.
Native prompt-cache dollars are **excluded** — these are fak's authored slice.
(The KV-reuse row is the one exception to the `--since` floor: the sparse KV
ledger logged no multi-turn sample in W28, so that row carries `latest_reuse_ratio`
from the **unfloored** report — W27, the most recent week it actually measured.
Under the floored command that field reads `INSUFFICIENT`.)

| Metric | Value | Provenance | Field |
|---|---|---|---|
| Realized KV-prefix reuse (W27 — latest week the sparse KV ledger measured) | **69.7%** | WITNESSED | `track1_witnessed_kernel.latest_reuse_ratio` |
| Compaction fires (this week) | 2,347 fires over 898 sessions | WITNESSED | `fleet_benefit.compaction_fired` / `exit_sessions` |
| Shed **per fire** (the honest unit — see fence below) | **≈34,000 tokens/fire** | WITNESSED | `compaction_shed_tokens ÷ compaction_fired` |
| fak's authored share of total token-equivalent value | **≈20%** | WITNESSED | `fleet_benefit.fak_share_pct` |
| fak-authored API cost attributed (this week) | ≈$398 over 3.8 days (≈$105/day) | ESTIMATED (WITNESSED shed × rate card) | `fleet_benefit.fak_api_cost_avoided_usd` |

Read these the honest way:

- **KV-prefix reuse (69.7%)** is a lossless, WITNESSED reuse of already-computed
  prefix — no output token changed. It's the latest week the KV ledger actually
  measured (W27); the ledger is **sparse** and logged no fresh multi-turn sample
  in W28, so this is the most recent reading, not a same-week number. Two fences,
  both stated plainly: (i) the WITNESSED KV track is a **small sample** (11
  sessions, 6 multi-turn) — a direct measurement, not a fleet-wide average; and
  (ii) **both** the W27 reading (0.697) and the all-time 0.69 currently sit
  **below the repo's 0.751 CI floor** (`fak nightrun score` exits non-zero on it
  right now), so cite it as a witnessed measurement *under active improvement*,
  **not** as a passing benchmark. Also
  note the vs-naive re-prefill multiple `1/(1-reuse)` is deliberately **excluded**
  — the honest single-session cache value is ~1.0× marginal over a tuned warm-KV
  server (the #1066 fence; `internal/cachevalueledger/ledger.go`).
- **Shed is cited per fire (~34k tokens), never as a sum.** The week's cumulative
  `compaction_shed_tokens` (≈81M this week) **re-counts the same aged middle
  every fire** as the client re-sends full history and fak re-trims from scratch.
  Turning that sum into "N context windows saved" or a "share of cache reads"
  is exactly the double-count this repo has an **on-record retraction** for (a
  since-corrected "75% share" claim; see
  [`context-shedding.md`](../explainers/context-shedding.md) and
  [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md)). Per fire it's "about a
  third of a turn's worth of context, trimmed each time compaction fires."
- **fak's authored slice is ~20% this week, and that's the honest shape** — it
  runs higher in a recent, heavy-usage window (the all-time figure is ~8.5%;
  recent sessions are longer and shed more, so the fires-per-session density is
  up). fak's *bigger* job is keeping the provider's much larger discount **alive**
  as the session grows; that larger discount is the part we're excluding on this
  page by design.

### Dogfood witness: this very session

This page is being updated **inside a `fak manage -- claude` session** (gateway
trace `guard`). As these edits were made, `fak_context_value` reported the session
had stayed alive across **36 context events over 39 turns** — thirty-six
mid-session sheds — against a raw transcript weight of **3,948,011
resident-token-turns** (`total_resident_token_turns`, WITNESSED). Guard sheds the
stale middle toward the 48,000-token budget each turn; the resident window runs
*above* budget right after a heavy tool-output turn (here 121,832 against the
48,000 budget, `peak` 127,409) and is trimmed back as those turns age out. Without
the lever, a session of this transcript weight would long since have hit the
context wall and forced a lossy native recompact. The session extension isn't a
projection here; it's the thing running this document.

---

## Lever 2 — reach Fable 5's separate allocation when Opus caps (Fable-specific)

This is the one lever that is *specifically* about getting **more Fable 5**.

Every seat starts on Opus 5 (`defaultLaunchModel = "claude-opus-5"`,
`cmd/fak/accounts_launch.go`). When that Opus start is refused **before the
session begins** by a usage/weekly/5-hour cap — matched by
`launchModelUsageLimitSignals` — guard's launcher walks the fallback CHAIN
(`defaultLaunchFallbackModel = "claude-opus-4-8,claude-fable-5"`), which fires via
`shouldRetryLaunchWithFallback`. The two rungs answer two different walls:

- **`claude-opus-4-8`** — the previous Opus generation. It covers the *unknown-model*
  wall (a CLI build or seat that does not know the Opus 5 id), so that case degrades
  inside the Opus class instead of dropping a tier.
- **`claude-fable-5`** — the Fable rung, which draws on a **separate allocation
  bucket**. A weekly cap is bucket-scoped and therefore walls *both* Opus rungs, so a
  capped seat walks through to Fable — the lever this page is about.

Both rungs use the **versioned** id. The bare `fable` alias 400-crashes a headless
worker (the same chain feeds `claude -p --fallback-model` in the dispatch fleet), which
is why the chain never spells a rung as an alias:

```bash
# The launcher path (fak accounts launch / the dispatch fleet): capped Opus
# automatically fails the seat over to Fable 5's separate bucket.
fak accounts launch --fallback-model claude-fable-5   # Opus weekly cap → Fable, no alias
```

There's also a cost-pressure gate that lets you *choose* Fable 5 to satisfy a
high-pressure session cleanly, where an Opus route would be refused without an
explicit justification:

```bash
fak manage --session-pressure-gate high --model claude-fable-5 -- claude
# an explicit Fable route SATISFIES current high cost/context-pressure actions
# (cmd/fak/guard_session_pressure.go:86-136; docs/cli-reference.md:550-554)
```

**Honest scope of this lever:** (i) the automatic fallover lives in the
**launcher** (`fak accounts launch` / dispatch worker), adjacent to `fak manage`,
not inside plain `fak manage -- claude`; (ii) it fires only on a **fast startup
refusal**, not mid-session; (iii) it engages only when Opus is walled and
`--model` is left implicit. It genuinely gets you onto Fable 5 when Opus is
capped — it does not continuously "top up" Fable usage.

---

## Our own sessions, priced (the preserved-discount lens)

Lever 1 reports **only fak's authored slice** and excludes the provider's native
prompt-cache discount by design. This section shows that excluded number — the
discount fak keeps alive byte-for-byte — measured on **this repo's own Claude Code
dev sessions**, so you can see the whole picture on real work instead of a model.
It is a **different lens**, reported side by side with Lever 1 and never summed
with it.

Source: `fak cachevalue report --dev-sessions --dev-session-days 3` — a
**point-in-time snapshot at 2026-07-09T21:53Z**, last 3 days, this workspace's
namespace only. The lens discovers and prices real, un-proxied Claude Code
transcripts under `~/.claude*/projects/C--work-fak`. These are **live,
still-growing** files (the totals climbed measurably between runs minutes apart),
so read the figures as a recent snapshot, not a fixed ledger.

### The aggregate — 15 priced Opus-tier sessions from the last 3 days

| Metric | Value | Provenance | Field |
|---|---|---|---|
| Sessions discovered (last 3 days, this repo) | 60 discovered — 15 priced (Opus-tier), 45 held out (36 synthetic, 9 other unpriced) | WITNESSED | `dev_session_benefit.sessions` / `priced_sessions` |
| Provider cache-read tokens preserved | 62.6M | OBSERVED | `cache_read_tokens` |
| API-equivalent cost **without** the preserved cache | $1,181.37 | OBSERVED | `observed_counterfactual_usd` |
| API-equivalent cost **with** it | $375.54 | OBSERVED | `observed_actual_spend_usd` |
| API cost avoided by the preserved discount | **$805.83 (68.2% reduction)** | OBSERVED | `observed_api_cost_avoided_usd` / `_reduction_pct` |

Read it the honest way: the **$806 is the *provider's* discount, not fak's** — fak
neither authors nor bills it. What fak authors is its **survival**: Lever 1 keeps
the session small and the cached prefix byte-identical so this discount keeps
paying as the transcript grows past the point a lossy recompact would otherwise
bust it. The pricing is sessionaudit's Opus base rate ($15/$75 per MTok, cache
reads at $1.50) and these sessions are Opus-tier, so the repo's Fable-price
disagreement (fence 3) does not touch these numbers. Because a guarded session
also leaves a transcript here, this lens **may overlap** the Lever-1 fleet
aggregate — the two are never added.

### Three recent sessions from this repo

The three most-recently-active sessions in the live account, each priced with
`fak cachevalue status --session <transcript> --json` (same 2026-07-09 snapshot):

| Session (this repo) | Assistant turns | Provider cache-hit | Cache-read tokens | API-equiv cost (Opus) |
|---|---|---|---|---|
| `8a9f820c…` | 49 | 89.3% | 3.36M | $18.64 |
| `a8aadc0c…` | 64 | 84.0% | 5.27M | $31.62 |
| `2aaab7c9…` | 39 | 83.6% | 3.54M | $26.06 |

The cache-hit fraction *is* the lever's payoff: on the 64-turn, 6.3M-token
session, 84.0% of the context tokens billed came back as cache reads (Opus
$1.50/MTok) rather than fresh input (Opus $15/MTok) — the byte-identical prefix
fak protects on every turn. The cost column is the **API-equivalent** price of
those tokens; on a subscription seat you paid your flat seat, not this — it is
what the same work would meter at, and on a pricey tier (Fable 5 more so than
Opus) that gap is the whole point of keeping the cache alive.

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
6. **The dev-session lens is the *preserved* discount, not the authored slice.**
   "Our own sessions, priced" reports the provider prompt-cache value fak keeps
   alive (OBSERVED, Opus base pricing, may overlap the fleet aggregate) — the
   opposite selection from Lever 1's authored slice. fak authors the cache's
   survival, not the discount, and the two lenses are never summed.

---

## Prove it yourself

```bash
# 1. Your own fak-authored slice, latest week, native cache excluded (the numbers above):
fak cachevalue report --since 2026-07-06 --json   # track1 + fleet_benefit, latest-week window

# 2. Price your recent Claude Code transcripts — live, still-growing files:
fak cachevalue report --dev-sessions --dev-session-days 3   # the recent "our own sessions" aggregate
fak cachevalue status --session <transcript.jsonl> --json   # one session, priced (the per-session table)

# 3. Watch the lever move on a live session (compaction on by default; 0 disables):
fak manage --compact-history-budget 48000 -- claude
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

> **These numbers are gated.** Every headline figure on this page is bound to a
> frozen `fak cachevalue report` snapshot and its arithmetic invariants by
> `tools/docnumbers/fable5-more-usage-for-free.json`, checked hermetically in
> `make cachedoc-numbers-lint`. When the window has moved on, refresh them with
> the `/refresh-cachedoc-numbers` skill (or `python3
> tools/cachedoc_numbers_audit.py --refresh`) rather than editing the numbers by
> hand — see `tools/docnumbers/README.md`.
