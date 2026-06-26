# fak 100k+ session value — end-to-end proof (2026-06-25)

This packet folds today's evidence that **fak delivers concrete, measurable value at 100k+
session length**, via the two paths a user actually has: the flagship `fak guard -- claude` /
`fak serve` Anthropic passthrough, and direct MCP adjudication + the guard floor. It is a
status artifact; quote [`CLAIMS.md`](../../CLAIMS.md) for shipped scope. Tracked by epic
[#745](https://github.com/anthony-chaudhary/fak/issues/745) (children
[#746](https://github.com/anthony-chaudhary/fak/issues/746) MCP,
[#747](https://github.com/anthony-chaudhary/fak/issues/747) guard).

## The problem, and why it is hard

A coding agent in a long session has an append-only transcript, so every turn re-sends the
whole history. The provider prompt-cache discounts most of it — **but only while the cached
prefix stays byte-identical.** Naive compaction (summarize old turns, re-serialize the body)
*breaks* the cache prefix, so a long session ends up costing **more**, not less. The hard
part is shedding history tokens *without* breaking the cache.

## What fak does (one flag)

```
fak guard --compact-history-budget 8000 -- claude      # or: fak serve --compact-history-budget 8000 ...
```

`agent.CompactAnthropicHistory` (consumer
`internal/gateway/messages.go:maybeCompactAnthropicRaw`) drops OLD whole turns to a
resident-token budget while keeping the protected `cache_control` prefix
**byte-identical** — it SPLICES on the original bytes (a memcpy of the protected prefix
through the stable breakpoint, never a re-marshal that would reorder JSON keys and break
the cache). Fail-safe identity on any ambiguity, including a candidate drop that itself
contains `cache_control`, so it never breaks a turn or silently bursts provider-warm
history. Request-side transform only — the kernel still adjudicates the FULL decoded
history, so the trust boundary is unchanged.

## Verdict

| Surface | Status | Evidence |
|---|---|---|
| Compaction sheds history at 100k+ while keeping the cache hit | **PASS** | `compact-100k-session-dogfood-2026-06-25.json` — 142,516-tok inbound → 6,597 forwarded (**95.4% shed**), OFF/ON cache-prefix sha256 equal. A larger run: 236,075 → 10,669 (**95.5% shed**), prefix identical. |
| Cache-prefix byte-identity (unit) | **PASS** | `go test ./internal/agent ./internal/gateway -run Compact` — `TestCompactPreservesCachePrefix` + 8 fail-safe `TestCompactIdentityCases` + tool-pair safety + gateway gating. |
| Floor + lever on ONE `fak serve` wrap | **PASS** | `compact-guard-combined-dogfood-2026-06-25.json` — `git_push`→DENY(POLICY_BLOCK), `git_status`→ALLOW, AND 93,220-tok inbound → 5,275 forwarded (94.3% shed), prefix identical. |
| MCP stdio adjudication (the `.mcp.json` path) | **PASS** | `python examples/mcp/verify.py` → summary PASS: handshake, tool discovery, `git_push` DENY, `git_status` ALLOW. Zero deps. |
| `fak guard` default floor + audit | **PASS** | `go test ./cmd/fak -run Guard` → ok. |

## How the value scales

Token shed grows with session length (the bigger the transcript, the more old turns drop),
and the cache prefix is preserved at every size. The lever is therefore *most* valuable
exactly where long sessions hurt most — past 100k tokens.

| Inbound session | Budget | Forwarded | Shed | Cache prefix |
|---|---|---|---|---|
| 93,220 tok | 4000 | 5,275 | 94.3% | byte-identical |
| 142,516 tok | 4000 | 6,597 | 95.4% | byte-identical |
| 236,075 tok | 8000 | 10,669 | 95.5% | byte-identical |

## Re-run commands

```bash
# Compaction-only dogfood (real fak serve Anthropic passthrough, mock upstream records bytes)
python tools/compact_history_dogfood.py --fak ./fak --out /tmp/compact.json --turns 900 --budget 4000

# Combined: capability floor + compaction on ONE fak serve wrap
python tools/compact_guard_combined_dogfood.py --fak ./fak --out /tmp/combined.json --turns 900 --budget 4000

# MCP stdio adjudication proof (zero deps)
python examples/mcp/verify.py

# Unit cache-safety + guard floor
go test ./internal/agent ./internal/gateway -run Compact -count=1
go test ./cmd/fak -run Guard -count=1
```

## Honest fences

- The dogfood upstream is a **mock** that records the bytes fak forwards; it proves fak sends
  a compacted body with a byte-identical prefix. The *provider-side* cache hit (real
  `cache_read_input_tokens` from Anthropic) is the credentialed-host follow-on (epic #745).
- Token counts use the **bytes/4 proxy** (`EstimateAnthropicTokens`, the budget unit); a real
  BPE tokenizer shifts absolutes, not the regime.
- Compaction drops old-turn DETAIL (a stub names the count) — the right trade for a coding
  agent whose recent window + durable system prompt carry the live working state; it is not
  loss-free recall.
- The compaction lever is **Anthropic passthrough only** (the one wire forwarded
  byte-for-byte). An OpenAI/Codex long session gets the floor + audit but its body is rebuilt
  downstream, so this specific cache-preserving rewrite does not apply there (a separate
  lever).
