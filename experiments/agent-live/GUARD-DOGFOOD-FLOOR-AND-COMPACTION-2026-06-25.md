---
title: "fak guard dogfood — the floor-friction fix and the compaction lever, proven and fenced"
description: "A dogfood pass over `fak guard -- claude` for our own typical work: the default floor was DEFAULT_DENYing the host harness's own orchestration tools (the dominant friction a historical-session replay flagged), now fixed; and the cache-prefix-preserving compaction lever ('safely bring context back as it overflows') re-verified on the current binary. Includes the honest prove/refute ledger: what is measured vs the one credentialed step still open."
date: 2026-06-25
---

# fak guard dogfood — floor friction + the compaction lever

A dogfood pass aimed at one question: does `fak guard -- claude` deliver real,
effective value for *our* typical work — including safely bringing context back
to a reasonable amount as it overflows — or does it get in the way? The answer
is **both**: it adjudicates real danger and sheds context safely, but it was
also actively crippling the agent until the fix below. Every number here is from
a witness on disk; the one step that is *not* measured is fenced explicitly.

## Finding 1 — the floor was denying the harness's own tools (fixed)

`fak guard` installs a default capability floor (`cmd/fak/guard-default-policy.json`).
It allowed `Task`/`TodoWrite` but **DEFAULT_DENYd the host harness's actual
orchestration surface**: `TaskCreate`/`TaskUpdate`/`TaskOutput`/`TaskList`/
`TaskGet`/`TaskStop`, `Agent`, `SendMessage`, `EnterPlanMode`, `Monitor`,
`ScheduleWakeup`, `AskUserQuestion`, the read-only MCP-resource readers, and
**`ToolSearch`**. So `fak guard -- claude` silently bricked the agent's own task
system, subagent spawning, plan mode, and — via `ToolSearch` — the ability to
load *any* deferred tool (WebFetch / WebSearch / MCP all become uncallable on a
harness that defers tool schemas).

This was not a hunch — it is the dominant friction the historical-session replay
surfaced on its own (`experiments/agent-live/CLAUDE-HISTORICAL-GUARD-AUDIT-2026-06-25.md`):
**42 `DEFAULT_DENY` on `Task*` alone** across 16 audited Claude Code sessions, and
the remediation engine named `align_policy_with_real_tool_shapes` +
`reduce_permission_interruptions_or_scope_policy` for **every** audited session.
The live guard journal independently showed `ToolSearch -> DENY / DEFAULT_DENY`.

**Fix:** `a0796fd` (`fix(guard): admit the host harness orchestration + ToolSearch
in the default floor`). Admitting these does **not** widen the danger floor: a
subagent the floor lets the agent *spawn* makes its real tool calls back through
this same gateway, so every effect is re-adjudicated downstream. Proven against
the shipped embedded floor with `fak preflight`:

| tool | before | after |
|---|---|---|
| `ToolSearch`, `TaskCreate`, `TaskUpdate`, `TaskOutput`, `Agent`, `SendMessage`, `EnterPlanMode`, `Monitor`, `ReadMcpResourceTool` | DEFAULT_DENY | **ALLOW** |
| `Bash {command:"rm -rf …"}` | POLICY_BLOCK | POLICY_BLOCK (unchanged) |
| unlisted `RemoteTrigger` | DEFAULT_DENY | DEFAULT_DENY (still fails closed) |
| `Write` into `.ssh/` | SELF_MODIFY | SELF_MODIFY (unchanged) |

`dos commit-audit a0796fd` → `OK` (diff-witnessed). Test:
`go test ./cmd/fak -run Guard` (WSL) → ok.

## Finding 2 — the compaction lever is real on the current binary

"Safely bring context back as it overflows" is `agent.CompactAnthropicHistory`,
default-on at a 48k resident-token budget (`DefaultCompactHistoryBudget`). It
drops OLD whole turns from the *outbound* Anthropic body by SPLICING on the
original bytes, so the `cache_control` prefix stays byte-identical and the
provider's prompt-cache hit survives. Re-verified on the binary built this
session (`compact-history-dogfood-current-binary-2026-06-25.json`):

| metric | value |
|---|---|
| inbound est. tokens | 142,516 |
| forwarded (compaction on) | 10,644 |
| **shed** | **131,872 (92.5%)** |
| cache prefix (10,660 bytes) byte-identical off-vs-on | **yes** (sha256 `f3e5aa83…`) |
| spliced body re-decodes valid | yes |

The *safe* property — it never re-marshals the cached prefix, and returns the
input unchanged on any ambiguity — is what the byte-identity check proves. It can
fail to *help*; it structurally cannot break a turn or bust the cache.

## The prove / refute ledger

| claim | status | evidence |
|---|---|---|
| The dos-style floor refuses a real dangerous call on a live Claude Code session and useful work continues | **PROVEN** | `claude-code-fak-guard-live-pilot-2026-06-25.json` (`rm -rf`→POLICY_BLOCK, same-session continuation) |
| The floor now fits the harness's real tool shapes (no longer self-denies) | **PROVEN** | `a0796fd` + preflight table above |
| Compaction sheds ~92% at 100k+ while keeping the cache prefix byte-identical | **PROVEN (mock upstream)** | witness above; the upstream is a mock that records the forwarded bytes |
| Anthropic actually *reuses* a byte-stable prefix on real Claude Code traffic | **PROVEN (premise)** | `vcache-claude-prefix-probe-2026-06-25.jsonl`: a cold turn writes `cache_creation=59,400`, three sibling turns each read **`cache_read=43,995`** — real `claude-opus-4-8` traffic, real cost |
| Dropping the MIDDLE and shifting the recent breakpoint *still* cascades back to the head prefix (the compaction-specific cascade) | **OPEN** | not measured; the vcache probe reuses an identical head, it does not drop a middle — see fence |

The honest read: the dos-style guard is **proven** to deliver value on both axes
(it adjudicates real danger; it sheds context byte-safely), and the cascade's
*premise* — that the provider discounts a byte-identical prefix at all — is now
**measured on real traffic**, not assumed. What remains a fence is narrow and
specific.

## Honest fences (still open)

- **The compaction cascade is one credentialed session from a number.** The
  instrument already emits both halves — `fak_gateway_compaction_shed_tokens_total`
  (WITNESSED) next to `fak_gateway_compaction_cache_read_tokens_total` (OBSERVED).
  Settling it means running a long real `fak guard -- claude` session that fires
  compaction (>48k resident, multi-breakpoint) and scraping the two series. That
  is a deliberate billing event on real Anthropic tokens, so it is surfaced as an
  operator decision rather than run autonomously. Tracked: epic #745.
- **Mock upstream for the byte numbers.** The 92.5%-shed witness proves fak ships
  a smaller body with a byte-identical prefix; it does not prove the provider
  billed less on that turn (that is the cascade above).
- **The large-result vector is unaddressed here.** The same replay flagged
  `cap_or_summarize_large_outputs` on 7 sessions (max single tool result 55,614
  chars). Compaction sheds OLD turns; it does not cap a single oversized result.
  That is a separate "bring context back" lever, not yet built.

## Re-run

```bash
go build ./cmd/fak
# floor before/after (shipped embedded floor):
./fak guard --dump-policy > /tmp/floor.json
./fak preflight --policy /tmp/floor.json --tool ToolSearch --args '{}'   # -> ALLOW
./fak preflight --policy /tmp/floor.json --tool Bash --args '{"command":"rm -rf /x"}'  # -> DENY POLICY_BLOCK
# compaction lever:
python tools/compact_history_dogfood.py --fak ./fak --out /tmp/compact.json --turns 900 --budget 8000
go test ./cmd/fak -run Guard -count=1   # (WSL on a native-Windows host)
```
