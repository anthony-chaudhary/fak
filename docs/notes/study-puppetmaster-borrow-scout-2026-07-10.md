---
title: "Borrow scout: Puppetmaster → fak (2026-07-10)"
description: "Study of Puppetmaster (professorpalmer/Puppetmaster; MIT; pinned SHA 9597e5beb54d95d3240a15a8a10421c905de3e8b, v1.18.0 'AGNT-style all-1h Anthropic prompt cache for agentic BYOK', cloned read-only to scratch) for techniques worth porting. 122-file Python BYOK proxy/harness read by fan-out; ~11 candidates witnessed, 1 borrow filed (#3940), the rest PRESENT (fak equal-or-more-rigorous on IFC-taint egress, leaseref fencing, elide/savings). One open measurement lead (all-1h scope vs #1850) deliberately documented, not filed. CORRECTION (2026-07-10, adversarial re-verify): an earlier draft of this note called Puppetmaster's header-free all-1h cache a latent bug fak was 'strictly ahead' on — that was WRONG. 1h prompt caching is GA on the direct Anthropic API as of 2026; the extended-cache-ttl-2025-04-11 beta was retired to a backwards-compat no-op, so Puppetmaster's inline ttl:1h with no beta header is correct current usage. No upstream bug; nothing filed against Puppetmaster."
---

# Borrow scout: Puppetmaster → fak (2026-07-10)

Study of **Puppetmaster** (`professorpalmer/Puppetmaster`) — a Python **BYOK provider proxy +
agent harness** whose headline release `v1.18.0` is literally *"AGNT-style all-1h Anthropic
prompt cache for agentic BYOK"* — **MIT** (© 2026 Cary), pinned SHA
`9597e5beb54d95d3240a15a8a10421c905de3e8b`, cloned read-only to scratch — for techniques worth
porting into fak. Every borrow is **INSPIRE**, not INTEGRATE: Puppetmaster is Python, fak is Go,
so any port is a clean-room reimplementation — no source is copied. MIT permits even a copy with
attribution; the attribution here is provenance, and the anchors below are the exact
`path:line @ 9597e5b` each candidate was read at.

Read by fan-out (122 Python files under `puppetmaster/`): the **provider proxy + Anthropic cache
placement** (`provider_proxy.py`, `adapters/agentic.py`, `adapters/_context_budget.py`), the
**memory store + retrieval** (`store.py`, `sqlite_store.py`, `mmr.py`), **leasing/liveness**
(`platform_lock.py`, `liveness.py`, `win_console.py`), **egress security** (`openai_security.py`,
`redaction.py`), **tool-output offload + savings** (`tool_offload.py`, `savings.py`), **skill
injection** (`skill_injection.py`), **cancellation/preflight** (`cancellation.py`, `preflight.py`),
and the **stitcher/conflicts** worker plane (`stitcher.py`, `conflicts.py`).

Scope note: Puppetmaster and fak are **the same genre** — a local kernel/proxy that sits between an
agent and a provider, does BYOK cache economics, leases, egress control, and memory. That makes the
witness unusually direct: nearly every Puppetmaster mechanism has a named fak counterpart, and the
question each row answers is not "does it fit" but "is fak's already at least as rigorous." The
honest headline: **it is, almost everywhere** — the one filed borrow (MMR recall diversity) is the
one axis a same-genre peer does something fak's retrieval path does not.

> **Correction (2026-07-10).** An earlier draft of this note claimed the release-naming candidate
> (all-1h cache, row 1) was a place fak is *strictly ahead* — that Puppetmaster ships a latent bug
> by omitting the `extended-cache-ttl-2025-04-11` beta header. **That was wrong**, and an
> adversarial re-verify (4 code readers + Anthropic-doc check + 3 refuters) caught it. As of 2026,
> 1h prompt caching is **GA** on the direct Anthropic API: you set `ttl:"1h"` inline in
> `cache_control` with **no** beta header; the 2025 beta header was retired to a backwards-compat
> no-op ([Claude Platform docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)).
> Puppetmaster's header-free inline `ttl:"1h"` is therefore **correct current usage**, not a bug —
> it does exactly what the official guidance recommends. Row 1 is corrected below; **nothing was
> filed against Puppetmaster** (verifying-before-filing is the whole point of the pass).

## Scorecard (candidate techniques, one technique each)

| # | Technique (Puppetmaster anchor @ 9597e5b) | fak witness | Verdict |
|---|---|---|---|
| 1 | **all-1h Anthropic prompt cache**: `cache_control` on system + last tool + last two history messages, all at `ttl:"1h"` (default; `PUPPETMASTER_ANTHROPIC_CACHE_TTL=5m` forces no-ttl), stamped **inline in the body** with **no** beta header — the v1.18.0 headline. Actual anchors (corrected): `providers.py:76-78` (the `{"type":"ephemeral","ttl":"1h"}` marker), `:161-168` (direct-API descriptor `api.anthropic.com/v1`, `x-api-key`, `anthropic-version:2023-06-01`), `:500-544`/`:713`/`:948` (apply + POST) | `internal/agent/anthropic_cachebp.go` `UpgradeAnthropicStableCacheTTL1h` also sets `ttl:1h` inline and additionally unions the (now-retired, no-op) `extended-cache-ttl-2025-04-11` beta (`internal/gateway/messages_tooldefer.go:60`), self-checked | **PRESENT — both correct; no borrow, no bug.** 1h caching is **GA** on the direct Anthropic API as of 2026: inline `ttl:"1h"` with no beta header is the recommended form; the 2025 beta header is a backwards-compat no-op ([docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)). Puppetmaster's header-free stamping is correct current usage. fak's beta-union is **safe either way** — a no-op if 1h is fully GA on every served path, still load-bearing if any account-revision/model guard fronts hasn't GA'd it — so its fail-open, revision-gated union (`messages_tooldefer.go:44-46`) is correct, **not a defect**. *(Earlier draft wrongly called this "fak strictly ahead / Puppetmaster ships a bug"; a follow-up draft over-corrected the other way and called fak's union "redundant dead weight / candidate cleanup" — both retracted. See the correction box above and the Outcome note.)* |
| 2 | **all-1h beats hybrid** economic claim: caching the *moving history* at 1h avoids the per-turn cache-WRITE the 5m history breakpoint pays every turn (AGNT measurement, release notes + `_context_budget.py`) | fak deliberately upgrades **only the stable head** to 1h and leaves the volatile message tail uncached (`#1850`; `anthropic_cachebp.go` "message-tail breakpoints are ignored"). fak has `internal/cachevaluereport` + cacheprice to measure its own workload | **OPEN LEAD — documented, not filed.** Whether lifting the client's recent-turn breakpoints to 1h is cheaper for fak is workload-dependent (1h writes cost 2× base vs 5m's 1.25×; wins only when turn cadence > 5m). Not witnessed against fak's cost model this pass; a future scout should run `cachevaluereport` before re-opening the #1850 scope. |
| 3 | **MMR memory-retrieval diversity**: `finalize_memory_retrieval` reranks the top-k memory pool with greedy MMR — `λ·relevance − (1−λ)·max_similarity_to_selected`, Jaccard word-set similarity over the memory `statement`, λ=0.7, env-gated, over a `3×limit` pool (`mmr.py:60,106`; wired at `store.py:918`, `sqlite_store.py:938`) | `internal/recall/journal_index.go:224` `Recall()` ranks provenance → recency → FTS-overlap → index-order and takes top-k with **no redundancy/diversity term**; `internal/recall/parity.go:88` notes dedup auto-wiring into the top-k path is **deferred** | **ABSENT — FILED [#3940](https://github.com/anthony-chaudhary/fak/issues/3940).** k near-duplicate rows all get injected, spending the very budget recall conserves. Borrow: a redundancy suppressor applied *within* the provenance ordering (never demoting a witnessed row below a claim), reusing the overlap score `score()` already computes. |
| 4 | **tool-output offload with a measured-savings gate** (`tool_offload.py`, `savings.py`) | `internal/savingsvector/savingsvector.go` + `internal/agent/message_elide.go` / `anthropic_elide.go` — offload/elide gated on measured replacement size, savings accounted | **PRESENT — no borrow.** fak's elide plane is a superset (CAS page-back, ctxmmu opaque pointers). |
| 5 | **budgeted skill injection** (`skill_injection.py`) | `cmd/fak/skill_footprint.go` (token footprint) + `internal/ctxmmu/skillbody.go` + `internal/policy/skill.go` + `internal/capindex/skill_resolver.go` | **PRESENT — no borrow.** fak injects skill bodies under a footprint budget with effectiveness tracking (`skill_effectiveness.go`). |
| 6 | **credential-egress / SSRF floor** (`openai_security.py`, `redaction.py`) | IFC taint exfil floor: `internal/gateway/proxy_exfil_floor_test.go` + `internal/egressfloor` + `internal/ifc` `SinkGate`/`StampGate` — an untrusted tool-result raises the trace taint high-water mark, which the sink-gate reads to DENY the proposed egress call | **PRESENT — fak ahead.** fak gates egress on *information-flow taint*, not a static host allowlist; the floor is armed on the auto proxy topology (`#77`). No borrow. |
| 7 | **fencing/lease + liveness** (`platform_lock.py`, `liveness.py`, `win_console.py`) | `internal/leaseref` `Record.Generation` monotonic **fencing token** (`#906 §3.3 / #1182`), bumped on transition / never on same-holder renew, `fence.go` CAS admission, `RenewedAt` heartbeat, `SessionID` binding + cascade-on-dead-session; Windows PID liveness across `internal/{accounts,dispatchaudit,safecommit}/alive_windows.go` + `internal/procguard/collect_windows_native.go` | **PRESENT — fak ahead.** fak's lease carries a true fencing token + heartbeat + session cascade. No borrow. |
| 8 | **live preflight probe** before committing a route (`preflight.py`) | `internal/accounts/probe.go` (probe request, OAuth beta asserted in `probe_test.go`) | **PRESENT — no borrow.** |
| 9 | **mid-stream cancellation** on client disconnect (`cancellation.py`) | fak `cancel_disconnect_test.go` — upstream cancelled when the inbound client disconnects | **PRESENT — no borrow.** |
| 10 | **structured output via a forced tool call** (`evaluators.py`, `gates.py`) | fak grammar/guided-decode + `internal/signals/schema.go` `ValidateAgainstSchema` + `enumContains` | **PRESENT — no borrow.** |
| 11 | **multi-worker finding stitch / write-conflict detection** (`stitcher.py`, `conflicts.py`) | dos lease/arbiter kernel (`dos_arbitrate`, lane leases with lock modes) prevents the collision upstream; fak does not stitch multiple workers onto one task's output | **PRESENT-in-spirit / LOW FIT — no ticket.** fak's model is disjoint-lane isolation, not post-hoc stitch. |

**Outcome: 1 borrow filed of ~11 candidates across a same-genre 122-file peer.** Witnessing
prevented ~9 duplicate/N-A tickets against machinery fak already ships — usually more rigorous
(the IFC exfil floor vs a static allowlist, the `leaseref` fencing token + heartbeat vs a bare
lock, the elide/savings plane, budgeted skill injection). The single filed borrow (MMR diversity
in recall selection) is the one place a same-genre peer does something fak's retrieval path does
not. One open lead (row 2, all-1h moving-history scope) was **deliberately not filed** — it is a
plausible economic win but a *measurement* question against fak's own `cachevaluereport`/cacheprice
that this pass did not run; re-opening the deliberate `#1850` stable-head-only scope should be
evidence-led, not vibe-led.

A retracted finding, kept visible as a process note: I first wrote row 1 up as a **"reverse borrow /
validation"** — Puppetmaster's all-1h cache omitting the `extended-cache-ttl-2025-04-11` beta that
"the direct Anthropic API requires," so its 1h was "silently 5m," a bug fak guards against. When the
user asked whether that was worth filing **upstream on Puppetmaster's repo**, I ran an adversarial
re-verify before touching a stranger's tracker (4 code readers + a current-Anthropic-doc check + 3
skeptical refuters + a fail-closed adjudicator). It **killed my own claim**: 1h prompt caching is
**GA** on the direct API as of 2026, the 2025 beta header was retired to a backwards-compat no-op,
and setting `ttl:"1h"` inline with no header — exactly what Puppetmaster does — is the *recommended*
form ([Claude Platform docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching);
Anthropic + Bedrock GA'd 1h by early 2026). So there is **no upstream bug and nothing was filed
against Puppetmaster**. Two second-order notes: (a) fak is *not* "strictly ahead" here — both stamp
inline `ttl:1h` correctly. (b) I then considered filing a fak-side "stale rationale" cleanup — the
beta-union (`messages_tooldefer.go:60`) and the `guard_cache_posture:42` / self-check "ttl:1h without
it is 400'd upstream" comments read as historical under current GA docs — and **declined**. fak's
union is safe in *both* worlds (no-op if 1h is fully GA, load-bearing if any account-revision/model
guard fronts hasn't GA'd it) and is fail-open + revision-gated by design (`messages_tooldefer.go:44-46`),
so it is **not a defect**. Advocating to remove a live validation gate would rest on a GA-everywhere
premise I could not primary-source (my own doc `WebFetch` was guard-refused; the workflow's dedicated
doc-fetch agent failed; the conclusion leans on a WebSearch summary), which would repeat — in fak's
own tree — the exact over-confident mistake this note just corrected against Puppetmaster. The real
lesson runs in both directions: verify-before-file caught a confident wrong claim in *my own* notes
before it hardened into repo lore, and the symmetric discipline (don't *act* to remove a safety gate
on under-sourced evidence, even when the same evidence is enough to decline to file) kept me from
over-correcting into a second wrong edit.

## Filed tickets (one borrow each, INSPIRE from Puppetmaster MIT @ 9597e5b)

- **[#3940](https://github.com/anthony-chaudhary/fak/issues/3940)** — `feat(recall)`: MMR-style **redundancy suppression** in the top-k `Recall()`
  selection. Apply a diversity penalty (`λ·relevance − (1−λ)·max_similarity_to_selected`, Jaccard
  over row `Text`, λ high enough that provenance stays dominant) **within** the existing
  provenance → recency ordering, so a near-duplicate lower-ranked row is dropped in favour of a
  novel one **without ever** demoting a witnessed row below an un-verified claim. Reuses the
  overlap relevance `journal_index.go score()` already computes; env-gated + λ-tunable like
  Puppetmaster's `PUPPETMASTER_MEMORY_MMR`. Consumer: the recall-injection budget path; adjacent:
  the deferred `parity.go:88` dedup hook.
