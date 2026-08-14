---
title: "Managed-cache 1h-TTL 400 — the forced --managed-cache on instant crash"
description: "Forced managed-cache 1h-TTL crashed workers with a 400 because the request lacked the extended-cache-ttl beta header; the fix unions it in on upgrade."
---

# Managed-cache 1h-TTL 400 — the forced `--managed-cache on` instant crash

**Date:** 2026-07-09
**Area:** `internal/gateway` (fak manage subscription-OAuth passthrough)
**Fix commit area:** `internal/gateway/messages_transform.go`, `messages_tooldefer.go`, `messages_compact_test.go`

## Symptom

Every worker launched with **forced** managed-cache — `FAK_MANAGED_CACHE=on` (equivalently
`fak manage --managed-cache on`), logged as `managed cache — ACTIVE (forced by --managed-cache on)`
— instant-crashed **at launch** with an upstream **HTTP 400 "malformed request"**. The worker never
completed a turn.

Measured blast radius on 2026-07-09: **21 of 21** forced-cache (ACTIVE) workers crashed this way
(pre-fix resolve logs `.dispatch-runs/resolve-*.log` matching `ACTIVE (forced`). Workers on the
default `auto`/PASSIVE posture (subscription-OAuth) were **unaffected**, which is why the fleet
kept limping — only the forced-cache lane died.

## Root cause

The gateway's managed-cache 1h-TTL rung upgrades a stable-head `cache_control` breakpoint from the
default **5-minute** tier to **`ttl:"1h"`** (see `maybeUpgradeCacheTTL1H`, gated by `s.cacheTTL1H`).

The Anthropic Messages API accepts a `cache_control` block carrying `ttl:"1h"` **only** when the
request also negotiates the **`extended-cache-ttl-2025-04-11`** beta in its `anthropic-beta` header.

The wrapped `claude` CLI negotiates its own betas (`claude-code-20250219,fine-grained-tool-streaming-2025-05-14`)
but **not** `extended-cache-ttl` — it defaults to the 5m tier and never asks for the 1h beta. So when
fak forced the 1h upgrade, the outbound body carried `ttl:"1h"` while the header lacked the required
beta → Anthropic rejected the body as malformed (400) → instant worker crash.

The defect was an **asymmetry** in `prepareServedAnthropicRequest` (`internal/gateway/messages_transform.go`):
the parallel **defer-cold-tools** path already unioned *its* beta (`toolSearchBeta`) whenever it mutated
the body, but the **TTL-upgrade** path had **no analogous union**. So the forced 1h posture shipped a
body the header didn't authorize.

## Fix

`internal/gateway/messages_transform.go` (~L152) — mirror the toolSearchBeta union: when the TTL
upgrade fired, union the extended-cache-ttl beta into the forwarded `anthropic-beta` set.

```go
// Beta union (managed-cache 1h TTL): the body now carries cache_control ttl:"1h", which
// Anthropic accepts only with the extended-cache-ttl beta negotiated. The wrapped claude CLI
// defaults to the 5m tier and does not send it, so union it in ourselves.
if ttl1hUpgraded {
    upstreamBeta = unionBeta(upstreamBeta, extendedCacheTTLBeta)
}
```

- `extendedCacheTTLBeta = "extended-cache-ttl-2025-04-11"` — declared at `internal/gateway/messages_tooldefer.go:60`.
- `unionBeta` unions comma-separated beta tokens (dedup, order-preserving), so the inbound
  claude-CLI betas **survive** alongside the added one — it is a UNION, not a replace.
- Gated on `ttl1hUpgraded`, so when managed-cache is off/passive (no `ttl:"1h"` in the body) the
  beta is **not** added — a gratuitous beta on a 5m body would be wrong.

## Regression test

`internal/gateway/messages_compact_test.go` → **`TestPrepareServedRequestUnionsExtendedCacheTTLBeta`**,
two subtests:

- `upgrade_fires_unions_beta`: with `s.cacheTTL1H = true`, asserts (a) the served body gains
  `"ttl":"1h"`, (b) `prep.upstreamBeta` contains `extended-cache-ttl-2025-04-11`, and (c) the inbound
  betas (`claude-code-20250219`, `fine-grained-tool-streaming-2025-05-14`) survive — union not replace.
- `no_upgrade_no_beta`: with the rung off, asserts no `ttl:"1h"` in the body and the beta is absent.

## Deploy

Built and hot-swapped into the three live fak binaries this session; each verified to contain the
`extended-cache-ttl-2025-04-11` literal in its bytes:

- `<windows-user-home>\bin\fak.exe`      (fleet launcher on PATH)
- `C:\work\fak\tools\.bin\fak.exe` (dispatch / worker binary)
- `C:\work\fak\fak.exe`            (repo root)

Old binaries backed up as `*.bak-20260709-0738*`.

## Scope: this is an API-key-path fix — leave subscription seats PASSIVE

> **Superseded (operator policy, 2026-07-10).** The "leave subscription seats PASSIVE —
> zero benefit (flat-rate)" guidance below was correct *as the default on 2026-07-09*, but
> is superseded on two counts. (1) **Policy:** `guard_cache_posture.go` now maps an unset
> `FAK_MANAGED_CACHE` → `on`, so fleet launchers force the 1h upgrade on **every** seat,
> subscription included (best-effort managed cache everywhere). (2) **Reasoning:** "zero
> benefit on flat-rate" was too strong — a subscription's binding constraint is the
> **compute-weighted usage limit**, not dollars, and cache reads cost ~0.1× the compute of
> a fresh write *and* don't count against the rate limit, so avoiding cold prefix rewrites
> buys **usage-limit headroom**. The one open item below still stands: whether the
> subscription-OAuth wire returns the read-rebate end-to-end is witnessed via
> `fak cachevalue report`, not assumed. See
> [what-is-managed-cache.md](../explainers/what-is-managed-cache.md) for the reader-facing
> version. The `off` opt-out remains the escape hatch for a seat where `on` misbehaves.

**Critical, do not skip.** 1h-TTL managed cache is **API-key-billing-only by design**
(`cmd/fak/guard_managed_cache.go`): the 1h tier **doubles the cache-write cost**, and on a
Pro/Max **subscription (flat-rate)** the marginal token price is flat while the provider cache
already rides the client's own breakpoints — so **AUTO stays PASSIVE on subscription-OAuth and
activates only when fak knows the session bills an API key** ("never speculate with someone
else's billing"). `--managed-cache on` / `FAK_MANAGED_CACHE=on` forces past that gate.

So this fix matters on the **API-key path**, where AUTO legitimately turns 1h-TTL on and the
missing beta would have 400'd it too. On **subscription seats, forcing managed cache is
off-label**: zero benefit (flat-rate) + 2× write cost, and whether the now-well-formed 1h
request even clears the subscription-OAuth wire is **unproven** — no clean ACTIVE worker has
been witnessed on a subscription seat post-fix. **Leave managed cache `auto`/passive on
subscription.** (The fable fleet runs on subscription-OAuth seats, so its correct posture is
passive; do not add `FAK_MANAGED_CACHE=on` to its spawn recipes.)

## Reproduce / verify

- **Unit (proven):** `go test ./internal/gateway -run TestPrepareServedRequestUnionsExtendedCacheTTLBeta`
  → green. Proves the header is now built correctly when the upgrade fires.
- **Live wire (API-key path, the fix's real target):** an API-key-billed session runs AUTO →
  1h-TTL active → the beta is now unioned → no `400 … malformed`. Pre-fix this path 400'd too.
- **Live wire on subscription (unproven, low-value):** `--managed-cache on` on a
  subscription seat *forces* ACTIVE. Pre-fix: instant `API Error: 400 … malformed`. Post-fix:
  the request is now well-formed, but acceptance on the subscription-OAuth wire is not yet
  witnessed — and it buys nothing there, so it is not worth a scarce seat to prove.

## Related learnings

- Memory `managed-cache-1h-ttl-400-missing-beta` — the reconciled one-fact summary of this note.
- Memory `claude-fable-5-confirmed-works` — where the 400 was first correctly correlated to
  forced cache (its "leave passive on subscription" guidance stands; this note adds the code cause).
- Memory `spawn-fable-guarded-worker-recipe` / `fable-cached-dispatch-knobs` — fable spawn
  recipes; corrected so `FAK_MANAGED_CACHE=on` is API-key-only, not a subscription default.
- Memory `fleet-binary-hotswap-three-copies` — how the fixed binary was deployed to all three `fak.exe`.
- `docs/notes/MANAGED-CACHE-PROVING-GROUND-2026-07-03.md` — the C6 1h-TTL lever's evidence
  ladder (rung `ttl_upgrade_1h`); notes the upgrade is "gated to API-key-billed sessions".
- `docs/notes/GUARD-OWN-CACHE-VALUE-PATH.md`, `FAK-GUARD-CACHE-VALUE-SHARE-2026-07-01.md` —
  why fak's guard-path cache share is ~0 on subscription (the provider owns caching there).
