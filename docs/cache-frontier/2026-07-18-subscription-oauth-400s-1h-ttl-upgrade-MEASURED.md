---
title: "Measured: the subscription-OAuth wire 400s the managed-cache 1h-TTL upgrade"
description: "Cache-frontier finding for #2183 / epic #1844 C6: a live controlled probe measures what the 2026-07-04 finding left as a missing witness. Forcing `--managed-cache on` on a Pro/Max subscription-OAuth seat makes the outbound 1h-TTL request 400 (`upstream rejected the request as malformed`) even on a current binary with the 2026-07-09 extended-cache-ttl beta union. The passive-on-OAuth AUTO default is therefore MEASURED-correct, not merely assumed."
---

# Measured: subscription-OAuth 400s the 1h-TTL managed-cache upgrade

- Issue: [#2183](https://github.com/anthony-chaudhary/fak/issues/2183) (epic
  [#1844](https://github.com/anthony-chaudhary/fak/issues/1844) C6).
- Date: 2026-07-18.
- Verdict: **measured-negative** — activating the managed-cache 1h-TTL upgrade on a
  subscription-OAuth seat is rejected by the provider with an HTTP 400. The
  passive-on-OAuth `--managed-cache` AUTO default is correct on the acceptance axis,
  now with a live witness instead of an untested prior.

## What the 2026-07-04 finding left open

[`2026-07-04-oauth-ratelimit-cache-read-finding.md`](2026-07-04-oauth-ratelimit-cache-read-finding.md)
classified the "activate AUTO on OAuth" bet as `gen/future`: the passive default rested
on the *dollar* axis (flat-rate subscription → no billing win), and the rate-limit-headroom
upside was unmeasured. It named the missing witness. A prior question sat underneath that
one and was never tested directly: **does the subscription wire even ACCEPT a well-formed
1h-TTL request?** The 2026-07-09 fix
([`MANAGED-CACHE-1H-TTL-400-FIX`](../notes/MANAGED-CACHE-1H-TTL-400-FIX-2026-07-09.md))
made the request well-formed (unions the `extended-cache-ttl-2025-04-11` beta when the
head is upgraded), but whether the provider honors 1h on a subscription seat stayed
"not proven" (no clean ACTIVE worker witnessed on a subscription seat post-fix).

## The probe

One guarded `claude` turn, managed cache forced ACTIVE, on the live subscription wire:

```
fak guard --managed-cache on -- claude --dangerously-skip-permissions -p "ok"
```

Guard startup banner confirmed the lever armed:

> `fak guard: managed cache — ACTIVE (forced by --managed-cache on): stable-prefix
> cache_control upgraded to the 1h TTL tier on the outbound wire …`

Outbound result, repeated across the turn's requests:

> `API Error: 400 upstream rejected the request as malformed (HTTP 400) — check the
> model name, message roles, and parameter ranges`

Binary: current trunk (build `e0fba267f64c`), i.e. WITH the 2025-04-11 beta union. The
same seat runs cleanly with the AUTO/passive default — the durable gateway-usage ledger
shows 1000+ subscription sessions serving with large provider `cached_prompt_tokens`
(the provider 5m prompt cache and fak's default-on star-anchor breakpoint placement both
work) and zero 1h-TTL upgrades. So the controlled A/B is unambiguous: **passive = serves;
forced-on = 400.**

## Result

- The provider rejects the 1h-TTL (`ttl:"1h"`) cache_control on the subscription-OAuth
  wire even when the request carries the extended-cache-ttl beta. The 1h tier is not
  available on this credential class.
- `--managed-cache` AUTO staying passive on subscription-OAuth
  ([`cmd/fak/guard_managed_cache.go`](../../cmd/fak/guard_managed_cache.go) `oauthSource != ""`
  branch) and the fleet's `FAK_MANAGED_CACHE=auto` posture are **measured-correct**, not
  merely conservative. Flipping the fleet default to `on` would 400 every subscription turn.
- The `fak score cache-health` `managed_cache_posture` / `upgrade_fired_rate` families
  reading low on a subscription fleet is therefore expected, not remediable debt: the one
  lever they measure is provider-blocked on this credential class. What IS working on these
  seats — provider 5m prompt cache + star-anchor placement + compaction — is not captured
  by those two families.

## The activation paths that remain

1. **API-key billing** — set `FAK_GUARD_API_KEY_ENV` so AUTO resolves ACTIVE on an
   API-key-billed seat. The API wire negotiates the beta and (per the C6 design) accepts
   1h; this is the sanctioned way to reach the 1h lever, at the cost of real dollars
   instead of flat-rate subscription.
2. **Graceful 400-fallback (recommended next step)** — teach the gateway to catch the
   1h-TTL 400 and retry the turn once with the upgrade stripped (the byte-identical 5m
   body), so a forced/best-effort `on` DEGRADES to the provider 5m cache instead of failing
   the turn. That would make `--managed-cache on` safe to default everywhere: active where
   the wire accepts 1h, transparently passive where it does not. Until then, subscription
   seats must stay AUTO/passive to avoid the crash.

## Provenance

Live single-session probe, not a fixture. The 400 is the provider's, surfaced verbatim by
the guard; the ACTIVE posture is the guard's own resolved banner. No fleet config was
changed — the probe forced the posture on its own throwaway session only.
