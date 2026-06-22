# Account switching, and why a provable speed-up doesn't need control of the kernel

> An ultracode fan-out launches many workers at once. If every worker authenticates as
> the *same* account, they all draw on **one** rate-limit bucket and the fan-out quietly
> serializes — you bought N workers and got 1 account's throughput. The **account
> switcher** (`tools/fleet_accounts.py`) hands each lane a *distinct* account, so N lanes
> draw on N independent per-account limits. The surprising part is that this is a
> **provable** win even though we control none of the things that look like they'd have
> to be involved: not the OS kernel, not Claude Code's internals, not Anthropic's serving
> engine or its rate limiter, and not whatever upstream features happen to be enabled.
> This page explains why it isn't magic.

This is the same move `fak` makes everywhere, applied to *credentials*. `fak` treats the
model like an untrusted program and the tool call like a syscall that
[passes through a kernel the model doesn't control](policy-in-the-kernel.md): you don't
secure an agent by owning the silicon, you secure it by owning the **one boundary the
effect has to cross**. Account switching is the throughput twin of that idea. You don't
get parallel speed-up by owning Anthropic's rate limiter; you get it by owning the **one
variable decided on your side of the wire** — which account each request is attributed
to.

## The thing that feels like magic

The setup that trips people up:

- We do **not** control the OS kernel — `fak` and the switcher are ordinary userspace.
- We do **not** control Claude Code's internals — it's a closed CLI we point at an
  endpoint.
- We do **not** control Anthropic's serving engine, its **rate limiter**, or its
  internal accounting — that's upstream, opaque, and not ours.
- We do **not** know **which upstream features are enabled** — prompt caching, request
  batching, priority tiers, per-route limits: we can't see the flags and they can change
  under us.

And yet a single `wave --explain` call reports a concrete, reproducible multiplier
(`headroom_multiplier: 3` on the box this was written on). How can a number be *provable*
when we're blind to almost everything that produces the limit it's multiplying?

## The reveal: throughput comes from *placement*, not from owning the substrate

The cleanest analogy is the one operating systems already settled. An OS scheduler does
not manufacture CPU cores. It cannot change the silicon, the microcode, or how an
individual core retires instructions. What it controls is **placement** — which core a
runnable thread lands on. Place two independent threads on two cores instead of one and
you provably double throughput, *without having built either core*. The speed-up is a
property of the placement decision, which the scheduler fully owns — not a property of
the hardware, which it doesn't.

The account switcher is a **credential scheduler**. The "cores" are Anthropic's
per-account rate-limit buckets: real, finite, and entirely upstream. We can't enlarge
one, can't inspect its remaining depth, can't see how the limiter is implemented. We
don't need to. We control the one decision that selects *which bucket a request lands
in* — the account a worker authenticates as (`CLAUDE_CONFIG_DIR` + the long-lived
`oauth_token`), chosen in userspace **before the request ever leaves the box**. Spread N
lanes across N buckets and you provably get N buckets' worth of headroom, for the same
reason two threads on two cores beat two threads on one. The substrate is not ours; the
**placement** is, completely.

That is why it is not magic and not fragile:

1. **The limit is keyed on a thing we choose.** A rate limit is per-account. The account
   is selected on our side of the wire. So the number of distinct accounts a fan-out
   places lanes onto is a quantity we *compute*, not one we discover from the upstream —
   and the per-account **rate** limit (the one that throttles a *burst*) is keyed exactly
   on that account. Whether two distinct accounts are *also* independent over longer
   horizons — no shared org-level or weekly cap — is upstream and opaque; that's the one
   honest gap, caveated below. The burst-rate headroom multiplies regardless.

2. **The win is arithmetic, not a feature.** `distinct_pools` counts the distinct accounts
   a fan-out spreads across; multiplying burst headroom by spreading load across them is
   addition of per-account rate capacities, not a behavior we have to switch on. It does
   not depend on prompt caching being on, on the serving engine being any particular one,
   or on the kernel doing anything special.

3. **It's structural, like the capability lock.** `fak`'s security floor doesn't depend
   on *catching* an attack; the dangerous
   [lever was never wired up](../concepts-and-story.md). The throughput floor here is the
   same shape: a wave lane *cannot* contend with a sibling lane's bucket, because it was
   handed a different bucket. There's no recognizer to fool and nothing to fail open.

## Why "we don't know which features are enabled" is a feature, not a bug

The benefit is **substrate-independent by construction** — the same property the repo
already claims for `fak`'s deterministic gates, which
[reproduce byte-for-byte across four hardware platforms](../HARDWARE-MATRIX.md) `fak`
does not control. We make **zero assumptions about enabled upstream features**:

- If caching is on, each lane caches against its own account; the wave neither needs nor
  breaks it. (The dogfood path already forwards `cache_control`
  [byte-for-byte in passthrough](../../DOGFOOD-CLAUDE.md) so a real upstream cache hit
  survives — orthogonal to, and composable with, the wave.)
- If the upstream adds priority tiers or per-route sub-limits tomorrow, the wave still
  spreads load across *accounts*, which is the coarsest and most durable bucketing there
  is.
- If we're wrong about the upstream's internals, we lose nothing, because we never relied
  on them. The only fact we lean on is "rate limits are per-account," which is the one
  thing about the limiter that is contractual and visible.

A benefit you have to *enable the right feature* to get is fragile — it evaporates when a
flag flips. A benefit that comes from **how you attribute requests** is robust, because
attribution is decided entirely on the side of the wire you own.

## "Provable" means witnessed, not asserted

`fak`'s honesty discipline is that a claim is only as good as the
[witness that the agent did not author](../proofs/witness.md) — *the kernel is the part
that doesn't believe the agents*. The wave's multiplier is held to the same bar. It is
not a number we declare; it's derived from ground truth and checkable three ways:

- **From on-disk identity.** Distinctness is by Anthropic `accountUuid`, read from each
  dir's `.claude.json` — not by directory name (a no-login or non-Claude dir, having no
  uuid, is its own pool keyed by directory name). Two dirs logged into one account are
  **one** pool and the wave hands out only one (the same identity reconciliation that
  stops the roster from presenting one throttled account as three healthy workers). The
  multiplier counts distinct *accounts* — the coarsest, most durable bucketing the
  per-account limit is keyed on, and a lower-bound proxy for truly independent buckets
  (see the caveats).

- **From a test.** `WaveAllocationTest` in `tools/fleet_accounts_test.py` pins the claim:
  `test_wave_beats_naive_resolve_burst` builds one roster, then — changing nothing but
  wave-vs-burst — shows a naive `resolve()` ×3 collapses to **1** distinct pool while
  `allocate_wave(3)` returns **3**. The serialization the wave removes is demonstrated,
  not described.

- **Reproducibly, from one command.** `wave --explain` prints `distinct_pools` and
  `headroom_multiplier` for the live roster, derived from the real accounts on the box:

  ```bash
  python tools/fleet_accounts.py wave --count 4 --t1 --explain
  # -> granted 3, distinct_pools 3, headroom_multiplier 3, naive_pools 1,
  #    lane_pools = three distinct accountUuids; shortfall 1 (only 3 tier-1 pools exist)
  ```

  The `shortfall` is part of the honesty: the wave **cannot conjure capacity that isn't
  there**. Ask for 4 tier-1 lanes when 3 distinct tier-1 accounts are available and it
  grants 3 and reports the deficit — it never silently re-collapses two lanes onto one
  bucket to hit the number.

## What this is, precisely — and what it is not

- It multiplies **rate-limit headroom**, i.e. how many lanes can be in flight before any
  one account throttles. It does **not** make a single request faster, change model
  quality, or raise any one account's own limit (it respects each account's limit
  exactly; it just stops piling every lane onto one).
- It is about an **operator's own authorized accounts** — the fleet's `gem5 / gem7 / gem8`
  org accounts on the box here — used within each account's own terms. The wave spreads a
  fan-out across accounts the operator already holds; it is fleet placement, not limit
  evasion on a single account.
- **`distinct_pools` counts accounts, which is a *proxy* for independent limit buckets —
  not a measured one.** The fleet has no remaining-quota telemetry (a probe returns only
  OK / blocked, never headroom), so the wave cannot see whether two distinct accounts
  share an upstream org-level or **weekly** cap (the live roster shows exactly such a
  weekly window on one account). The per-account *rate* limit that throttles a burst
  multiplies cleanly; a shared longer-horizon total is a separate ceiling the wave does
  not lift. So `distinct_pools` is an *upper bound* on realized headroom — honest about
  what it counts (accounts) and what it cannot see (the upstream's bucket topology). This
  is the same "we don't control the substrate" gap the whole page is about, stated against
  our own number.
- The multiplier is **bounded by reality**: `distinct_pools` is `min(count, distinct
  available accounts at the requested tier)`. A box with one account gets `1×` and the
  wave says so. The honest under-fill is the point.
- The in-process Workflow fan-out (subagents under *one* Claude Code session) shares that
  session's single credential and is **not** what this multiplies. The wave is for the
  **headless fleet** path — N detached workers, one distinct account each — which is where
  per-account limits are the binding constraint.

## Where it plugs in

`wave` is a sibling of the single-account `resolve` front door every dispatcher already
uses ([`DOGFOOD-CLAUDE.md`](../../DOGFOOD-CLAUDE.md)). A fan-out makes **one** call and
gets N ready-to-launch lane records — each the same flat shape `resolve` returns
(`config_dir` to pin as `CLAUDE_CONFIG_DIR`, `oauth_token` to serve on, `selected_tier`),
plus a `pool` key for distinctness auditing:

```bash
python tools/fleet_accounts.py wave --count 8 --work-kind engineering
# -> {ok, requested:8, granted, shortfall, distinct_pools, lanes:[{config_dir, oauth_token,
#     tier, pool}, ...]}  -- launch one detached worker per lane, each on its own bucket.
```

The launcher loop is mechanical: for each lane, set `CLAUDE_CONFIG_DIR` /
`CLAUDE_CODE_OAUTH_TOKEN` and spawn a detached `claude` — the exact wiring
`tools/launch_goal_detached.ps1` already proves for one worker, run once per lane on a
different bucket. The kernel idea underneath is unchanged: own the boundary the effect
crosses, and the substrate you don't control stops being something you have to control.
