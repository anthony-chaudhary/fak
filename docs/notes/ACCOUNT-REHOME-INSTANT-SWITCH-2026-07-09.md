---
title: "Account rehome — the manual \"go\" instant seat switch"
description: "How the operator-forced proactive account switch works on a live guarded session: POST /v1/fak/account/rehome, `fak accounts rehome`, and why prompting \"go\" makes it apply instantly. Versioned baseline for the automatic-push follow-up."
version: 1
---

# Account rehome — the manual "go" instant seat switch (2026-07-09)

This is the **versioned baseline** for the seat-switch behavior an operator already
has on a live `fak manage` session, written down before we improve it. It documents
what exists today and the one behavior that reads as magic — prompting **"go"** and
watching the session hop to a fresh account instantly. The follow-up (make the
background tick do this push automatically) is tracked separately; this doc is the
"what we have now" it builds on.

## The three ways a session changes account — don't conflate them

| Mechanism | Trigger | Code | Registry change? |
|---|---|---|---|
| **Reactive auto-failover** | An account-scoped **403/429 mid-turn** (org OAuth/subscription disabled, region/billing wall, live 429 cap) | `accountFailover.failover` (`cmd/fak/guard_account_failover.go`) | No — heals in place |
| **Manual rehome ("go")** | Operator asks for it: `POST /v1/fak/account/rehome` / `fak accounts rehome` | `accountFailover.forceRehome` | No — only the live serving seat moves |
| **Tombstone rehome** | `fak accounts remove --rehome-to` | registry mutation | **Yes** — permanent |

This doc is about the **middle row**. It is the *proactive, on-demand* form of the
reactive failover: instead of waiting for a wall to hit, the operator switches the
seat **before** the wall, on demand.

## What the manual rehome does

`POST /v1/fak/account/rehome` (wired into the guard's gateway via
`srv.SetAccountRehomeFunc(af.forceRehome)` in `cmd/fak/guard.go`) runs
`accountFailover.forceRehome`:

1. Discovers the sibling roster under `~` (`accounts.Discover`).
2. Picks the **next available** seat — a *different* account that is enabled, logged
   in, holds a live (non-expired) access token, and is **not** already walled or
   rehomed-off this session (`pickFailoverAccount`).
3. Records the seat being **left** in the `moved` set — so a later *automatic*
   failover never bounces the session back onto a seat the operator deliberately
   left. It is **not** marked `walled`: it was not proven bad, and the status area
   must not claim it was.
4. Advances the sticky `currentDir`, so the per-request token source (`apiKeyFunc`)
   reads the adopted account's rotating token from then on.

The returned metadata is **seat display identity only** (name + email) — never a
token.

When no sibling qualifies, the state is left untouched (the session keeps its seat)
and the error names the real fix, derived from the picker's typed
`failoverNoTargetReason`: `no_siblings` → enroll one; `all_walled` → wait/enroll;
`needs_login` → `claude /login` under the seat's `CLAUDE_CONFIG_DIR`;
`all_disabled` → re-enable/remove.

## Why prompting "go" switches "instantly"

The swap is **staged, not immediate**: `forceRehome` advances `currentDir`, but the
new token is only read on the session's **next upstream request** (`apiKeyFunc` runs
per request). So:

- If the session is **mid-turn**, the swap lands on the next turn automatically.
- If the session is **idle** (waiting on the operator), nothing crosses the wire
  until the operator sends *something*. Typing **"go"** (or any prompt) is what
  produces that next upstream request — and the staged seat swap rides it. That is
  why it *looks* instant: the switch was already decided; "go" is just what flushes
  it onto the wire.

This is the crux the automatic-push follow-up targets: today a **human** supplies the
"go" that flushes a pending/needed switch. The improvement is to let the background
tick supply it when the active seat is near its cap.

## Operator surface

```
fak accounts rehome [--addr <gateway-url>] [--key <token>] [--reason <text>] [--json]
```

- `--addr` — the gateway URL from the guard banner (or `FAK_ADDR`).
- Exit `0` on an applied swap; exit `1` on transport error or gateway refusal
  (`404` no roster in force — is `fak manage` running?; `409` no available sibling).
- Human output names `from -> to (reason ...)` and reminds that the swap applies on
  the session's next upstream request and the left seat is excluded from
  auto-reselection for the rest of the session.

Code: `cmd/fak/accounts_rehome.go` (CLI), `cmd/fak/guard_account_failover.go`
(`forceRehome` + `pickFailoverAccount` + `failoverNoTargetReason`),
`cmd/fak/guard.go` (wiring), gateway route `/v1/fak/account/rehome`.

## Near-cap signal already available (the input the tick will read)

`internal/fleetaccounts` already computes the offerability signal the automatic push
needs:

- **Headroom tier** — `accounts.Classify(score)` → `TierWalled` / `TierUnknown` /
  `TierOfferable` (`internal/accounts/headroomtier.go`). Only `TierOfferable` is
  proven safe to prefer.
- **`usage_soon_reset`** — an advisory a fresh OK probe carries when it reopened a
  seat over a still-active **daily** cap (`markUsageSoon`,
  `RuntimeStatus.UsageSoonReset` in `internal/fleetaccounts/status.go`). The seat
  stays *Available* but is flagged as about to re-wall.

These are exactly the "the active seat is about to run out" facts a proactive tick
would branch on — no new signal source is required, only a consumer.

## Baseline invariants the follow-up must preserve

1. Never switch **onto** a walled or dead-token seat (`pickFailoverAccount` already
   enforces this).
2. Never bounce back onto an operator-`moved` seat.
3. A refusal is a **typed reason with one fix**, never silent.
4. The swap is applied by the per-request token source — nothing restarts the
   session.

## Progress — the consumer now exists (2026-07-09)

The "only a consumer is required" gap above is now closed at the **read layer**,
still fully dormant (zero live-behavior change — nothing in `guard.go` calls it):

- **`proactiveSignalsForSeat(activeDir, roster, hr)`** (`cmd/fak/guard_account_autorehome.go`)
  is that consumer. It projects the annotated fleet roster + the rotation-headroom map
  (`rotationHeadroom` → `headroomFromRoster`, the same fold `fak accounts next/launch`
  builds) down to the **one active seat**, returning the exact
  `(headroom *float64, usage_soon bool)` pair `proactiveWantsSwitch` branches on. It
  matches the seat by normalized dir (`guardrotate.NormalizeDir`), reads the bucket
  score through `hr` (nil = no signal, distinct from a 0 `TierUnknown`), and reads
  `usage_soon` off the row's `UsageSoonReset`. Pure — table-tested without a live fleet.
- **`accountFailover.proactiveTickFromRoster(enabled, roster, hr)`** composes it with the
  proven `proactiveRehomeTick`: resolve the active seat's signals, then drive the swap.
  No I/O — the caller passes roster + headroom in — so `roster → decision → swap` is
  tested end to end (`TestProactiveTickFromRoster`).

**The only step left is the live driver:** a guard-side tick cadence + an enable flag
that loads the roster/headroom (as `rotationHeadroom` already does) and calls
`proactiveTickFromRoster` on the active session. The spine, the signal reader, and the
composition are all proven first; the loop that supplies the automatic "go" lands on
top and preserves every baseline invariant above.
