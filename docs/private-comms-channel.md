---
title: "Private comms channel (public stub)"
description: "A public stub that names fak's private control bridge to the lab GPU servers and shows how to reach it, without leaking any host, channel, or token."
---

# Private comms channel (stub -> `fak-private`)

**This is a stub.** It names the private comms channel and tells you how to reach it.
None of the live plumbing lives here — it is all in the **`fak-private`** companion repo.
This file exists so the channel is *findable from the public tree* without ever leaking
a host, a channel id, or a token into public history.

## What the channel is

The private comms channel is the **control bridge** to the lab GPU servers. It is driven
by private client code: you post a command, a session on the GPU server runs it, and you
read the result back from the private transcript. It is the operator's out-of-band control
and comms plane for the hardware-gated work (real-kernel-on-GPU witnesses, throughput
runs) that cannot happen on the dev box.

It is **private on purpose.** The connection subsystem speaks a private lab protocol and
carries lab identifiers, so the source is scrubbed from this public repo and the commit
gate refuses it (see [the boundary doc](gpu-server-private-boundary.md)). Public `fak` keeps only
*scrubbed* evidence (generic "GPU server" language); the live channel stays private.

## Where it lives

| Thing | Location |
|---|---|
| The bridge client (source of truth) | private repo bridge docs |
| The companion repo | `fak-private`, normally checked out next to this clone as `../fak-private` |
| Host / channel id / token | **never in public** — they resolve from a gitignored local env file in the private repo |

## How to reach it (when `../fak-private` is available)

1. Confirm the private repo is checked out alongside this one (`../fak-private`). If it is
   not, you cannot reach the channel from here — that is the intended boundary.
2. Read the private bridge README. It is the operating runbook: the
   discovery/readback grammar, the persistent-vs-`default` session rule, and the exact
   build + run commands (with the host/channel/token that must stay private).
3. The bridge builds **inside this `fak` Go module** from the private snapshot: stage the
   private bridge command and support package, build a throwaway bridge binary, run it to
   enumerate the live sessions, then **remove the staged private bridge files**. The public
   scrub must stay intact, so never commit them. The commit gate refuses the private bridge
   path class as a backstop.

## Operating recipe (the part that bites — read this)

Once the bridge is built from the private snapshot, these are the rules that
separate "it works" from "I wrongly concluded the bridge is dead." They carry **no private
values** — the host/channel/token resolve from the gitignored env file in `fak-private`.

**The bridge is usually LIVE but SLOW.** It is a private round-trip through a hub transcript,
not SSH. A short probe is the single biggest trap: a status probe with the default
`-probe-wait` (or a sub-minute `-timeout`) routinely returns `STALE (no control reply within
timeout)` / "an operator must restart the bridge" when the shell is **actually fine**. That is
a **false negative**, not a dead bridge.

The recipe that works — probe patiently, run a real command, and do it in the background so a
2-minute foreground cap can't truncate the round-trip into a false negative:

```sh
# Confirm a live session and pick it with one cheap real command.
<private-bridge> -probe -probe-wait 90s -settle 12s -timeout 5m run 'echo BRIDGE_OK'
```

- **Patient flags:** `-probe-wait 90s -settle 12s -timeout 5m`. The default 15s probe-wait is
  too short for a busy box.
- **Run it backgrounded** (your harness's background mode) so the slow round-trip completes
  off the foreground clock.
- **Prefer a real command over a bare `status`** — `run 'echo … $(hostname)'` both proves
  liveness and prints which session it picked.
- **Multi-line output** from a single `run` can lose the async transcript tail. For anything
  beyond a line or two, wrap the output in a **nonce sentinel** (`echo NONCE_X; …; echo
  NONCE_END_X`) and read between the sentinels, or use the private helper's background/poll
  mode for a long job that writes to the private scratch area.
- **Per-box channel** is selected with `-channel <id>` (the ids live in `fak-private`'s
  node→channel map); omitting it uses the default control channel.

If `-probe` genuinely finds only STALE banners after a patient wait, *then* an operator must
(re)start the remote control shell — a bare `default` login shell exits before delayed stdin,
so the box needs a persistent/tmux control session.

## See also

- [GPU-server private boundary](gpu-server-private-boundary.md) — the source of truth for *what is
  public vs private* and which gates enforce it.
- [`AGENTS.md`](../AGENTS.md) — the agent entry point links here from the repo-layout map.
