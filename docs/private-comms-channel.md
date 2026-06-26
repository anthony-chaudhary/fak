# Private comms channel (stub → `fak-private`)

**This is a stub.** It names the private comms channel and tells you how to reach it.
None of the live plumbing lives here — it is all in the **`fak-private`** companion repo.
This file exists so the channel is *findable from the public tree* without ever leaking
a host, a channel id, or a token into public history.

## What the channel is

The private comms channel is the **Slack control-bridge** to the lab GPU servers (the
`DGX` boxes). It is a Slack channel driven by a small pure-Go client (`cmd/dgxbridge`):
you post a command, a session on the GPU server runs it, and you read the result back
from channel history. It is the operator's out-of-band control + comms plane for the
hardware-gated work (real-kernel-on-GPU witnesses, throughput runs) that cannot happen on
the dev box.

It is **private on purpose.** The connection subsystem speaks a private lab protocol and
carries lab identifiers, so the source is scrubbed from this public repo and the commit
gate refuses it (see [the boundary doc](dgx-slack-boundary.md)). Public `fak` keeps only
*scrubbed* evidence (generic "GPU server" language); the live channel stays private.

## Where it lives

| Thing | Location |
|---|---|
| The bridge client (source of truth) | `fak-private/tools/dgxbridge/` — start at its `README.md` |
| The companion repo | `fak-private`, normally checked out next to this clone as `../fak-private` |
| Host / channel id / token | **never in public** — they resolve from a gitignored local env file in the private repo |

## How to reach it (when `../fak-private` is available)

1. Confirm the private repo is checked out alongside this one (`../fak-private`). If it is
   not, you cannot reach the channel from here — that is the intended boundary.
2. Read **`fak-private/tools/dgxbridge/README.md`**. It is the operating runbook: the
   discovery/readback grammar, the persistent-vs-`default` session rule, and the exact
   build + run commands (with the host/channel/token that must stay private).
3. The bridge builds **inside this `fak` Go module** from the private snapshot: stage the
   `cmd/dgxbridge` + `internal/dgxbridge` files, `go build` a throwaway `dgxbridge.exe`,
   run it to enumerate the live sessions, then **remove the staged `cmd|internal/*dgx*`
   files** — the public scrub must stay intact, so never commit them. The commit gate
   (`tools/check_committed_files.py`) refuses any `cmd|internal/*dgx*` path as a backstop.

## See also

- [GPU-server / Slack boundary](dgx-slack-boundary.md) — the source of truth for *what is
  public vs private* and which gates enforce it.
- [`AGENTS.md`](../AGENTS.md) — the agent entry point links here from the repo-layout map.
