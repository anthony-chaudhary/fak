---
title: "Private control channel"
description: "The public route for authorized operators who need the private control bridge to fak's lab GPU servers."
---

# Reach the private control channel

**Audience:** an authorized fak operator who needs to run or read back work on a lab GPU
server. Public readers can use this page to understand the boundary; the credentials,
node map, commands, and transcript remain in the private companion repository.

**Next action:** after updating your authorized `fak-private` checkout through its normal
`main`-branch workflow, open `../fak-private/tools/dgxbridge/README.md` from the `fak`
clone root and follow its discovery and readback procedure. Access to that private repository
is the authorization check. If the sibling or your access is absent, stop and ask a fak lab
maintainer for access; do not reconstruct the route from public notes.

## What this route controls

The channel is fak's private, out-of-band control bridge to the lab GPU servers. An
authorized operator submits a command through the bridge, a persistent server session runs
it, and the operator reads the result from the private transcript. Use it for hardware-gated
work such as real GPU-kernel witnesses and throughput runs.

This public page is a **route, not an operating runbook**. The current commands, session
selection rules, node map, credentials, and recovery procedure are maintained together in
the private README named above. Following that runbook is the supported choice; copying an
old command from a public note is not.

## Current boundary and support

| Context | Current contract |
|---|---|
| Mode | Private operator control; it is separate from the public `fak serve` data path. |
| Generation | Current public `fak` plus the updated `main` version of the adjacent authorized `fak-private` checkout. The private README, not this stub or a dated public note, is the command authority. |
| Lifecycle | Active lab-operator route. Public notes and benchmark records are evidence or history, not bridge instructions. |
| Support | Maintained for authorized lab operators. A public-only checkout intentionally provides no live access. |

The public repository may contain scrubbed outcomes in generic “GPU server” language. It
must not contain a host, endpoint, channel identifier, token, account identifier, raw
transcript, or staged private bridge source. The commit-time admission and public-leak gates
enforce that boundary; see the
[GPU-server private boundary](gpu-server-private-boundary.md) for ownership and gate details.

## When a readback fails: which class, which fix

A failed readback is not one condition. The bridge client separates the classes because
they have different fixes, and until #5103 they all surfaced under a single
`READBACK_WEDGED` token — which sent operators to restart sessions that were never the
problem. Read the class before acting.

| Class | What it proves | Operator action |
|---|---|---|
| `READBACK_WEDGED: sentinel_missing` | The hub answered this session's tail request and the client parsed the reply, but the command's completion marker was not in it. The transport is healthy; the session's shell is wedged. | Restart the control bridge for that session on the server. This is the only class that warrants it, and it is server-side: an authorized operator must do it from the server console, because the boundary allows no inbound login. |
| `HUB_UNRESPONSIVE: hub_timeout` | The poll itself deadlined or errored, so no reply was ever inspected. A slow or unreachable control transport. | Retry in a quieter window with a longer timeout. Do not restart anything; the session is not implicated. The transport oscillates, so a failure here is not durable. |
| `HUB_UNRESPONSIVE: no_tail_reply` | Every poll succeeded and the hub answered nothing at all. | Check that the hub process is running and still joined to the control channel. Again a hub-side condition, not a session one. |
| `HUB_UNRESPONSIVE: tail_reply_unrecognized` | The hub replied, but no reply matched the shape this client parses — client/hub protocol drift. | Update the client's reply parser in the private tree. Restarting sessions cannot fix a parser mismatch. |

Two consequences worth keeping. First, a spent probe budget never reports a wedged shell:
if the loop never polled, it cannot have observed a missing marker, and it says so.
Second, the same session can produce different classes minutes apart — a short fail-fast
probe can deadline (`hub_timeout`) while a full-budget check on the same session reaches
the hub and reports `sentinel_missing`. That is not a contradiction; it means the shell is
wedged *and* the transport is slow, and only the first needs an operator.

Run the client's `selftest` verb before committing to a long command. It uses the full
timeout instead of the short fail-fast probe budget, so it is the cheapest way to learn
which class you are in.

## Public readback

When lab capacity is being considered for dev-worker dispatch, follow the scrub and schema
rules in [`fleet.md`](fleet.md), then derive the public `fak.lab_readiness/v1` record with
`fak lab readiness --from-reports --write-default --json`. Only
`READY_FOR_DEV_WORK` admits lab-backed dispatch. The other public-safe outcomes—
`WAIT_PRIVATE_RECOVERY`, `GATEWAY_UNREACHABLE`, `AUTH_OR_CHANNEL_BLOCKED`, and
`INDETERMINATE`—hold dispatch while an authorized operator follows the private recovery
runbook.

## Related routes

- [Fleet compute nodes](fleet-compute-nodes.md) — choose the sanctioned compute target for a
  hardware-gated task.
- [GPU-server private boundary](gpu-server-private-boundary.md) — decide what may cross from
  private control into public evidence.
- [Lab development loop](fak/lab-dev-loop.md) — return a scrubbed hardware witness to the
  public development workflow.
