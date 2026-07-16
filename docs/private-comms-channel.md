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
