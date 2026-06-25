# A2A Channel Demo

`a2ademo` is a no-key, no-model proof that fak can adjudicate agent-to-agent messages inside the kernel. It exercises point-to-point delivery, the capability floor, cross-session handoff, cross-window self-handoff, and pub/sub fanout.

## Prerequisites

Requires Go only. It does not need a model, API key, GPU, network service, or files outside this repo.

## Quick Start

From the repo root:

```bash
go run ./cmd/a2ademo
```

The run completes in a few seconds and returns exit code 0 only if every invariant holds. Output is deterministic because the demo uses an in-memory bus and fixed messages.

## What You See

```text
fak a2achan - in-kernel agent-to-agent message channel (no key, no model)
    alpha SEND shared -> work                    -> ALLOW
    alpha SEND private -> another agent's channel -> DENY
    coordinator PUBLISH (fanout=2)               -> ALLOW

a2ademo: OK - adjudicated A2A delivery across in-kernel / session / window locales + pub/sub
```

## What This Does Not Claim

This demo does not claim networked, multi-process, or cross-host A2A transport. It proves the in-kernel channel semantics and capability checks over a deterministic in-memory bus.

## Related Docs

- [A2A in-kernel channel](../../docs/a2a-in-kernel-channel.md)
- [fak architecture](../../ARCHITECTURE.md)
