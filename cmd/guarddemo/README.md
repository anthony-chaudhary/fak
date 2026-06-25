# guarddemo - safety floor side by side

`guarddemo` replays the same attack trace twice: once without fak in the loop and once
through the real kernel. The browser view shows the breach count collapse to zero behind
fak, and the terminal modes use the same replay path.

## Prerequisites

Go 1.26+ from the repository root. No model, GPU, API key, or network is required. The
run is deterministic and usually completes in a few seconds for `-print` or `-selfcheck`;
the browser server starts in seconds and waits for a local browser.

## Run it

```bash
go run ./cmd/guarddemo
go run ./cmd/guarddemo -print
go run ./cmd/guarddemo -selfcheck
FAK_DEMO_BASE_PATH=/guarddemo go run ./cmd/guarddemo
```

Open the printed loopback URL, pick a scenario, then run both agents. `-base-path` or
`FAK_DEMO_BASE_PATH` mounts the browser demo behind a preserved HTTPS reverse-proxy path.

## What you see

```text
== guarddemo -selfcheck: replay each scenario through the kernel (browserless) ==
guard-redteam        WITHOUT fak: 4 breaches  WITH fak: 0 breaches  PASS
turntax-airline      WITHOUT fak: 2 breaches  WITH fak: 0 breaches  PASS
turntax-happy        WITHOUT fak: 0 breaches  WITH fak: 0 breaches  PASS
OK - all scenarios reproduced the documented safety-floor invariants.
```

The `-print` view renders a two-column diff. The browser uses the same `api/run` replay
surface, so a PASS in `-selfcheck` is a direct check on the data path the page drives.

## Scope

This demo shows the capability/safety floor on frozen tool-call traces. It does not claim
model quality, live upstream behavior, result containment, or detection strength. For the
broader claim ledger, see [`../../CLAIMS.md`](../../CLAIMS.md) and the public run guide in
[`../../docs/run-the-demos.md`](../../docs/run-the-demos.md).
