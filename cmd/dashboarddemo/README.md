# Live dashboard LCD demo

## Quickstart

One command proves the default lightweight dashboard and its lazy rich-dashboard
discovery surface without a key, network service, Docker, model, or GPU. Go 1.26+
is the only prerequisite; the repository toolchain setting can fetch it
automatically.

```bash
go run ./cmd/dashboarddemo -selfcheck
```

On a warm Go build cache the selfcheck completes in under 5 seconds; the first run
can take longer while Go fetches the declared toolchain and compiles the package.
The fixture and stdout receipt are deterministic and safe to rerun.

## What you see

A passing JSON receipt proves the real gateway handler returns HTTP 200 for `/`,
`/healthz`, and `/metrics`; the page refreshes every five seconds; and all nine
rich dashboards remain on demand. The exact stdout receipt is captured in
[`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md) and locked to the handler assertions by
`main_test.go`. Timestamped gateway diagnostics go to stderr and are not part of
that stable receipt.

## What this does not claim

The deterministic mock planner is intentional: this demo witnesses the dashboard
HTTP control plane, not model quality, browser rendering, authentication, or a
long-running production server. See the [gateway explainer](../../docs/explainers/gateway.md)
for the surrounding request path and deployment boundary.
