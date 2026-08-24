# Live dashboard LCD demo

One command proves the default lightweight dashboard and its lazy rich-dashboard discovery surface without a key, network service, Docker, model, or GPU:

```bash
go run ./cmd/dashboarddemo -selfcheck
```

A passing JSON receipt proves the real gateway handler returns HTTP 200 for `/`, `/healthz`, and `/metrics`; the page refreshes every five seconds; and all nine rich dashboards remain on demand. The deterministic mock planner is intentional: this demo witnesses the dashboard control plane, not model quality.
