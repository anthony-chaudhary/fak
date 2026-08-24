# Live dashboard dogfood — 2026-08-23

## Verdict

The default `fak serve` dashboard worked from a clean committed Windows build, and the rich dashboard route reused an already-running Grafana stack without changing or stopping it. The scoped dashboard suite also passed under the WSL race detector.

## Boundaries

- Source: clean archive of trunk commit `8aa17064a8` (`internal/gateway r15+g8aa17064`).
- Host: Windows 11 control point with Docker Desktop 28.3.3.
- Gateway: `fak serve --addr 127.0.0.1:8080`, deterministic mock planner, no model/GPU required for this control-plane witness.
- Rich stack: pre-existing bundled Compose project at `http://localhost:3000`, Grafana 11.5.2.
- Dashboard UID exercised first: `fak-gateway-observability`; reuse exercised with `fak-cache-health`.

## Captured run

1. Built `fak-dogfood.exe` from the clean archive and started it on `127.0.0.1:8080`.
2. `GET /` returned HTTP 200 and rendered `Rich dashboards` without any click-time setup.
3. Before activation, the `grafana` Compose project contained the same three container IDs later observed: `fleet-grafana`, `fleet-prometheus`, and `fleet-alertmanager`.
4. `GET /?dashboard=rich&uid=fak-gateway-observability` returned HTTP 303 in 133 ms with `Location: http://localhost:3000/d/fak-gateway-observability`.
5. Grafana `/api/health` returned `database=ok`, version 11.5.2.
6. A second dashboard selection returned HTTP 303 to `http://localhost:3000/d/fak-cache-health`.
7. Container IDs and states were unchanged after both clicks.
8. After gateway shutdown, all three pre-existing containers remained running with the same IDs and states.

The first live attempt exposed an ownership defect: an unset `FAK_GRAFANA_URL` always ran Compose and then marked the stack owned, even when Grafana was already healthy. Issue #8704 captured it; commit `8aa17064a8` now probes the canonical bundled endpoint first, reuses it as unowned when ready, and retains startup/owned teardown when it is absent.

## Cross-platform witness

From WSL on the committed gateway change:

```text
$ go test -race ./internal/gateway -run 'RichDashboard|GatewayHomepage' -count=1
ok github.com/anthony-chaudhary/fak/internal/gateway 1.195s
```

That run exposed a stale test-helper engine ID before it could become green. Issue #8706 captured the defect; commit `650a68b369` restored the registered deterministic mock engine and the same WSL race command then passed.

## Defects filed and closed

- #8704 — preserve a pre-existing Grafana stack; fixed by `internal/gateway r15+g8aa17064`.
- #8706 — use the registered engine in the rich-dashboard helper; fixed by `internal/gateway r16+g650a68b3`.

No other defect surfaced in this bounded run.
