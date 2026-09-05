---
loop: goal
goal_slug: grafana-data-end-to-end-working-all-panels
witness: python3 tools/grafana_panel_audit.py --verify-all
budget: { max_iters: 20 }
---
# Objective
grafana data end to end working all panels. many show zeroes. use sub agents

# Non-Goals
- Modifying production core model engine weights or training
- Altering core frozen ABI (`internal/abi`)
- Adding external cloud dependencies outside the local Prometheus/Grafana stack

# Plan
- [ ] 1. Baseline audit: enumerate all 15 dashboards, parse every panel and query, evaluate against live Prometheus (localhost:9091) and Grafana (localhost:3000), classifying active, zero, and no-data panels.
- [ ] 2. Subagent delegation 1: Investigate and resolve zero/no-data in Fleet & Run Operations dashboards (`fak-fleet-overview`, `fak-fleet-session`, `fleet-bottleneck`).
- [ ] 3. Subagent delegation 2: Investigate and resolve zero/no-data in Gateway, Dogfood, Guard, and Cache dashboards (`fak-gateway-observability`, `fak-dogfood-slow-requests`, `fak-guard-adjudication`, `fak-startup-load`, `fak-cache-health`, `fak-cache-value-rollup`).
- [ ] 4. Subagent delegation 3: Investigate and resolve zero/no-data in Harness & Native Performance dashboards (`fak-harness-toolcall-fleet`, `fak-harness-toolcall-session`, `fak-native-*`).
- [ ] 5. End-to-end data pipeline drive: feed live traffic and telemetry across all subsystems to populate all metrics end-to-end.
- [ ] 6. Final verification witness: run `python3 tools/grafana_panel_audit.py --verify-all` ensuring all panels render live data or contracted unavailable states.

# Scratch / last-refusal
