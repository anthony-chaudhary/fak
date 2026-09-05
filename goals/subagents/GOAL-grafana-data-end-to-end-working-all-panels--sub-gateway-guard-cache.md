---
parent_goal: goals/GOAL-grafana-data-end-to-end-working-all-panels.md
sub_step: gateway-guard-cache
witness: python3 -c 'from tools.grafana_panel_audit import audit_dashboard; _, _, r = audit_dashboard("tools/grafana/dashboards/fak-gateway-observability.json"); assert not any(x["status"] == "ERROR" for x in r)'
target_files:
  - tools/grafana/dashboards/fak-gateway-observability.json
  - tools/grafana/dashboards/fak-dogfood-slow-requests.json
  - tools/grafana/dashboards/fak-guard-adjudication.json
  - tools/grafana/dashboards/fak-startup-load.json
  - tools/grafana/dashboards/fak-cache-health.json
  - tools/grafana/dashboards/fak-cache-value-rollup.json
---
# Sub-Goal Objective
Investigate and resolve zero / no-data panel issues in Gateway, Dogfood, Guard, Startup, and Cache dashboards:
- fak-gateway-observability
- fak-dogfood-slow-requests
- fak-guard-adjudication
- fak-startup-load
- fak-cache-health
- fak-cache-value-rollup

# Scope Fence
- Focus on metric feeds, traffic paths, and query definitions.
- Prohibited: Modifying frozen ABI or core security policies.

# Witness Command
go test -v ./internal/gateway/... -run TestMetrics
