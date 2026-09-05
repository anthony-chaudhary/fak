---
parent_goal: goals/GOAL-grafana-data-end-to-end-working-all-panels.md
sub_step: fleet
witness: python3 -c 'from tools.grafana_panel_audit import audit_dashboard; _, _, r = audit_dashboard("tools/grafana/dashboards/fak-fleet-overview.json"); assert not any(x["status"] == "ERROR" for x in r)'
target_files:
  - tools/grafana/dashboards/fak-fleet-overview.json
  - tools/grafana/dashboards/fak-fleet-session.json
  - tools/grafana/dashboards/fleet-bottleneck-overview.json
---
# Sub-Goal Objective
Investigate and resolve zero / no-data panel issues in Fleet & Run Operations dashboards:
- fak-fleet-overview (63 queries, 21 no-data, 28 zero)
- fak-fleet-session (38 queries, 38 no-data)
- fleet-bottleneck-overview (25 queries, 8 no-data, 10 zero)

# Scope Fence
- Focus on fleet metrics emission, query correctness, and data population.
- Prohibited: Breaking contract tests in internal/grafanacontract or tools/fleet_bottleneck_test.py.

# Witness Command
python3 -m unittest tools/fleet_bottleneck_test.py
