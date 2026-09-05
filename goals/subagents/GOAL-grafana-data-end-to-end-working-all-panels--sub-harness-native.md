---
parent_goal: goals/GOAL-grafana-data-end-to-end-working-all-panels.md
sub_step: harness-native
witness: python3 -c 'from tools.grafana_panel_audit import audit_dashboard; _, _, r = audit_dashboard("tools/grafana/dashboards/fak-native-backends.json"); assert not any(x["status"] == "ERROR" for x in r)'
target_files:
  - tools/grafana/dashboards/fak-harness-toolcall-fleet.json
  - tools/grafana/dashboards/fak-harness-toolcall-session.json
  - tools/grafana/dashboards/fak-native-artifacts.json
  - tools/grafana/dashboards/fak-native-backends.json
  - tools/grafana/dashboards/fak-native-kernel-performance.json
  - tools/grafana/dashboards/fak-native-slo.json
---
# Sub-Goal Objective
Investigate and resolve zero / no-data panel issues in Harness & Native Performance dashboards:
- fak-harness-toolcall-fleet
- fak-harness-toolcall-session
- fak-native-artifacts
- fak-native-backends
- fak-native-kernel-performance
- fak-native-slo

# Scope Fence
- Work in dashboard queries and telemetry generators.
- Prohibited: Changing frozen native contracts or breaking nativeperfcoverage invariants.

# Witness Command
go test -v ./internal/nativeperfcoverage/...
