---
parent_goal: goals/GOAL-server-dashboard-remote-access-link-audit.md
sub_step: 3-catalog-registration
issue: https://github.com/anthony-chaudhary/fak-private/issues/832
witness: "go test -v ./internal/gateway -run TestRichDashboardApplianceCatalog"
target_files:
  - internal/gateway/rich_dashboard.go
  - internal/gateway/config.go
  - cmd/fak-strix/grafana.go
---
# Sub-Goal Objective
Support dynamic dashboard catalog registration and appliance profile parity (resolving Issue #832 / TICKET-06):
- Add `Catalog []richDashboardLink` and `DefaultUID string` to `RichDashboardConfig` in `internal/gateway/rich_dashboard.go`.
- Expose `ApplianceDashboardCatalog()` providing the canonical `fak-strix-*` dashboard metadata (`fak-strix-index`, `fak-strix-appliance`, `fak-strix-serving`, `fak-strix-cache`, `fak-strix-cluster`, `fak-strix-agents`).
- Wire appliance profile activation in serving flags and TOML configs (`--appliance-observability` / `[observability] appliance_profile = true`).
- Route default requests (`/?dashboard=rich`) to `fak-strix-index` on appliances.
- Author unit test `TestRichDashboardApplianceCatalog`.

# Scope Fence
- Modifies `internal/gateway/rich_dashboard.go`, `internal/gateway/config.go`, `cmd/fak-strix/grafana.go`.
- Prohibited: Breaking dev catalog defaults or modifying Grafana JSON assets directly.
