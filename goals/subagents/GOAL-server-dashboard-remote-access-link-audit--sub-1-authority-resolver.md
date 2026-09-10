---
parent_goal: goals/GOAL-server-dashboard-remote-access-link-audit.md
sub_step: 1-authority-resolver
issue: https://github.com/anthony-chaudhary/fak-private/issues/830
witness: "go test -v ./internal/gateway -run TestRichDashboardClientBaseURL"
target_files:
  - internal/gateway/rich_dashboard.go
  - internal/gateway/rich_dashboard_test.go
---
# Sub-Goal Objective
Implement dynamic client base URL and authority resolution in `internal/gateway/rich_dashboard.go` (resolving Issue #830 / TICKET-04):
- Eliminate hardcoded `http://localhost:3000` redirects when accessed via remote LAN IP (e.g. `192.168.1.200:8080`), mDNS (`strix-halo-fak.local:8080`), or Tailscale.
- Implement `clientBaseURL(r *http.Request)` mapping loopback Grafana hosts to the client's dialed host while retaining Grafana's port (`:3000`) and request scheme (`http` / `https`).
- Preserve explicit external `FAK_GRAFANA_URL` overrides without modification.
- Author unit tests in `internal/gateway/rich_dashboard_test.go` asserting correct redirect targets across all 6 access contexts.

# Scope Fence
- Modifies `internal/gateway/rich_dashboard.go` and `internal/gateway/rich_dashboard_test.go`.
- Prohibited: Modifying frozen ABI or core security policies.
