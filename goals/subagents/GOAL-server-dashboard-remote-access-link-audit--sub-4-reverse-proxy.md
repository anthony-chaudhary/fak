---
parent_goal: goals/GOAL-server-dashboard-remote-access-link-audit.md
sub_step: 4-reverse-proxy
issue: https://github.com/anthony-chaudhary/fak-private/issues/833
witness: "go test -v ./internal/gateway -run TestGrafanaReverseProxy"
target_files:
  - internal/gateway/gateway.go
  - internal/gateway/config.go
  - internal/gateway/rich_dashboard.go
  - internal/gateway/rich_dashboard_test.go
---
# Sub-Goal Objective
Implement single-port ingress reverse proxy for co-located Grafana under `/grafana/` (resolving Issue #833 / TICKET-07):
- Add `ProxyGrafana bool` and `ProxyGrafanaPrefix string` to `RichDashboardConfig`.
- Mount `httputil.ReverseProxy` under `/grafana/` forwarding to `http://127.0.0.1:3000/`.
- Strip prefix and set `X-Forwarded-Host`, `X-Forwarded-Proto`, and `X-Forwarded-Prefix`.
- When enabled, update `richDashboardDestination` to emit relative `/grafana/d/<uid>` URLs, eliminating cross-port redirects.
- Author unit test `TestGrafanaReverseProxy`.

# Scope Fence
- Modifies `internal/gateway/gateway.go`, `internal/gateway/config.go`, `internal/gateway/rich_dashboard.go`, `internal/gateway/rich_dashboard_test.go`.
- Prohibited: General-purpose third-party proxying or modifying core gateway inference paths.
