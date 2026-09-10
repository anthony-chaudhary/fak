---
parent_goal: goals/GOAL-server-dashboard-remote-access-link-audit.md
sub_step: 2-key-retention
issue: https://github.com/anthony-chaudhary/fak-private/issues/831
witness: "go test -v ./internal/gateway -run TestRichDashboardKeyPreservation"
target_files:
  - internal/gateway/home.go
  - internal/gateway/rich_dashboard.go
  - internal/gateway/rich_dashboard_test.go
---
# Sub-Goal Objective
Implement key retention and remote navigation hints in `internal/gateway/home.go` and `internal/gateway/rich_dashboard.go` (resolving Issue #831 / TICKET-05):
- Ensure `richDashboardPage` return links (`<a href="/">`) preserve `?key=<KEY>` when present on the incoming request, preventing operator lockouts on WAN/firewalled networks.
- Render clear remediation guidance (`ssh -L 8080:localhost:8080 -L 3000:localhost:3000 <host>`) on dormant or unavailable dashboard error states.
- Author unit test `TestRichDashboardKeyPreservation` asserting key preservation in HTML templates.

# Scope Fence
- Modifies `internal/gateway/home.go`, `internal/gateway/rich_dashboard.go`, `internal/gateway/rich_dashboard_test.go`.
- Prohibited: Changing auth algorithms or security posture in `internal/gateway/auth_gate.go`.
