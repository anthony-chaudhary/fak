---
parent_goal: goals/GOAL-audit-server-warm-on-bring-up-by-default.md
sub_step: 1_cli_bringup
witness: "code audit of cmd/fak/serve*.go, cmd/fak/up*.go, and server config defaults"
target_files:
  - cmd/fak/serve.go
  - cmd/fak/up.go
  - internal/config/
---
# Sub-Goal Objective
Audit CLI bringup verbs (`fak serve`, `fak up`) to determine whether server warmup is triggered by default on startup.
Identify:
1. Are there flags or config settings for warmup / prewarm / preheat? What are their default values?
2. During `fak serve` or `fak up` execution flow, does the bringup code invoke any engine/model warmup routine before declaring the server ready or opening the port/socket?
3. If warmup is enabled or disabled by default, identify exact file and line numbers proving it.
