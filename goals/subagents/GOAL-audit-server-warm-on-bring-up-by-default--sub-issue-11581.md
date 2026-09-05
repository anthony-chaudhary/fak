---
parent_goal: goals/GOAL-audit-server-warm-on-bring-up-by-default.md
sub_step: issue_11581_gate_warmup_endpoints
issue: 11581
witness: "go test -v ./internal/gateway -run 'TestWarmup.*|TestInferenceGatedDuringWarmup'"
target_files:
  - internal/gateway/readiness_warmup.go
  - internal/gateway/http.go
  - internal/gateway/messages.go
  - internal/gateway/readiness_warmup_test.go
---
# Sub-Goal Objective
Implement Issue #11581: Gate incoming inference requests on warmup completion.

## Requirements
1. In `internal/gateway`, provide a method on `Server`:
   ```go
   // checkWarmupPending checks if the server warmup gate is still armed and incomplete.
   // If warmup is pending, it writes HTTP 503 (StatusServiceUnavailable) with a Retry-After: 1
   // header and a typed "warmup_pending" error, returning true.
   func (s *Server) checkWarmupPending(w http.ResponseWriter) bool
   ```
2. Integrate `checkWarmupPending(w)` at the entrypoint of inference handlers:
   - `handleChatCompletions` in `internal/gateway/http.go`
   - `handleAnthropicMessages` in `internal/gateway/messages.go`
   - and any other inference endpoints (`handleCompletions`, `handleResponses`, etc. in `internal/gateway/http.go`).
   Verify that if `checkWarmupPending` returns true, the handler returns immediately.
3. Write a reproduction & regression test `TestInferenceGatedDuringWarmup` in `internal/gateway/readiness_warmup_test.go`:
   - An armed warmup gate causes POST `/v1/chat/completions` and POST `/v1/messages` to return HTTP 503 with `"code":"warmup_pending"` and header `Retry-After: 1`.
   - Once `s.MarkWarmupComplete(...)` is called, POST `/v1/chat/completions` and POST `/v1/messages` are admitted (returning HTTP 200 with the mock planner).
4. Run `go test -v ./internal/gateway -run 'TestWarmup.*|TestInferenceGatedDuringWarmup'` to witness the fix.
