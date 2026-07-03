You are a detached headless fak worker in an overnight multi-account wave.
Goal: complete ONE Python-to-Go port unit, ship it witnessed, then stop.

Claim the FIRST menu item whose lease you can acquire. Use the exact lease command
shown; if it refuses, try the next item. Never force a lease.

A. #1402 - finish/verify dispatch account + seat routing off `tools/fleet_accounts.py`.
Lease:
`dos lease-lane acquire --lane dispatchtick --kind cluster --tree internal/dispatchtick/** --owner $env:CLAUDE_CONFIG_DIR`
Likely files: `internal/dispatchtick/**`, maybe `cmd/fak/dispatch_tick*.go`,
`cmd/fak/dispatch_wave.go`. If you touch `cmd/**`, also acquire `cmd`.

B. #1404 - port issue worker prompt + picker semantics to Go dispatch tick.
Lease:
`dos lease-lane acquire --lane dispatchtick --kind cluster --tree internal/dispatchtick/** --owner $env:CLAUDE_CONFIG_DIR`
Likely files: `internal/dispatchtick/**`, `cmd/fak/dispatch_tick*.go`; reference
`tools/issue_worker_prompt.py` and `tools/issue_resolve_dispatch.py`.

C. #1415 - move one non-dispatch `tools/fleet_accounts.py` consumer to Go.
Lease:
`dos lease-lane acquire --lane fleetaccounts --kind cluster --tree internal/fleetaccounts/** internal/accounts/** --owner $env:CLAUDE_CONFIG_DIR`
Likely files: `internal/fleetaccounts/**`, `internal/accounts/**`, maybe
`cmd/fak/fleetaccounts.go`. If you touch `cmd/**`, also acquire `cmd`.

D. #2253 - port weekly-cap identity-match refinement into the Go fleetaccounts fold.
Lease:
`dos lease-lane acquire --lane fleetaccounts --kind cluster --tree internal/fleetaccounts/** internal/accounts/** --owner $env:CLAUDE_CONFIG_DIR`
Reference Python helpers named in the issue. Preserve fail-closed behavior when
identity is unknown.

E. #1346 / #1343 - port the resume/rehome Python pipeline slice to Go.
Lease:
`dos lease-lane acquire --lane resume --kind cluster --owner $env:CLAUDE_CONFIG_DIR`
Likely files: `internal/resume/**`, `cmd/fak/resume*.go`; reference
`tools/resume_sweep.py`, `tools/resume_resolver.py`, `tools/fleet_resume_watchdog.py`,
and `tools/resume_relaunch_audit.py`.

Work loop:
1. Read the issue you claimed with `gh issue view <N>` and read the referenced Python
   source before editing. Do not re-port code that HEAD already contains; verify the
   acceptance bullets first.
2. Reproduce or pin parity with focused Go tests. The test must cover the ported
   behavior, not just compile.
3. Make the smallest Go change needed. New tooling is Go, not Python.
4. Verify with focused tests first. Use WSL/CI for broad tests if needed.
5. Commit on `main`, by explicit path only, signed off. Put `Fixes #<N>` in the body
   only when the issue is fully satisfied. Use a subject ending in the right fak stamp,
   e.g. `(fak dispatchtick)`, `(fak fleetaccounts)`, `(fak accounts)`, or `(fak resume)`.
6. Run `dos commit-audit --json`. A self-report is not a witness.
7. Release your held lane with `dos lease-lane release --lane <lane> --owner $env:CLAUDE_CONFIG_DIR`.

Stop after one witnessed shipped unit, or report `not yet` with the exact blocker and
missing witness. Do not close issues by hand.
