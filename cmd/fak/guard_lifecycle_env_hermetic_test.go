package main

import "os"

// guard_lifecycle_env_hermetic_test.go — keeps every lifecycle-signal test in this package
// hermetic against the environment of a LIVE `fak guard` session.
//
// The trap: `fak guard` exports the supervisor's lifecycle IPC endpoint
// (FAK_GUARD_LIFECYCLE_SOCKET/_TOKEN, guard_lifecycle_ipc.go) into the wrapped child's
// environment, so a `go test ./cmd/fak` run launched INSIDE a guard session inherits a
// pointer to that session's live signal socket. Both hook fetchers PREFER the IPC endpoint
// over an HTTP metrics URL and treat it as authoritative by design
// (fetchGuardStopHookSignalsPreferred, fetchGuardPreCompactSignalsPreferred) — so every
// runGuardStopHook / guard-precompact test silently read the WRAPPING session's gauges
// (deny_all_consecutive=0, i.e. "clean stop") instead of the --metrics-url httptest fixture it
// had just stood up. Twelve stop-hook tests failed that way — an inherited-environment defect
// that looks exactly like a precedence regression in the guidance chain, because the deny-all
// fixture never reaches the decision and the clean-stop rungs (task handoff, shadow-allow) fire
// in its place.
//
// Scrubbing the pair here rather than in each test keeps a future test from re-inheriting it.
// init runs before testing's generated main, so it lands before TestMain and any test body. The
// tests that exercise the IPC preference itself set both vars explicitly with t.Setenv
// (guard_lifecycle_ipc_test.go, guard_precompact_test.go), so they are unaffected; nothing in
// this binary READS the pair except the two fetchers above, and the guard parent writes its own
// values when it starts a lifecycle server.
func init() {
	_ = os.Unsetenv(guardLifecycleSocketEnv)
	_ = os.Unsetenv(guardLifecycleTokenEnv)
}
