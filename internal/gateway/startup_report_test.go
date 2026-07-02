package gateway

import (
	"testing"
	"time"
)

// TestStartupReportRoundTripsThroughDebugVars is the witness for the `fak info --startup`
// read path: a report the host records at boot comes back verbatim on /debug/vars, an
// unset report stays an EMPTY (omitted) field rather than a fabricated blank page, and
// clearing restores the omitted state.
func TestStartupReportRoundTripsThroughDebugVars(t *testing.T) {
	srv := newTestServer(t)
	if got := srv.debugVars(time.Now()).StartupReport; got != "" {
		t.Fatalf("startup_report before SetStartupReport = %q, want empty (omitted)", got)
	}

	const report = "fak guard 9.9.9 — kernel-adjudicated: claude\n  floor      : built-in guard floor\n"
	srv.SetStartupReport(report)
	if got := srv.debugVars(time.Now()).StartupReport; got != report {
		t.Fatalf("startup_report = %q, want the recorded report verbatim", got)
	}

	srv.SetStartupReport("")
	if got := srv.debugVars(time.Now()).StartupReport; got != "" {
		t.Fatalf("startup_report after clear = %q, want empty", got)
	}
}

// TestStartupReportSafeOnNilServer pins the nil-Server contract shared by the other
// Set* seams (SetModelLoadProfile): recording or reading on nil must not panic.
func TestStartupReportSafeOnNilServer(t *testing.T) {
	var srv *Server
	srv.SetStartupReport("x")
	if got := srv.startupReportText(); got != "" {
		t.Fatalf("nil Server startupReportText = %q, want empty", got)
	}
}
