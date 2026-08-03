package fleetaccounts

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// status_seatcoverage_test.go — #5391's witness that a seat the prober has not spoken about
// lately is published as unknown-health rather than as a proven-free one, EVEN THOUGH the
// registry it is published out of can derive blocks perfectly well for other accounts.
//
// #5439 already made a registry that can derive nothing say so (status_ledgergate_test.go).
// That grade is per-REGISTRY, and the host that filed #5391 shows why it is not sufficient:
// probe_ledger.jsonl there was present, derivable and busy — opencode-* rows current to the
// minute — while several claude seats' newest rows were 8-9 days old. Every registry-level
// question answered "yes, blocks are derivable here", and none of that was evidence about
// those seats. They read available, the allocator saw full headroom on a seat whose org had
// disabled Claude access (a 403 burns no quota), and it preferred the one seat that could not
// work.
//
// Both halves are asserted on every case on purpose: the weakened claim must never become a
// block. Converting absence into blocked is self-sealing — the roster routes the work that
// runs the prober — and #5391 files itself as a coverage gap for exactly that reason.

// seatCoverageFixture pins every registry rung at one controlled dir, writes the ledger lines
// there, and returns a non-empty Registry carrying one live session for account. The registry
// must be non-empty or the fold reports status_source "none" (nothing was consulted), which is
// a different and more precise statement than unknown-health.
func seatCoverageFixture(t *testing.T, account string, lines ...string) Registry {
	t.Helper()
	rd := t.TempDir()
	writeProbeLedger(t, rd, lines...)
	// sessions.json beside the ledger, so the dir grades as a real registry rung.
	if err := os.WriteFile(filepath.Join(rd, "sessions.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
	pinRegistryRungs(t, rd)
	return Registry{
		GeneratedUTC: time.Now().UTC().Format("2006-01-02T15:04:05Z"),
		Sessions:     []Session{{Account: account, Project: "work", Disp: "LIVE"}},
	}
}

// TestBusyLedgerWithNoRowForSeatPublishesUnknownHealth is #5391's first acceptance line —
// "never probed" and "probed OK" must be distinguishable. The prober is demonstrably running
// (it wrote a row minutes ago) and has simply never named this seat, so the fold has no
// evidence about it and must not claim any.
func TestBusyLedgerWithNoRowForSeatPublishesUnknownHealth(t *testing.T) {
	reg := seatCoverageFixture(t, ".claude-a",
		probeLine(t, "opencode-fixture", "OK", time.Now().Add(-2*time.Minute), ""))

	st := computeRuntimeStatus(".claude-a", "", reg)
	if !st.Available || st.Blocked || st.Throttled {
		t.Fatalf("an unmeasured seat must stay offered — unknown health is not a block, got %+v", st)
	}
	if st.StatusSource != "registry-unknown" {
		t.Fatalf("status_source = %q, want registry-unknown for a seat a busy ledger never mentions",
			st.StatusSource)
	}
	// The weakened claim must survive onto the published row, not stop at the fold.
	var acc Account
	applyStatus(&acc, st)
	if derefStr(acc.StatusSource) != "registry-unknown" {
		t.Fatalf("Account.status_source = %q, want registry-unknown", derefStr(acc.StatusSource))
	}
}

// TestEightDayOldSeatRowPublishesUnknownHealth is the observed shape verbatim: one account
// current to the minute, this one last heard from eight days ago. A row that old is not
// evidence about now, so the seat is unmeasured exactly as if it had never been probed — the
// distinction the ledger's presence would otherwise paper over.
func TestEightDayOldSeatRowPublishesUnknownHealth(t *testing.T) {
	reg := seatCoverageFixture(t, ".claude-a",
		probeLine(t, ".claude-a", "OK", time.Now().Add(-8*24*time.Hour), ""),
		probeLine(t, "opencode-fixture", "OK", time.Now().Add(-2*time.Minute), ""))

	st := computeRuntimeStatus(".claude-a", "", reg)
	if !st.Available || st.Blocked || st.Throttled {
		t.Fatalf("a stale-probe seat must stay offered, got %+v", st)
	}
	if st.StatusSource != "registry-unknown" {
		t.Fatalf("status_source = %q, want registry-unknown for an 8-day-old probe row", st.StatusSource)
	}
}

// TestSeatRowInsideCoverageBudgetKeepsRegistryStatusSource is the control that keeps the new
// trigger from being a blanket rename. Two windows are in play and they are not the same
// window: the fresh-VERDICT window (ProbeLedgerFreshMin, 20m) decides whether a probe result
// still overrides the registry, and the COVERAGE budget (accountprobe.SeatCoverageMaxAgeMin,
// 24h) decides whether the seat has been measured at all. A two-hour-old OK is past the first
// and well inside the second: the fold correctly falls through to the registry rung, and the
// claim it publishes there is a confident one because the seat WAS measured today.
func TestSeatRowInsideCoverageBudgetKeepsRegistryStatusSource(t *testing.T) {
	reg := seatCoverageFixture(t, ".claude-a",
		probeLine(t, ".claude-a", "OK", time.Now().Add(-2*time.Hour), ""))

	st := computeRuntimeStatus(".claude-a", "", reg)
	if !st.Available || st.Blocked {
		t.Fatalf("a measured, unblocked seat must be published available, got %+v", st)
	}
	if st.StatusSource != "registry" {
		t.Fatalf("status_source = %q, want registry for a seat probed inside the coverage budget",
			st.StatusSource)
	}
}

// TestBusyLedgerBlockedSeatKeepsRegistryStatusSource pins the ordering: a BLOCKED seat keeps
// the confident source even with no probe row of its own. "Blocked" is a positive derivation
// from the registry's own throttle row, not a statement about probe evidence, so its
// provenance is not in doubt and unknown-health has nothing to add.
func TestBusyLedgerBlockedSeatKeepsRegistryStatusSource(t *testing.T) {
	reg := seatCoverageFixture(t, ".claude-a",
		probeLine(t, "opencode-fixture", "OK", time.Now().Add(-2*time.Minute), ""))
	reg.Throttle = map[string]any{
		".claude-a": map[string]any{"reset": futureResetStr(48 * time.Hour)},
	}

	st := computeRuntimeStatus(".claude-a", "", reg)
	if st.Available || !st.Blocked || !st.Throttled {
		t.Fatalf("the carried throttle must still block the seat, got %+v", st)
	}
	if st.StatusSource != "registry" {
		t.Fatalf("status_source = %q, want registry for a blocked seat", st.StatusSource)
	}
}

// TestEmptyLedgerKeepsRegistryStatusSource is the boundary #5439 drew and this change does not
// move: a ledger present but EMPTY leaves every seat unmeasured, yet the registry-level grade
// already describes that host and a seat-level downgrade would only restate it. The trigger
// added here fires on a ledger that demonstrably recorded rows for someone else — the case
// where the registry-level answer is affirmatively misleading — and nowhere else. Without that
// precondition this case would flip, so the assertion is load-bearing for the precondition and
// not merely a restatement of TestDerivableRegistryKeepsRegistryStatusSource.
func TestEmptyLedgerKeepsRegistryStatusSource(t *testing.T) {
	reg := seatCoverageFixture(t, ".claude-a") // ledger present, zero rows

	st := computeRuntimeStatus(".claude-a", "", reg)
	if st.StatusSource != "registry" {
		t.Fatalf("status_source = %q, want registry — an empty ledger is #5439's case, not #5391's",
			st.StatusSource)
	}
}
