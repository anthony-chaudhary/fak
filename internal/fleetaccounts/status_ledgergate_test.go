package fleetaccounts

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// status_ledgergate_test.go — #5439's witness that the probe-ledger rung is gated on what
// the resolved registry can DERIVE, not on FLEET_REG_DIR merely being set, and that a
// registry which can derive nothing publishes unknown-health rather than a proven-free seat.

// pinRegistryRungs aims every registry rung accountprobe.ResolveRegDir surveys at a
// controlled location, so a test's verdict is a function of what the test wrote and not of
// whatever registry the host running it happens to carry.
//
// This is the price #5439 makes explicit. shouldConsultProbeLedger is now a question about
// the FILESYSTEM, so setting FLEET_REG_DIR alone no longer pins the answer: leaving it empty
// falls through to the DISCOVERED rungs, and any developer machine that actually runs the
// prober has a ledger-bearing per-user Fleet registry sitting on exactly one of them. A test
// that skipped this would pass or fail on a fact about the operator's laptop.
//
// regDir is what FLEET_REG_DIR should name — pass "" to leave the env rung empty. Every
// other rung is aimed at a freshly created empty dir. The clone rung (tools/_registry,
// relative to the working directory) needs no pinning: `go test` runs in the package dir,
// which has no tools/ subtree.
func pinRegistryRungs(t *testing.T, regDir string) {
	t.Helper()
	empty := filepath.Join(t.TempDir(), "no-registry-here")
	if err := os.MkdirAll(empty, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FLEET_REG_DIR", regDir)
	t.Setenv("FLEET_STATE_DIR", "")
	t.Setenv("LOCALAPPDATA", empty)
	t.Setenv("TMP", empty)
	t.Setenv("TEMP", empty)
}

// TestLedgerRungRunsWhenBlocksDerivableWithoutRegDir is the #5439 acceptance line: the rung
// runs whenever blocks are DERIVABLE, regardless of whether FLEET_REG_DIR is set. The env var
// is empty here and the ledger lives where the prober actually writes it — the per-user Fleet
// registry — which is precisely the wiring the old env-var gate could not see. Under that
// gate the rung never ran, so a fresh OK probe was invisible to the roster and the carried
// throttle won no matter how correctly accountprobe resolved the dir.
func TestLedgerRungRunsWhenBlocksDerivableWithoutRegDir(t *testing.T) {
	userRoot := t.TempDir()
	pinRegistryRungs(t, "") // nothing derivable anywhere...
	t.Setenv("LOCALAPPDATA", userRoot)
	// ...except the per-user Fleet registry, which carries the ledger the prober writes.
	writeProbeLedger(t, filepath.Join(userRoot, "Fleet", "registry"),
		probeLine(t, ".claude-a", "OK", time.Now(), ""))

	reg := Registry{Throttle: map[string]any{
		".claude-a": map[string]any{"reset": futureResetStr(48 * time.Hour)},
	}}
	st := computeRuntimeStatus(".claude-a", "", reg)
	if !st.Available || st.Blocked || st.Throttled {
		t.Fatalf("a derivable registry's fresh ledger OK must clear the carried throttle even with"+
			" FLEET_REG_DIR unset, got %+v", st)
	}
	if st.StatusSource != "probe-ledger" {
		t.Fatalf("status_source = %q, want probe-ledger", st.StatusSource)
	}
}

// TestLedgerRungSkippedWhenRegDirCannotDeriveBlocks is the other half of the same acceptance
// line, and the ONE case whose verdict the change flips from true to false: FLEET_REG_DIR is
// set, so the old gate turned the rung on, but it names a dir carrying sessions.json with no
// probe ledger beside it. The rung must not run. Nothing is lost by refusing — every read
// under that dir returned nothing anyway — and the ledger that does exist elsewhere is NOT
// the one this process resolves to, so consulting it would be reading another host-local
// registry's answer to a question about this one.
func TestLedgerRungSkippedWhenRegDirCannotDeriveBlocks(t *testing.T) {
	elsewhere := t.TempDir()
	ledgerLess := t.TempDir()
	if err := os.WriteFile(filepath.Join(ledgerLess, "sessions.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
	writeProbeLedger(t, elsewhere, probeLine(t, ".claude-a", "OK", time.Now(), ""))
	pinRegistryRungs(t, ledgerLess)

	reg := Registry{Throttle: map[string]any{
		".claude-a": map[string]any{"reset": futureResetStr(48 * time.Hour)},
	}}
	st := computeRuntimeStatus(".claude-a", "", reg)
	if st.Available || !st.Blocked || !st.Throttled {
		t.Fatalf("a ledger-less FLEET_REG_DIR must leave the passive fold's carried throttle"+
			" standing, got %+v", st)
	}
	// A blocked seat keeps the confident source: "blocked" was derived from the registry's own
	// throttle row, so its provenance is not in doubt.
	if st.StatusSource != "registry" {
		t.Fatalf("status_source = %q, want registry", st.StatusSource)
	}
}

// TestUnderivableRegistryPublishesUnknownHealth is #5439's second acceptance line. A registry
// that cannot derive a block must not publish a seat as proven-free — but it must not strand
// it either. Both halves are asserted here on purpose: the seat stays OFFERED (available, not
// blocked), and only the CLAIM is weakened to registry-unknown. Converting absence into a
// block would be self-sealing — the roster routes the work that runs the prober, so a block
// imposed for want of a probe forbids the probe that would lift it.
func TestUnderivableRegistryPublishesUnknownHealth(t *testing.T) {
	ledgerLess := t.TempDir()
	if err := os.WriteFile(filepath.Join(ledgerLess, "sessions.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
	pinRegistryRungs(t, ledgerLess)

	reg := Registry{
		GeneratedUTC: time.Now().UTC().Format("2006-01-02T15:04:05Z"),
		Sessions:     []Session{{Account: ".claude-a", Project: "work", Disp: "LIVE"}},
	}
	st := computeRuntimeStatus(".claude-a", "", reg)
	if !st.Available || st.Blocked || st.Throttled {
		t.Fatalf("unknown health must not strand the seat — it stays offered, got %+v", st)
	}
	if st.StatusSource != "registry-unknown" {
		t.Fatalf("status_source = %q, want registry-unknown", st.StatusSource)
	}
	// The weakened claim must survive onto the published row, not stop at the fold.
	var acc Account
	applyStatus(&acc, st)
	if derefStr(acc.StatusSource) != "registry-unknown" {
		t.Fatalf("Account.status_source = %q, want registry-unknown", derefStr(acc.StatusSource))
	}
}

// TestDerivableRegistryKeepsRegistryStatusSource is the control that keeps the new state from
// being a blanket rename: the same roster read out of a registry that DOES carry a ledger
// still publishes the confident "registry", because there the absence of a block is a finding
// rather than a gap in the evidence.
func TestDerivableRegistryKeepsRegistryStatusSource(t *testing.T) {
	rd := t.TempDir()
	if err := os.WriteFile(filepath.Join(rd, "sessions.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
	writeProbeLedger(t, rd) // present but empty: a prober is wired, it has said nothing
	pinRegistryRungs(t, rd)

	reg := Registry{
		GeneratedUTC: time.Now().UTC().Format("2006-01-02T15:04:05Z"),
		Sessions:     []Session{{Account: ".claude-a", Project: "work", Disp: "LIVE"}},
	}
	st := computeRuntimeStatus(".claude-a", "", reg)
	if !st.Available || st.Blocked {
		t.Fatalf("a derivable registry with no block must publish an available seat, got %+v", st)
	}
	if st.StatusSource != "registry" {
		t.Fatalf("status_source = %q, want registry", st.StatusSource)
	}
}

// TestEmptyRegistryKeepsNoneStatusSource pins the boundary on the other side: an EMPTY
// registry already says "nothing was consulted" with status_source=none, and unknown-health
// must not overwrite that — "none" is the more precise statement of the two.
func TestEmptyRegistryKeepsNoneStatusSource(t *testing.T) {
	ledgerLess := t.TempDir()
	if err := os.WriteFile(filepath.Join(ledgerLess, "sessions.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
	pinRegistryRungs(t, ledgerLess)

	st := computeRuntimeStatus(".claude-a", "", Registry{})
	if st.StatusSource != "none" {
		t.Fatalf("status_source = %q, want none", st.StatusSource)
	}
}
