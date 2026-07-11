package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// writeGovernorLedger writes the given JSONL rows to a temp resume_ledger.jsonl and
// returns its path.
func writeGovernorLedger(t *testing.T, rows string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "resume_ledger.jsonl")
	if err := os.WriteFile(p, []byte(rows), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// Witness: a spawn the broker DENIED must never count as a launch — not in the
// trailing-24h stats and not in the spacing-floor snapshot. One misclassified denial
// sets LastLaunchUnix=now and trips LAUNCH_SPACING_FLOOR for every following session
// in the tick (the 7-day backlog stall of 2026-07).
func TestBrokerDeniedIsNotALaunch(t *testing.T) {
	now := time.Now().UTC()
	ts := now.Format("2006-01-02T15:04:05Z")
	p := writeGovernorLedger(t, `{"ts":"`+ts+`","session":"s1","phase":"broker_denied","reason":"policy"}`+"\n")

	st := scanGovernorLedgerStats(p, now)
	if st.Launched24h != 0 {
		t.Errorf("broker_denied counted as a launch: Launched24h=%d, want 0", st.Launched24h)
	}
	if st.LastLaunchUnix != 0 {
		t.Errorf("broker_denied bumped LastLaunchUnix=%d, want 0 (poisons spacing floor)", st.LastLaunchUnix)
	}
	// The gate path itself: foldSourceSnapshot consumes scanLaunchLedger, so a denial
	// must contribute neither a window timestamp nor a last-launch.
	if times, last := scanLaunchLedger(p); len(times) != 0 || last != 0 {
		t.Errorf("broker_denied counted as launch pressure: times=%v last=%d, want none", times, last)
	}
}

// Witness: the full phase vocabulary the watchdog writes to resume_ledger.jsonl
// (resume_watchdog_cli.go: settled/deferred/broker_denied/launched) plus the legacy
// pre-phase schema. Only "launched" and the phase-less legacy rows are launches.
func TestGovernorLedgerPhaseVocabulary(t *testing.T) {
	now := time.Now().UTC()
	recent := now.Add(-2 * time.Hour).Format("2006-01-02T15:04:05Z")
	rows := `{"ts":"` + recent + `","session":"a","phase":"launched","attempt":1}
{"ts":"` + recent + `","session":"b","account":".claude-x","pid":6048,"rehomed":true,"cause":"STOPPED_LIMIT"}
{"ts":"` + recent + `","session":"c","phase":"broker_denied","cause":"STOPPED_APIERR","reason":"policy"}
{"ts":"` + recent + `","session":"d","phase":"deferred","cause":"source_concurrency_gate"}
{"ts":"` + recent + `","session":"e","phase":"settled","action":"consolidate-auth-plan-row"}
`
	p := writeGovernorLedger(t, rows)

	st := scanGovernorLedgerStats(p, now)
	if st.Launched24h != 2 { // "launched" + the legacy phase-less row, nothing else
		t.Errorf("Launched24h = %d, want 2 (launched + legacy phase-less)", st.Launched24h)
	}
	if st.Deferred24h != 1 {
		t.Errorf("Deferred24h = %d, want 1", st.Deferred24h)
	}
	if st.LastLaunchUnix == 0 {
		t.Error("LastLaunchUnix not set by the real launches")
	}
	if times, last := scanLaunchLedger(p); len(times) != 2 || last == 0 {
		t.Errorf("scanLaunchLedger: times=%v last=%d, want exactly the 2 real launches", times, last)
	}
}

// Witness: a legacy row with NO phase field (the pre-phase schema — 114 such rows in
// the production ledger, carrying pid/rehomed/cause) is a genuine launch and must keep
// counting, or historical launch accounting silently zeroes out.
func TestLegacyPhaselessRowIsALaunch(t *testing.T) {
	now := time.Now().UTC()
	ts := now.Format("2006-01-02T15:04:05Z")
	p := writeGovernorLedger(t, `{"ts":"`+ts+`","session":"7ef0a89e","account":".claude-x","pid":6048,"cause":"STOPPED_LIMIT"}`+"\n")

	st := scanGovernorLedgerStats(p, now)
	if st.Launched24h != 1 {
		t.Errorf("legacy phase-less row: Launched24h=%d, want 1", st.Launched24h)
	}
	if st.LastLaunchUnix == 0 {
		t.Error("legacy phase-less row must set LastLaunchUnix")
	}
}

// Witness: the reader-side vocabulary — every non-launch phase any writer appends to a
// ledger a launch scanner might read is denylisted, and the launch phases are not.
func TestIsNonLaunchPhaseVocabulary(t *testing.T) {
	for _, ph := range []string{
		"broker_denied", "deferred", "considered", "skipped", "gate_fail_open",
		"queued", "detected", "status", "tick", "snapshot", "progress",
		"settled", "operator_settled", "consolidated",
	} {
		if !isNonLaunchPhase(ph) {
			t.Errorf("isNonLaunchPhase(%q) = false, want true", ph)
		}
	}
	for _, ph := range []string{"launched", ""} {
		if isNonLaunchPhase(ph) {
			t.Errorf("isNonLaunchPhase(%q) = true, want false (a real launch)", ph)
		}
	}
}
