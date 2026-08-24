package main

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resume"
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

// #8722: releasing a stale unproven once-latch is a decision, not a spawn. The
// watchdog can write one revived row on every tick while the session remains queued;
// those repeated rows must never renew LastLaunchUnix and indefinitely trip the
// source governor's launch-spacing floor. The 05:43/05:45 ticks mirror the live
// recurrence that exposed the defect.
func TestRevivedRowsDoNotRenewLaunchSpacingAcrossTicks(t *testing.T) {
	launchAt := time.Date(2026, 8, 24, 5, 40, 0, 0, time.UTC)
	p := writeGovernorLedger(t, fmt.Sprintf(
		`{"ts":"%s","session":"launched-first","phase":"launched"}`+"\n",
		launchAt.Format(time.RFC3339),
	))
	policy := resume.SourcePolicy{MinLaunchSpacingSeconds: 8}

	for _, tick := range []time.Time{
		time.Date(2026, 8, 24, 5, 43, 0, 0, time.UTC),
		time.Date(2026, 8, 24, 5, 45, 0, 0, time.UTC),
	} {
		f, err := os.OpenFile(p, os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			t.Fatal(err)
		}
		_, writeErr := fmt.Fprintf(f,
			`{"ts":"%s","session":"stale-latch","phase":"revived"}`+"\n",
			tick.Format(time.RFC3339),
		)
		closeErr := f.Close()
		if writeErr != nil {
			t.Fatal(writeErr)
		}
		if closeErr != nil {
			t.Fatal(closeErr)
		}

		times, last := scanLaunchLedger(p)
		if len(times) != 1 || last != launchAt.Unix() {
			t.Fatalf("tick %s: revived row counted as launch pressure: times=%v last=%d, want [%d] / %d",
				tick.Format("15:04"), times, last, launchAt.Unix(), launchAt.Unix())
		}
		st := scanGovernorLedgerStats(p, tick)
		if st.Launched24h != 1 || st.LastLaunchUnix != launchAt.Unix() {
			t.Fatalf("tick %s: revived row counted in governor stats: %+v, want one launch at %d",
				tick.Format("15:04"), st, launchAt.Unix())
		}
		d := resume.AdmitSource(resume.SourceSnapshot{
			LaunchUnixTimes: times,
			LastLaunchUnix:  last,
		}, policy, tick)
		if !d.Admit {
			t.Fatalf("tick %s: revived decision renewed launch spacing: %+v", tick.Format("15:04"), d)
		}
	}
}

// The #8722 fix must not weaken the real pressure signal: a fired launched row
// inside the spacing floor still defers the next queued session.
func TestLaunchedRowStillEnforcesLaunchSpacing(t *testing.T) {
	launchAt := time.Date(2026, 8, 24, 5, 45, 0, 0, time.UTC)
	p := writeGovernorLedger(t, fmt.Sprintf(
		`{"ts":"%s","session":"real-launch","phase":"launched"}`+"\n",
		launchAt.Format(time.RFC3339),
	))
	times, last := scanLaunchLedger(p)
	d := resume.AdmitSource(resume.SourceSnapshot{
		LaunchUnixTimes: times,
		LastLaunchUnix:  last,
	}, resume.SourcePolicy{MinLaunchSpacingSeconds: 8}, launchAt.Add(5*time.Second))
	if d.Admit || d.Reason != resume.ReasonLaunchSpacing {
		t.Fatalf("real launched row lost spacing pressure: %+v", d)
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
		"queued", "detected", "status", "tick", "snapshot", "progress", "decision",
		"settled", "operator_settled", "consolidated", "revived",
		"some_novel_bookkeeping_token",
	} {
		if !isNonLaunchPhase(ph) {
			t.Errorf("isNonLaunchPhase(%q) = false, want true", ph)
		}
	}
	for _, ph := range []string{"launched", "resumed", ""} {
		if isNonLaunchPhase(ph) {
			t.Errorf("isNonLaunchPhase(%q) = true, want false (a real launch)", ph)
		}
	}
}
