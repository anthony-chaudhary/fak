package doomloop

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return string(raw)
}

// runDecisions synthesizes one worker's ledger for a single burning-flat run that
// climbs to peak streak P, with the correction at each streak derived from cfg the
// same way the live classifier would, optionally followed by a reset row (streak
// 0) that marks recovery. This lets the golden tests build episodes with an exact
// known outcome and peak.
func runDecisions(session string, base int64, peak int, cfg Config, reset bool) []Decision {
	var ds []Decision
	for s := 1; s <= peak; s++ {
		corr := CorrectObserve
		switch {
		case s >= cfg.EscalateWindows:
			corr = CorrectEscalate
		case s >= cfg.TripWindows:
			corr = CorrectNudge
		}
		ds = append(ds, Decision{
			UnixMillis: base + int64(s)*1000,
			Session:    session,
			Correction: string(corr),
			Streak:     s,
		})
	}
	if reset {
		ds = append(ds, Decision{
			UnixMillis: base + int64(peak+1)*1000,
			Session:    session,
			Correction: string(CorrectNone),
			Streak:     0,
		})
	}
	return ds
}

// TestCalibrateGoldenProposals is the end-to-end golden: a fixed ledger folds to
// exact counts and exact worst-first threshold proposals.
func TestCalibrateGoldenProposals(t *testing.T) {
	cfg := DefaultCalibrateConfig() // Base: Trip=3, Escalate=6; MinNudgeEpisodes=3
	base := cfg.Base

	var led []Decision
	add := func(ds []Decision) { led = append(led, ds...) }
	// 4 nudged episodes that recovered, peak streaks 3,3,4,4.
	add(runDecisions("nr1", 0, 3, base, true))
	add(runDecisions("nr2", 1_000_000, 3, base, true))
	add(runDecisions("nr3", 2_000_000, 4, base, true))
	add(runDecisions("nr4", 3_000_000, 4, base, true))
	// 1 nudged episode that escalated (peak 6).
	add(runDecisions("esc1", 4_000_000, 6, base, true))
	// 3 non-nudged stalls that self-recovered, peak streaks 1,1,1.
	add(runDecisions("sr1", 5_000_000, 1, base, true))
	add(runDecisions("sr2", 6_000_000, 1, base, true))
	add(runDecisions("sr3", 7_000_000, 1, base, true))

	rep := Calibrate(led, cfg)

	if rep.Verdict != CalibrationOK {
		t.Fatalf("verdict = %s, want OK (reason=%q)", rep.Verdict, rep.Reason)
	}
	if rep.NudgeEpisodes != 5 || rep.Recovered != 4 || rep.Escalated != 1 || rep.Ongoing != 0 {
		t.Fatalf("counts: nudge=%d recovered=%d escalated=%d ongoing=%d, want 5/4/1/0", rep.NudgeEpisodes, rep.Recovered, rep.Escalated, rep.Ongoing)
	}
	if rep.SelfRecovered != 3 {
		t.Fatalf("self-recovered = %d, want 3", rep.SelfRecovered)
	}
	if rep.RecoveryRate != 0.8 {
		t.Fatalf("recovery rate = %v, want 0.8", rep.RecoveryRate)
	}
	wantRecovery := Distribution{Count: 4, Min: 3, P50: 3, P90: 4, Max: 4}
	if rep.RecoveryStreak != wantRecovery {
		t.Fatalf("recovery streak dist = %+v, want %+v", rep.RecoveryStreak, wantRecovery)
	}
	wantSelf := Distribution{Count: 3, Min: 1, P50: 1, P90: 1, Max: 1}
	if rep.SelfRecoverPeak != wantSelf {
		t.Fatalf("self-recovery peak dist = %+v, want %+v", rep.SelfRecoverPeak, wantSelf)
	}

	if len(rep.Proposals) != 2 {
		t.Fatalf("proposals = %d, want 2: %+v", len(rep.Proposals), rep.Proposals)
	}
	// Worst-first: both deltas have magnitude 1, tie broken by name -> EscalateWindows first.
	esc := rep.Proposals[0]
	if esc.Name != "EscalateWindows" || esc.Current != 6 || esc.Proposed != 5 || esc.Delta != -1 {
		t.Fatalf("escalate proposal = %+v, want {EscalateWindows current=6 proposed=5 delta=-1}", esc)
	}
	trip := rep.Proposals[1]
	if trip.Name != "TripWindows" || trip.Current != 3 || trip.Proposed != 2 || trip.Delta != -1 {
		t.Fatalf("trip proposal = %+v, want {TripWindows current=3 proposed=2 delta=-1}", trip)
	}
	// The honest caveat must ride along in the evidence.
	if !strings.Contains(esc.Evidence, "censored") {
		t.Fatalf("escalate evidence should name the censoring caveat, got %q", esc.Evidence)
	}
}

// TestCalibrateInsufficientFloor: below the nudge-episode floor the fold refuses
// to propose (fail closed) but still reports the counts it observed.
func TestCalibrateInsufficientFloor(t *testing.T) {
	cfg := DefaultCalibrateConfig()
	base := cfg.Base
	// Only 2 nudge episodes < MinNudgeEpisodes(3).
	led := append(runDecisions("a", 0, 3, base, true), runDecisions("b", 100_000, 4, base, true)...)

	rep := Calibrate(led, cfg)
	if rep.Verdict != CalibrationInsufficient {
		t.Fatalf("verdict = %s, want INSUFFICIENT", rep.Verdict)
	}
	if len(rep.Proposals) != 0 {
		t.Fatalf("INSUFFICIENT must emit no proposals, got %+v", rep.Proposals)
	}
	if rep.NudgeEpisodes != 2 || rep.RecoveryStreak.Count != 2 {
		t.Fatalf("counts should still be reported: nudge=%d recoveryCount=%d, want 2/2", rep.NudgeEpisodes, rep.RecoveryStreak.Count)
	}
}

// TestCalibrateAllEscalateProposesEarlyEscalate: when every nudge escalates and
// none recover, the soft nudge is not working — propose escalating right after the
// trip (Trip+1), and rank that dial worst-first.
func TestCalibrateAllEscalateProposesEarlyEscalate(t *testing.T) {
	cfg := DefaultCalibrateConfig()
	base := cfg.Base
	var led []Decision
	for i, sess := range []string{"e1", "e2", "e3"} {
		led = append(led, runDecisions(sess, int64(i)*1_000_000, 6, base, true)...)
	}

	rep := Calibrate(led, cfg)
	if rep.Verdict != CalibrationOK {
		t.Fatalf("verdict = %s, want OK", rep.Verdict)
	}
	if rep.Escalated != 3 || rep.Recovered != 0 {
		t.Fatalf("escalated=%d recovered=%d, want 3/0", rep.Escalated, rep.Recovered)
	}
	esc := rep.Proposals[0]
	if esc.Name != "EscalateWindows" || esc.Proposed != 4 || esc.Delta != -2 {
		t.Fatalf("escalate proposal = %+v, want {EscalateWindows proposed=4 delta=-2}", esc)
	}
	trip := rep.Proposals[1]
	if trip.Name != "TripWindows" || trip.Proposed != 3 || trip.Delta != 0 {
		t.Fatalf("trip proposal = %+v, want held at 3 (no self-recovery evidence)", trip)
	}
}

// TestParseDecisions covers blank-line skipping, malformed-line errors, and the
// out-of-order rows the fold must sort by UnixMillis.
func TestParseDecisions(t *testing.T) {
	raw := "{\"session\":\"a\",\"burning_flat_streak\":0,\"correction\":\"NONE\",\"unix_millis\":400}\n" +
		"\n" +
		"   \n" +
		"{\"session\":\"a\",\"burning_flat_streak\":3,\"correction\":\"NUDGE\",\"unix_millis\":300}\n"
	ds, err := ParseDecisions(strings.NewReader(raw))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(ds) != 2 {
		t.Fatalf("parsed %d rows, want 2 (blank lines skipped)", len(ds))
	}
	if _, err := ParseDecisions(strings.NewReader("{not valid json")); err == nil {
		t.Fatal("a malformed ledger line must error, not silently drop")
	}
}

// TestCalibratePercentile pins the nearest-rank distribution summary the proposals
// rest on.
func TestCalibratePercentile(t *testing.T) {
	cases := []struct {
		xs   []int
		want Distribution
	}{
		{[]int{3, 3, 4, 4}, Distribution{Count: 4, Min: 3, P50: 3, P90: 4, Max: 4}},
		{[]int{1, 1, 1}, Distribution{Count: 3, Min: 1, P50: 1, P90: 1, Max: 1}},
		{[]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, Distribution{Count: 10, Min: 1, P50: 5, P90: 9, Max: 10}},
		{nil, Distribution{}},
	}
	for _, c := range cases {
		if got := distOf(c.xs); !reflect.DeepEqual(got, c.want) {
			t.Fatalf("distOf(%v) = %+v, want %+v", c.xs, got, c.want)
		}
	}
}

// TestCalibrateReaderRoundTrip: the reader convenience parses then folds to the
// same report the in-memory fold produces.
func TestCalibrateReaderRoundTrip(t *testing.T) {
	cfg := DefaultCalibrateConfig()
	led := runDecisions("only", 0, 6, cfg.Base, true)
	direct := Calibrate(led, cfg)

	var b strings.Builder
	for _, d := range led {
		b.WriteString(mustJSON(t, d))
		b.WriteByte('\n')
	}
	viaReader, err := CalibrateReader(strings.NewReader(b.String()), cfg)
	if err != nil {
		t.Fatalf("CalibrateReader: %v", err)
	}
	if !reflect.DeepEqual(direct, viaReader) {
		t.Fatalf("reader fold != direct fold:\n direct: %+v\n reader: %+v", direct, viaReader)
	}
}
