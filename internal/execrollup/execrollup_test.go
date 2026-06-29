package execrollup

import (
	"encoding/json"
	"strings"
	"testing"
)

// jmap unmarshals a JSON literal into the map[string]any shape the live
// collectors hand the fold — so the tests exercise the exact coercion path.
func jmap(t *testing.T, s string) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		t.Fatalf("bad test JSON: %v", err)
	}
	return m
}

// A clean, fully-measured fleet: every plane reports, nothing deviates → GREEN
// with an empty attention list. This is the signal-to-noise floor: no noise in,
// no lines out.
func TestFoldClean(t *testing.T) {
	in := Inputs{
		Dispatch: PlaneInput{Payload: jmap(t, `{
			"closure": {"na": false, "closure_rate": 0.92, "counts": {"TRUE_RESOLVED": 920, "CLAIMED_CLOSED": 80}, "open_witnessed_closable": 0},
			"throughput": {"na": false, "verdict": "ON_TARGET", "completed_rate_per_hour": 12.0, "target_per_hour": 10.0, "primary_window_hours": 6},
			"backend_health": {"dead_count": 0},
			"workers": {"silent_count": 0},
			"backlog": {"open_issues": 40}
		}`)},
		Loops:   PlaneInput{Payload: jmap(t, `{"rollup": {"loops": 8, "live": 8, "dark": 0}}`)},
		Cadence: PlaneInput{Payload: jmap(t, `{"scores": {"debt": 5, "trend_direction": "flat"}, "work": {"commits": 100, "ships": 90, "window_days": 7}}`)},
		Fleet:   PlaneInput{Payload: jmap(t, `{"total": 3, "reachable": 3, "attention": [{"level": "ok", "title": "all good"}]}`)},
	}
	r := Fold(in)
	if r.Verdict != VerdictGreen {
		t.Fatalf("verdict = %s, want GREEN", r.Verdict)
	}
	if !r.OK {
		t.Fatal("OK = false, want true on a clean fleet")
	}
	if len(r.Attention) != 0 {
		t.Fatalf("attention = %d items, want 0 (an OK fleet line is noise)", len(r.Attention))
	}
	if r.Unmeasured != 0 {
		t.Fatalf("unmeasured = %d, want 0", r.Unmeasured)
	}
	// S/N marquee must be populated and correct.
	if !r.SignalNoise.ClosureMeasured || r.SignalNoise.ClosureHonest != 0.92 {
		t.Fatalf("closure honesty = %v (measured=%v), want 0.92", r.SignalNoise.ClosureHonest, r.SignalNoise.ClosureMeasured)
	}
	if !r.SignalNoise.ShipMeasured || r.SignalNoise.ShipStampRate != 0.9 {
		t.Fatalf("ship-stamp rate = %v, want 0.9", r.SignalNoise.ShipStampRate)
	}
}

// Low closure honesty (<0.5) is the loudest signal — volume that is mostly
// unwitnessed — and must escalate the whole fleet to RED.
func TestFoldLowHonestyIsCrit(t *testing.T) {
	in := Inputs{
		Dispatch: PlaneInput{Payload: jmap(t, `{
			"closure": {"na": false, "closure_rate": 0.21, "counts": {"TRUE_RESOLVED": 167, "CLAIMED_CLOSED": 641}}
		}`)},
		Loops:   PlaneInput{Payload: jmap(t, `{"rollup": {"loops": 2, "live": 2, "dark": 0}}`)},
		Cadence: PlaneInput{Payload: jmap(t, `{"scores": {"debt": 0, "trend_direction": "flat"}, "work": {"commits": 10, "ships": 10, "window_days": 7}}`)},
		Fleet:   PlaneInput{Payload: jmap(t, `{"total": 0, "reachable": 0, "attention": []}`)},
	}
	r := Fold(in)
	if r.Verdict != VerdictRed {
		t.Fatalf("verdict = %s, want RED on 21%% honesty", r.Verdict)
	}
	if len(r.Attention) == 0 || r.Attention[0].Level != LevelCrit {
		t.Fatalf("top attention item must be crit, got %+v", r.Attention)
	}
	if !strings.Contains(r.Attention[0].Title, "Closure honesty") {
		t.Fatalf("top item should be the honesty crit, got %q", r.Attention[0].Title)
	}
	if r.Attention[0].Prov != Witnessed {
		t.Fatalf("honesty item provenance = %q, want WITNESSED", r.Attention[0].Prov)
	}
}

// An unmeasured plane must force WATCH, never a silent GREEN — a missing witness
// is honest, a fake green is a defect.
func TestUnmeasuredNeverGreen(t *testing.T) {
	in := Inputs{
		Dispatch: PlaneInput{Err: "dispatch_status.py timed out"},
		Loops:    PlaneInput{Payload: jmap(t, `{"rollup": {"loops": 1, "live": 1, "dark": 0}}`)},
		Cadence:  PlaneInput{Payload: jmap(t, `{"scores": {"debt": 0, "trend_direction": "flat"}, "work": {"commits": 5, "ships": 5, "window_days": 7}}`)},
		Fleet:    PlaneInput{Payload: jmap(t, `{"total": 1, "reachable": 1, "attention": []}`)},
	}
	r := Fold(in)
	if r.Verdict != VerdictWatch {
		t.Fatalf("verdict = %s, want WATCH when a plane is unmeasured", r.Verdict)
	}
	if r.OK {
		t.Fatal("OK = true with an unmeasured plane — must be false")
	}
	if r.Unmeasured != 1 {
		t.Fatalf("unmeasured = %d, want 1", r.Unmeasured)
	}
	// The dispatch plane row must be honestly flagged, not absent.
	var found bool
	for _, p := range r.Planes {
		if p.Name == "dispatch" {
			found = true
			if p.Measured || p.Verdict != PlaneUnmeasured {
				t.Fatalf("dispatch plane = %+v, want measured=false verdict=UNMEASURED", p)
			}
			if !strings.Contains(p.Err, "timed out") {
				t.Fatalf("dispatch plane err = %q, want the collector error", p.Err)
			}
		}
	}
	if !found {
		t.Fatal("dispatch plane row missing from coverage table")
	}
	// Closure S/N must stay unmeasured, not collapse to 0.
	if r.SignalNoise.ClosureMeasured {
		t.Fatal("closure S/N marked measured despite a failed dispatch collector")
	}
	// With no warns, the finding must name the unmeasured gap, not claim "nothing critical".
	if !strings.Contains(r.Finding, "unmeasured") || !strings.Contains(r.Finding, "GREEN") {
		t.Fatalf("WATCH-due-to-unmeasured finding should name the gap, got %q", r.Finding)
	}
}

// crit beats warn in ranking, and a 3+-dark-loop fleet is itself crit.
func TestRankingAndDarkLoops(t *testing.T) {
	in := Inputs{
		Dispatch: PlaneInput{Payload: jmap(t, `{
			"closure": {"na": false, "closure_rate": 0.70, "counts": {"TRUE_RESOLVED": 70, "CLAIMED_CLOSED": 30}},
			"throughput": {"na": false, "verdict": "BELOW_TARGET", "completed_rate_per_hour": 4.0, "target_per_hour": 10.0, "primary_window_hours": 6}
		}`)},
		Loops:   PlaneInput{Payload: jmap(t, `{"rollup": {"loops": 6, "live": 3, "dark": 3}}`)},
		Cadence: PlaneInput{Payload: jmap(t, `{"scores": {"debt": 2, "trend_direction": "flat"}, "work": {"commits": 20, "ships": 18, "window_days": 7}}`)},
		Fleet:   PlaneInput{Payload: jmap(t, `{"total": 2, "reachable": 2, "attention": []}`)},
	}
	r := Fold(in)
	if r.Verdict != VerdictRed {
		t.Fatalf("verdict = %s, want RED (3 dark loops is crit)", r.Verdict)
	}
	// First item must be the crit (dark loops); warns (honesty 70%, throughput) follow.
	if r.Attention[0].Level != LevelCrit {
		t.Fatalf("first item level = %s, want crit; list=%+v", r.Attention[0].Level, r.Attention)
	}
	for i := 1; i < len(r.Attention); i++ {
		if levelRank(r.Attention[i-1].Level) > levelRank(r.Attention[i].Level) {
			t.Fatalf("attention not rank-sorted at %d: %+v", i, r.Attention)
		}
	}
	// Three distinct warns expected after the crit (honesty, throughput).
	var warns int
	for _, it := range r.Attention {
		if it.Level == LevelWarn {
			warns++
		}
	}
	if warns < 2 {
		t.Fatalf("want >=2 warns (honesty + throughput), got %d in %+v", warns, r.Attention)
	}
}

// A regressed scorecard trend is a warn, and witnessed-closable issues surface as
// an actionable warn even when honesty itself is fine.
func TestScoresRegressionAndClosable(t *testing.T) {
	in := Inputs{
		Dispatch: PlaneInput{Payload: jmap(t, `{
			"closure": {"na": false, "closure_rate": 0.90, "counts": {"TRUE_RESOLVED": 90, "CLAIMED_CLOSED": 10}, "open_witnessed_closable": 8}
		}`)},
		Loops:   PlaneInput{Payload: jmap(t, `{"rollup": {"loops": 1, "live": 1, "dark": 0}}`)},
		Cadence: PlaneInput{Payload: jmap(t, `{"scores": {"debt": 12, "trend_direction": "regressed", "trend_summary": "doc-debt +7"}, "work": {"commits": 30, "ships": 25, "window_days": 7}}`)},
		Fleet:   PlaneInput{Payload: jmap(t, `{"total": 1, "reachable": 1, "attention": []}`)},
	}
	r := Fold(in)
	if r.Verdict != VerdictWatch {
		t.Fatalf("verdict = %s, want WATCH", r.Verdict)
	}
	var sawRegress, sawClosable bool
	for _, it := range r.Attention {
		if strings.Contains(it.Title, "debt regressed") {
			sawRegress = true
		}
		if strings.Contains(it.Title, "closable now") {
			sawClosable = true
		}
	}
	if !sawRegress {
		t.Fatalf("missing scores-regression warn in %+v", r.Attention)
	}
	if !sawClosable {
		t.Fatalf("missing witnessed-closable warn in %+v", r.Attention)
	}
}

// Fleet box attention rows must be lifted (crit/warn) and OK rows dropped.
func TestFleetAttentionPassthrough(t *testing.T) {
	in := Inputs{
		Dispatch: PlaneInput{Payload: jmap(t, `{"closure": {"na": true}}`)},
		Loops:    PlaneInput{Payload: jmap(t, `{"rollup": {"loops": 0, "live": 0, "dark": 0}}`)},
		Cadence:  PlaneInput{Payload: jmap(t, `{"scores": {"debt": 0, "trend_direction": "flat"}, "work": {"commits": 1, "ships": 1, "window_days": 7}}`)},
		Fleet: PlaneInput{Payload: jmap(t, `{"total": 4, "reachable": 3, "attention": [
			{"level": "crit", "title": "box gpu-1 down", "detail": "no report 40m"},
			{"level": "ok", "title": "box gpu-2 live"}
		]}`)},
	}
	r := Fold(in)
	var fleetItems int
	for _, it := range r.Attention {
		if it.Plane == "fleet" {
			fleetItems++
			if it.Prov != Observed {
				t.Fatalf("fleet item provenance = %q, want OBSERVED", it.Prov)
			}
		}
	}
	if fleetItems != 1 {
		t.Fatalf("fleet items lifted = %d, want 1 (crit kept, ok dropped)", fleetItems)
	}
	if r.Verdict != VerdictRed {
		t.Fatalf("verdict = %s, want RED (a box is down)", r.Verdict)
	}
}

// Render must lead with the verdict, show the S/N marquee, and list the ranked
// attention — and never silently drop an unmeasured plane.
func TestRender(t *testing.T) {
	in := Inputs{
		Dispatch: PlaneInput{Err: "boom"},
		Loops:    PlaneInput{Payload: jmap(t, `{"rollup": {"loops": 4, "live": 2, "dark": 2}}`)},
		Cadence:  PlaneInput{Payload: jmap(t, `{"scores": {"debt": 3, "trend_direction": "flat"}, "work": {"commits": 50, "ships": 40, "window_days": 7}}`)},
		Fleet:    PlaneInput{Payload: jmap(t, `{"total": 1, "reachable": 1, "attention": []}`)},
	}
	r := Fold(in)
	out := Render(r)
	for _, want := range []string{"Fleet state: WATCH", "Signal-to-noise", "Ship-stamp rate: 80%", "What needs you", "DARK", "Plane coverage", "**NO**", "unmeasured: boom"} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q\n---\n%s", want, out)
		}
	}
	// JSON round-trips.
	b, err := JSON(r)
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}
	if !strings.Contains(string(b), `"schema": "fak.exec-rollup/v1"`) {
		t.Fatalf("JSON missing schema:\n%s", string(b))
	}
}
