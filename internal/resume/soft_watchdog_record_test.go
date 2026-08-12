package resume

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

func stalledCurve(points ...int64) *trajctl.ObjectiveCurve {
	pts := make([]trajctl.CurvePoint, 0, len(points))
	for _, ms := range points {
		pts = append(pts, trajctl.CurvePoint{Value: .4, UnixMillis: ms})
	}
	return &trajctl.ObjectiveCurve{
		ObjectiveID: "issue-5287",
		Signal:      trajctl.SignalStall,
		Latest:      .4,
		Detail:      "flat witnessed progress",
		Methods:     []trajctl.MethodCurve{{Method: "commit-progress", Points: pts}},
	}
}

// The stall clock is the NEWEST timestamped point across every method curve, so the
// soft timeout is measured against real witnessed progress and not append order.
func TestCurveLastProgressTakesNewestWitnessedPoint(t *testing.T) {
	base := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	curve := stalledCurve(base.Add(2*time.Minute).UnixMilli(), base.UnixMilli())
	curve.Methods = append(curve.Methods, trajctl.MethodCurve{
		Method: "alignment",
		Points: []trajctl.CurvePoint{{Value: .5, UnixMillis: base.Add(5 * time.Minute).UnixMilli()}},
	})
	if got := CurveLastProgress(curve); !got.Equal(base.Add(5 * time.Minute)) {
		t.Fatalf("CurveLastProgress = %v, want %v", got, base.Add(5*time.Minute))
	}
	if got := CurveLastProgress(nil); !got.IsZero() {
		t.Fatalf("a nil curve carries no stall clock, got %v", got)
	}
	if got := CurveLastProgress(stalledCurve()); !got.IsZero() {
		t.Fatalf("an unstamped curve carries no stall clock, got %v", got)
	}
}

// The observation is derived from the anchor curve the trajectory watchdog already
// reads — no extra IO — and a curve with no stall clock captures nothing, because
// the soft TIMEOUT is the gate and an unknown clock cannot be proven elapsed.
func TestSoftObservationFromCurveGatesOnTheWitnessedStallClock(t *testing.T) {
	now := time.Date(2026, 7, 18, 12, 30, 0, 0, time.UTC)
	stalledSince := now.Add(-30 * time.Minute)

	in := SoftObservationFromCurve("sess-5287", true, stalledCurve(stalledSince.UnixMilli()), "claude --resume sess-5287", now)
	if in.Signal != trajctl.SignalStall || in.LastProgressMarker != "flat witnessed progress" ||
		!in.LastProgressAt.Equal(stalledSince) || in.PendingAction != "claude --resume sess-5287" || !in.Alive {
		t.Fatalf("observation not derived from the anchor curve: %+v", in)
	}
	dump, ok := DecideSoftStateDump(in, time.Minute)
	if !ok || dump.ElapsedSinceProgressMillis != (30*time.Minute).Milliseconds() {
		t.Fatalf("stalled-past-grace must dump with the witnessed elapsed: ok=%v dump=%+v", ok, dump)
	}

	// Same stall, no timestamped progress point: the clock is unknown, so nothing
	// is captured rather than a fabricated "infinitely old" stall.
	noClock := SoftObservationFromCurve("sess-5287", true, stalledCurve(), "", now)
	if _, ok := DecideSoftStateDump(noClock, time.Minute); ok {
		t.Fatal("a stall with no witnessed progress timestamp must not dump")
	}
	if _, ok := NewSoftWatchdog(time.Minute).Observe(noClock); ok {
		t.Fatal("the episode tracker must honour the unknown stall clock too")
	}
}

// The dump becomes a schema-pinned, session-keyed row that survives a JSONL
// round-trip through the durable session record and folds back last-row-wins.
func TestSoftDumpRowRoundTripsThroughTheDurableRecord(t *testing.T) {
	now := time.Date(2026, 7, 18, 12, 30, 0, 0, time.UTC)
	in := SoftObservationFromCurve("sess-5287", true, stalledCurve(now.Add(-30*time.Minute).UnixMilli()), "claude --resume sess-5287", now)
	dump, ok := DecideSoftStateDump(in, time.Minute)
	if !ok {
		t.Fatal("setup: expected a dump")
	}
	row := NewSoftDumpRow(dump, "trace-5287")
	if row.Schema != SoftDumpSchema || row.Phase != SoftDumpPhase || row.Session != "sess-5287" || row.Trace != "trace-5287" {
		t.Fatalf("row envelope not durable-record ready: %+v", row)
	}
	if row.TS != "" {
		t.Fatal("the pure constructor must not stamp a clock; the writing shell does")
	}

	line, err := json.Marshal(row)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(line), `"phase":"soft_state_dump"`) {
		t.Fatalf("an operator must be able to grep the phase: %s", line)
	}
	var back SoftDumpRow
	if err := json.Unmarshal(line, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back.Dump.LivenessVsProgress != SoftSplitAliveWithoutProgress ||
		back.Dump.ElapsedSinceProgressMillis != dump.ElapsedSinceProgressMillis ||
		back.Dump.PendingAction != "claude --resume sess-5287" ||
		back.Dump.LastProgressMarker != "flat witnessed progress" {
		t.Fatalf("diagnostic fields lost in the durable round-trip: %+v", back.Dump)
	}

	older := NewSoftDumpRow(dump, "trace-old")
	folded := FoldSoftDumps([]SoftDumpRow{older, back, {Session: ""}})
	if len(folded) != 1 || folded["sess-5287"].Trace != "trace-5287" {
		t.Fatalf("fold must be last-row-wins per session and skip sessionless rows: %+v", folded)
	}
}
