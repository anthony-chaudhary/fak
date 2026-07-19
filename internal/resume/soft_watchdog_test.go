package resume

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

func softObs(alive bool, sig trajctl.Signal, stalledFor time.Duration) SoftWatchdogObservation {
	base := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	return SoftWatchdogObservation{
		SessionID:          "sess-1",
		Alive:              alive,
		Signal:             sig,
		LastProgressMarker: "commit abc123 phase p2",
		LastProgressAt:     base,
		PendingAction:      "go test ./internal/resume",
		Now:                base.Add(stalledFor),
	}
}

// An alive-but-stalled session past the grace window triggers exactly one dump
// carrying every diagnostic field; the same unbroken stall never dumps twice,
// and a healthy interlude closes the episode so a fresh stall dumps again.
func TestSoftWatchdogDumpsAliveStalledExactlyOnce(t *testing.T) {
	w := NewSoftWatchdog(time.Minute)
	in := softObs(true, trajctl.SignalStall, 3*time.Minute)

	dump, ok := w.Observe(in)
	if !ok {
		t.Fatal("first alive-but-stalled observation past grace must dump")
	}
	if dump.SessionID != "sess-1" ||
		dump.Signal != trajctl.SignalStall ||
		dump.LastProgressMarker != "commit abc123 phase p2" ||
		dump.ElapsedSinceProgressMillis != (3*time.Minute).Milliseconds() ||
		dump.PendingAction != "go test ./internal/resume" ||
		!dump.Alive || !dump.ProgressStalled ||
		dump.LivenessVsProgress != SoftSplitAliveWithoutProgress ||
		dump.CapturedAtUnixMillis != in.Now.UnixMilli() ||
		dump.Reason == "" {
		t.Fatalf("dump missing diagnostic fields: %+v", dump)
	}
	if _, again := w.Observe(in); again {
		t.Fatal("same unbroken stall episode must not dump twice")
	}

	// Progress resumes (healthy) -> episode closes -> a later stall re-dumps.
	if _, healthy := w.Observe(softObs(true, trajctl.SignalHealthy, 0)); healthy {
		t.Fatal("healthy observation must not dump")
	}
	if _, second := w.Observe(in); !second {
		t.Fatal("a fresh stall episode after recovery must dump again")
	}
}

// Healthy or still-within-grace sessions capture nothing.
func TestSoftWatchdogHealthyOrWithinGraceDumpsNothing(t *testing.T) {
	tests := []struct {
		name string
		in   SoftWatchdogObservation
	}{
		{"healthy progressing session", softObs(true, trajctl.SignalHealthy, 3 * time.Minute)},
		{"stalled but inside grace window", softObs(true, trajctl.SignalStall, 30 * time.Second)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := NewSoftWatchdog(time.Minute)
			if _, ok := w.Observe(tt.in); ok {
				t.Fatalf("%s must not trigger a soft dump", tt.name)
			}
		})
	}
}

// A dead session is the hard path's business: no soft dump, and the unchanged
// intervention core still revives from the fresh anchor.
func TestSoftWatchdogLeavesDeadSessionsToHardPath(t *testing.T) {
	w := NewSoftWatchdog(time.Minute)
	dump, decision := w.ObserveThenDecide(softObs(false, trajctl.SignalStall, 3*time.Minute), false)
	if dump != nil {
		t.Fatalf("dead session must not soft-dump, got %+v", *dump)
	}
	if decision.Action != TrajectoryReviveAnchor {
		t.Fatalf("dead session decision = %+v, want %s", decision, TrajectoryReviveAnchor)
	}
}

// The dump fires BEFORE the nudge/revive decision and never alters it: the
// composed decision is byte-identical to calling the intervention core alone.
func TestSoftWatchdogDumpPrecedesUnchangedDecision(t *testing.T) {
	w := NewSoftWatchdog(time.Minute)
	in := softObs(true, trajctl.SignalStall, 3*time.Minute)
	dump, decision := w.ObserveThenDecide(in, false)
	if dump == nil {
		t.Fatal("alive-but-stalled session must carry a forensic dump into the decision")
	}
	want := DecideTrajectoryWatchdog(TrajectoryWatchdogInput{Alive: true, Signal: trajctl.SignalStall})
	if decision != want || decision.Action != TrajectoryNudge {
		t.Fatalf("composed decision %+v differs from bare intervention core %+v", decision, want)
	}
}
