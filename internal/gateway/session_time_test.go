package gateway

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestSessionTimeIsZero pins the omitzero contract: a session with no wall-clock
// budget must marshal byte-identically to a pre-#1584 SessionState — i.e. the "time"
// key is absent entirely, not an empty object.
func TestSessionTimeIsZero(t *testing.T) {
	if !(SessionTime{}).IsZero() {
		t.Fatal("zero SessionTime must report IsZero() = true")
	}
	if (SessionTime{ElapsedSeconds: 1}).IsZero() {
		t.Fatal("a SessionTime carrying elapsed time must not report IsZero()")
	}
	if (SessionTime{Bounded: true}).IsZero() {
		t.Fatal("a bounded SessionTime must not report IsZero()")
	}

	// A state with no time budget: the "time" key must be OMITTED from the wire.
	noTime, err := json.Marshal(SessionState{TraceID: "sess-1", Run: "running"})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(noTime), "\"time\"") {
		t.Fatalf("a session with no wall-clock budget must omit the time key; got %s", noTime)
	}
}

// TestSessionTimeMarshalsWhenPresent confirms a real wall-clock budget crosses the
// wire with all four legible fields — the projection `fak session status` renders.
func TestSessionTimeMarshalsWhenPresent(t *testing.T) {
	st := SessionState{
		TraceID: "sess-1",
		Run:     "running",
		Time: SessionTime{
			Bounded:          true,
			ElapsedSeconds:   300,
			RemainingSeconds: 6900,
			LimitSeconds:     7200,
		},
	}
	b, err := json.Marshal(st)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var round SessionState
	if err := json.Unmarshal(b, &round); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if round.Time != st.Time {
		t.Fatalf("time round-trip = %+v, want %+v", round.Time, st.Time)
	}
	for _, want := range []string{`"bounded":true`, `"elapsed_seconds":300`, `"remaining_seconds":6900`, `"limit_seconds":7200`} {
		if !strings.Contains(string(b), want) {
			t.Fatalf("wire form missing %q; got %s", want, b)
		}
	}
}
