package gateway

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// debug_request_admission_test.go — the #13120 witness suite for the /debug/vars
// request_admission join. It asserts the one-trace contract: the block renders the last
// admission decision AND the last native phase for the SAME most-recent live trace, is
// deterministic, omits itself when neither producer is present, and still renders the half
// available when only one producer is wired.

// phaseOnlyPlanner is a minimal agent.Planner that also satisfies agent.NativePhaseReporter.
// It exists ONLY to drive the debug phase half; its Complete is never reached by the read path.
type phaseOnlyPlanner struct {
	stubPlanner
	byTrace map[string]agent.NativePhaseObservation
}

func (p *phaseOnlyPlanner) NativePhaseObservation(traceID string) (agent.NativePhaseObservation, bool) {
	o, ok := p.byTrace[traceID]
	return o, ok
}

// liveSessions returns a listSessions stub that reports the given states verbatim.
func liveSessions(states ...SessionState) func(context.Context) []SessionState {
	return func(context.Context) []SessionState { return states }
}

// TestDebugRequestAdmissionJoinsBothHalves is the primary witness: with both producers wired
// the block renders the decision and phase halves bound to the SAME most-recent live trace.
func TestDebugRequestAdmissionJoinsBothHalves(t *testing.T) {
	srv := newTestServer(t)
	at := time.Date(2026, 9, 15, 12, 0, 0, 0, time.UTC)

	// "newer" has the greater Rev, so it is the deterministic pick over "older".
	srv.listSessions = liveSessions(
		SessionState{TraceID: "older", Run: "running", Rev: 1},
		SessionState{TraceID: "newer", Run: "running", Rev: 9},
	)

	ctl := NewAdmissionController(DefaultAdmissionPolicy())
	ctl.SetClock(func() time.Time { return at })
	if v := ctl.Offer(SeqRequest{TraceID: "newer", SessionID: "newer", Tokens: 42, Priority: 3}); v != VerdictAdmitted {
		t.Fatalf("Offer = %s, want admitted", v)
	}
	srv.SetAdmissionController(ctl)

	srv.planner = &phaseOnlyPlanner{byTrace: map[string]agent.NativePhaseObservation{
		"newer": {TraceID: "newer", Phase: agent.NativePhaseDecode, At: at.Add(2 * time.Second), Elapsed: 1500 * time.Millisecond, Completed: true},
	}}

	got := srv.debugRequestAdmission(at.Add(3 * time.Second))
	if got == nil {
		t.Fatal("block nil with both producers wired")
	}
	if got.TraceID != "newer" {
		t.Fatalf("TraceID = %q, want the max-Rev live trace newer", got.TraceID)
	}
	if got.Decision == nil {
		t.Fatal("decision half nil, want the admitted receipt")
	}
	if got.Decision.Verdict != "admitted" || !got.Decision.Admitted {
		t.Errorf("decision verdict/admitted = %q/%v, want admitted/true", got.Decision.Verdict, got.Decision.Admitted)
	}
	if got.Decision.Tokens != 42 || got.Decision.Priority != 3 || got.Decision.SessionID != "newer" {
		t.Errorf("decision = %+v, want tokens=42 priority=3 session=newer", got.Decision)
	}
	if got.Decision.AtUnixNano != at.UnixNano() {
		t.Errorf("decision at_unix_nano = %d, want %d", got.Decision.AtUnixNano, at.UnixNano())
	}
	if got.NativePhase == nil {
		t.Fatal("native_phase half nil, want the decode observation")
	}
	if got.NativePhase.Phase != string(agent.NativePhaseDecode) {
		t.Errorf("phase = %q, want %q", got.NativePhase.Phase, agent.NativePhaseDecode)
	}
	if !got.NativePhase.Completed || got.NativePhase.ElapsedMs != 1500 {
		t.Errorf("native_phase = %+v, want completed elapsed_ms=1500", got.NativePhase)
	}
	if got.NativePhase.AtUnixNano != at.Add(2*time.Second).UnixNano() {
		t.Errorf("phase at_unix_nano = %d, want %d", got.NativePhase.AtUnixNano, at.Add(2*time.Second).UnixNano())
	}

	// The full debug response must serialize the block under its wire key.
	b, err := json.Marshal(srv.debugVars(at.Add(3 * time.Second)))
	if err != nil {
		t.Fatalf("marshal debugVars: %v", err)
	}
	if !strings.Contains(string(b), `"request_admission"`) {
		t.Errorf("response JSON omits request_admission: %s", b)
	}
	if !strings.Contains(string(b), `"native_phase"`) {
		t.Errorf("response JSON omits the native_phase half: %s", b)
	}
}

// TestDebugRequestAdmissionTraceChoiceIsDeterministic pins the pick rule directly: stopped
// sessions and blank trace ids are excluded, the greatest Rev wins, and an equal-Rev tie is
// broken by the lexicographically smaller trace id (so repeated scrapes agree).
func TestDebugRequestAdmissionTraceChoiceIsDeterministic(t *testing.T) {
	srv := newTestServer(t)

	cases := []struct {
		name   string
		states []SessionState
		want   string
	}{
		{
			name:   "max rev wins over insertion order",
			states: []SessionState{{TraceID: "a", Run: "running", Rev: 2}, {TraceID: "b", Run: "throttled", Rev: 5}},
			want:   "b",
		},
		{
			name:   "equal rev tie broken lexicographically",
			states: []SessionState{{TraceID: "zzz", Run: "running", Rev: 4}, {TraceID: "aaa", Run: "running", Rev: 4}},
			want:   "aaa",
		},
		{
			name:   "stopped sessions excluded",
			states: []SessionState{{TraceID: "gone", Run: "stopped", Rev: 99}, {TraceID: "live", Run: "running", Rev: 1}},
			want:   "live",
		},
		{
			name:   "blank trace ids ignored",
			states: []SessionState{{TraceID: "   ", Run: "running", Rev: 99}, {TraceID: "real", Run: "running", Rev: 1}},
			want:   "real",
		},
		{
			name:   "all stopped yields no trace",
			states: []SessionState{{TraceID: "gone", Run: "stopped", Rev: 3}},
			want:   "",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv.listSessions = liveSessions(tc.states...)
			if got := srv.mostRecentLiveTrace(context.Background()); got != tc.want {
				t.Errorf("mostRecentLiveTrace = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestDebugRequestAdmissionOmittedWhenNeitherProducer asserts the all-absent contract: a
// proxy/mock serve with no admission controller and a non-reporting planner omits the block
// entirely (no fabricated zero row) from both the accessor and the serialized response.
func TestDebugRequestAdmissionOmittedWhenNeitherProducer(t *testing.T) {
	srv := newTestServer(t)
	srv.listSessions = liveSessions(SessionState{TraceID: "live", Run: "running", Rev: 1})
	// srv.planner defaults to a plain stubPlanner (not a NativePhaseReporter).

	if got := srv.debugRequestAdmission(time.Now()); got != nil {
		t.Fatalf("debugRequestAdmission = %+v, want nil with neither producer", got)
	}
	b, err := json.Marshal(srv.debugVars(time.Now()))
	if err != nil {
		t.Fatalf("marshal debugVars: %v", err)
	}
	if strings.Contains(string(b), "request_admission") {
		t.Errorf("response must omit request_admission with no producer: %s", b)
	}
}

// TestDebugRequestAdmissionRendersDecisionHalfOnly asserts the partial case: with just the
// admission controller wired the block renders, the decision half is populated, and the phase
// half is nil rather than a fabricated empty observation.
func TestDebugRequestAdmissionRendersDecisionHalfOnly(t *testing.T) {
	srv := newTestServer(t)
	srv.listSessions = liveSessions(SessionState{TraceID: "live", Run: "running", Rev: 1})

	ctl := NewAdmissionController(DefaultAdmissionPolicy())
	ctl.Offer(SeqRequest{TraceID: "live", Tokens: 7})
	srv.SetAdmissionController(ctl)

	got := srv.debugRequestAdmission(time.Now())
	if got == nil {
		t.Fatal("block nil with the controller wired")
	}
	if got.Decision == nil || got.Decision.Tokens != 7 {
		t.Fatalf("decision = %+v, want tokens=7", got.Decision)
	}
	if got.NativePhase != nil {
		t.Errorf("native_phase = %+v, want nil with no phase reporter", got.NativePhase)
	}
}

// TestDebugRequestAdmissionRendersPhaseHalfOnly asserts the mirrored partial case: with only a
// phase-reporting planner wired the decision half is nil but the phase half still renders.
func TestDebugRequestAdmissionRendersPhaseHalfOnly(t *testing.T) {
	srv := newTestServer(t)
	at := time.Date(2026, 9, 15, 12, 0, 0, 0, time.UTC)
	srv.listSessions = liveSessions(SessionState{TraceID: "live", Run: "running", Rev: 1})
	srv.planner = &phaseOnlyPlanner{byTrace: map[string]agent.NativePhaseObservation{
		"live": {TraceID: "live", Phase: agent.NativePhaseTerminal, At: at, Completed: true},
	}}

	got := srv.debugRequestAdmission(at)
	if got == nil {
		t.Fatal("block nil with a phase reporter wired")
	}
	if got.Decision != nil {
		t.Errorf("decision = %+v, want nil with no controller", got.Decision)
	}
	if got.NativePhase == nil || got.NativePhase.Phase != string(agent.NativePhaseTerminal) {
		t.Fatalf("native_phase = %+v, want terminal", got.NativePhase)
	}
}

// TestDebugRequestAdmissionNilWhenNoLiveTrace asserts the no-trace guard: producers wired but
// no live session means no join key, so the block is nil (never keyed on a phantom trace).
func TestDebugRequestAdmissionNilWhenNoLiveTrace(t *testing.T) {
	srv := newTestServer(t)
	srv.listSessions = liveSessions(SessionState{TraceID: "gone", Run: "stopped", Rev: 5})

	ctl := NewAdmissionController(DefaultAdmissionPolicy())
	srv.SetAdmissionController(ctl)
	srv.planner = &phaseOnlyPlanner{byTrace: map[string]agent.NativePhaseObservation{}}

	if got := srv.debugRequestAdmission(time.Now()); got != nil {
		t.Fatalf("debugRequestAdmission = %+v, want nil with no live trace", got)
	}
}
