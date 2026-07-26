package gateway

// Tests for the deployment-boundary SESSION SATURATION signal (#3425). The contract:
// a configured ceiling (FAK_MAX_SESSIONS) yields a live saturation readout and a
// /metrics gauge, and past the ceiling a NEW session is backpressured with the closed
// reason SESSION_CEILING_SATURATED — while a trace already resident in the registry is
// NEVER refused (an in-flight loop is never sacrificed to admit a new one). All run
// under `go test ./internal/gateway -run TestSessionCeiling` (the dos.toml floor).

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestSessionCeilingSaturationUnboundedInert(t *testing.T) {
	// A zero/negative ceiling is unbounded: no saturation, and every admission (new or
	// resident) is allowed — the historical fail-open path.
	for _, ceiling := range []int{0, -1} {
		sat := sessionSaturation(9, ceiling)
		if sat.Bounded {
			t.Fatalf("ceiling %d: Bounded = true, want false (unbounded)", ceiling)
		}
		if sat.Ratio != 0 || sat.Headroom != 0 {
			t.Fatalf("ceiling %d: unbounded readout should carry no ratio/headroom, got %+v", ceiling, sat)
		}
		if !sat.admitNewSession(false) || !sat.admitNewSession(true) {
			t.Fatalf("ceiling %d: unbounded must admit both new and resident", ceiling)
		}
	}
}

func TestSessionCeilingSaturationRatioHeadroom(t *testing.T) {
	sat := sessionSaturation(3, 4)
	if !sat.Bounded || sat.Ceiling != 4 || sat.Live != 3 {
		t.Fatalf("readout = %+v, want bounded ceiling=4 live=3", sat)
	}
	if sat.Ratio != 0.75 {
		t.Fatalf("Ratio = %v, want 0.75", sat.Ratio)
	}
	if sat.Headroom != 1 {
		t.Fatalf("Headroom = %d, want 1", sat.Headroom)
	}
	// Over-ceiling: ratio > 1, headroom floored at 0 (never negative).
	over := sessionSaturation(6, 4)
	if over.Ratio != 1.5 || over.Headroom != 0 {
		t.Fatalf("over-ceiling readout = %+v, want ratio=1.5 headroom=0", over)
	}
}

func TestSessionCeilingNeverSacrificesInFlight(t *testing.T) {
	// At exactly the ceiling: a NEW (non-resident) session is refused, but a session
	// already resident in the registry is still admitted — no in-flight loop is dropped.
	full := sessionSaturation(4, 4)
	if full.admitNewSession(false) {
		t.Fatal("a new session at the ceiling must be refused")
	}
	if !full.admitNewSession(true) {
		t.Fatal("a RESIDENT session at the ceiling must still be admitted (never sacrifice in-flight)")
	}
	// Below the ceiling a new session is admitted.
	if !sessionSaturation(3, 4).admitNewSession(false) {
		t.Fatal("a new session below the ceiling must be admitted")
	}
}

func TestSessionCeilingServerRefusal(t *testing.T) {
	ctx := context.Background()
	// Two sessions already resident; ceiling of 2 → the box is full.
	s := &Server{listSessions: func(context.Context) []SessionState {
		return []SessionState{{TraceID: "gw-1", Run: "running"}, {TraceID: "gw-2", Run: "running"}}
	}}

	// No ceiling configured → inert: never refuses, even a brand-new trace.
	t.Setenv(envMaxSessions, "")
	if ref := s.sessionCeilingRefusal(ctx, "gw-new"); ref != nil {
		t.Fatalf("unbounded ceiling must not refuse, got %+v", ref)
	}
	if s.sessionSaturationNow(ctx).Bounded {
		t.Fatal("unbounded ceiling must report Bounded=false")
	}

	// Ceiling reached (2 live, ceiling 2): a NEW trace is refused with the closed reason.
	t.Setenv(envMaxSessions, "2")
	ref := s.sessionCeilingRefusal(ctx, "gw-new")
	if ref == nil {
		t.Fatal("new session past the ceiling must be refused")
	}
	if ref.Reason != ReasonSessionCeilingSaturated {
		t.Fatalf("refusal reason = %q, want %q", ref.Reason, ReasonSessionCeilingSaturated)
	}
	// A trace already resident is NEVER refused, even at the ceiling.
	if got := s.sessionCeilingRefusal(ctx, "gw-1"); got != nil {
		t.Fatalf("resident session must not be refused at the ceiling, got %+v", got)
	}
	// The live readout reflects the configured ceiling.
	sat := s.sessionSaturationNow(ctx)
	if !sat.Bounded || sat.Ceiling != 2 || sat.Live != 2 || sat.Ratio != 1.0 {
		t.Fatalf("saturation = %+v, want bounded ceiling=2 live=2 ratio=1.0", sat)
	}
}

func TestSessionCeilingMetricsGauge(t *testing.T) {
	s := &Server{listSessions: func(context.Context) []SessionState {
		return []SessionState{{TraceID: "gw-1"}, {TraceID: "gw-2"}}
	}}

	// Unarmed (no ceiling): the family is absent entirely — quiet until armed.
	t.Setenv(envMaxSessions, "")
	var off strings.Builder
	s.writeSessionSaturationMetrics(&off)
	if off.Len() != 0 {
		t.Fatalf("unarmed gauge must emit nothing, got %q", off.String())
	}

	// Armed: the ceiling/live/saturation gauges appear with the derived values.
	t.Setenv(envMaxSessions, "4")
	var on strings.Builder
	s.writeSessionSaturationMetrics(&on)
	out := on.String()
	for _, want := range []string{
		"fak_gateway_session_ceiling 4",
		"fak_gateway_sessions_live 2",
		"fak_gateway_session_saturation 0.5",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("gauge output missing %q:\n%s", want, out)
		}
	}
}

// TestSessionCeilingReadinessField witnesses the OTHER half of the saturation signal
// (#3425): the readout is a field on the readiness surface (/healthz), not only a
// /metrics gauge — so an operator or autoscaler probing readiness sees headroom. Being
// AT the ceiling must not flip ok:false: the deployment is healthy, it just sheds NEW
// sessions while the in-flight loops run untouched.
func TestSessionCeilingReadinessField(t *testing.T) {
	s := &Server{listSessions: func(context.Context) []SessionState {
		return []SessionState{{TraceID: "gw-1"}, {TraceID: "gw-2"}}
	}}

	probe := func(t *testing.T) map[string]any {
		t.Helper()
		rec := httptest.NewRecorder()
		s.handleHealth(rec, httptest.NewRequest(http.MethodGet, "/healthz", nil))
		var body map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
			t.Fatalf("decode /healthz: %v (body %q)", err, rec.Body.String())
		}
		return body
	}

	// Unarmed (no ceiling): the field is absent entirely — historical payload.
	t.Setenv(envMaxSessions, "")
	if _, ok := probe(t)["session_saturation"]; ok {
		t.Fatal("unbounded deployment must not carry session_saturation on /healthz")
	}

	// Armed and exactly AT the ceiling: the field reports the readout, ok stays true.
	t.Setenv(envMaxSessions, "2")
	body := probe(t)
	if body["ok"] != true {
		t.Fatalf("saturation must not flip readiness to not-ok, got ok=%v", body["ok"])
	}
	sat, ok := body["session_saturation"].(map[string]any)
	if !ok {
		t.Fatalf("armed deployment must carry session_saturation on /healthz, got %v", body["session_saturation"])
	}
	if sat["ceiling"] != float64(2) || sat["live"] != float64(2) || sat["ratio"] != float64(1) {
		t.Fatalf("session_saturation = %v, want ceiling=2 live=2 ratio=1", sat)
	}
	if sat["headroom"] != float64(0) || sat["bounded"] != true {
		t.Fatalf("session_saturation = %v, want headroom=0 bounded=true", sat)
	}
}

// TestSessionCeilingReasonDeclaredInDosToml is the closed-vocabulary conformance
// witness (#3425): the reason the runtime emits at the saturation boundary MUST be a
// declared, refusable dos.toml [reasons] table, or `dos_check_reason
// SESSION_CEILING_SATURATED` returns known=false / UNCLASSIFIED — the free-text drift
// the closed vocabulary exists to kill. It reuses the package-local dos-reason helpers.
func TestSessionCeilingReasonDeclaredInDosToml(t *testing.T) {
	content := readRepoDosTomlForReversibility(t)
	header := "[reasons." + ReasonSessionCeilingSaturated + "]"
	if !strings.Contains(content, header) {
		t.Fatalf("gateway emits refusal token %q but dos.toml has no %s table — "+
			"dos_check_reason %s would return known=false (UNCLASSIFIED drift)",
			ReasonSessionCeilingSaturated, header, ReasonSessionCeilingSaturated)
	}
	block := reversibilityDosReasonBlock(content, header)
	if !reversibilityDosReasonFieldTrue(block, "refusal") {
		t.Fatalf("reason %q is declared but not marked refusal = true — "+
			"dos_check_reason would resolve it as non-refusable", ReasonSessionCeilingSaturated)
	}
}
