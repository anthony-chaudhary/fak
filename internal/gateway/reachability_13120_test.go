package gateway

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// reachability_13120_test.go — the PRODUCTION-REACHABILITY witness for the #13120
// admission/native-phase producer. The original producer suites pass by construction:
// admission_decision_test.go reads back the exact suffixed key Acquire stored, and
// native_phase_test.go hand-injects the plain-string "trace_id" context key. Neither
// crosses a production setter, so neither caught that the served path cannot render the
// /debug/vars request_admission block.
//
// These tests cross the production seams instead:
//   - LastAdmissionDecision must resolve a decision from the BARE session trace id that
//     mostRecentLiveTrace() returns, even though Acquire stores it under "<bare>#<seq>".
//   - nativePhaseTraceID must read the trace id from the context key the served path
//     actually sets (agent.WithRequestTraceID), not a bare-string key no producer sets.

// TestLastAdmissionDecisionResolvesBareServedTrace drives the LIVE Acquire boundary with
// the bare id the served path uses, then reads it back with that SAME bare id — exactly
// what debugRequestAdmission does with mostRecentLiveTrace(). Today the receipt is stored
// under "gw-7#<seq>" so the bare lookup misses and the debug block vanishes.
func TestLastAdmissionDecisionResolvesBareServedTrace(t *testing.T) {
	c := NewAdmissionController(DefaultAdmissionPolicy())

	lease, err := c.Acquire(context.Background(), SeqRequest{TraceID: "gw-7", SessionID: "gw-7", Tokens: 64, Priority: 3})
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	defer lease.Release()

	rec, ok := c.LastAdmissionDecision("gw-7")
	if !ok {
		t.Fatal("LastAdmissionDecision(bare served id) missed; the debug request_admission decision half cannot render in production")
	}
	if rec.SessionID != "gw-7" || rec.Tokens != 64 {
		t.Fatalf("receipt = %+v, want the gw-7 admitted receipt", rec)
	}
	if rec.TraceID == "gw-7" {
		t.Fatalf("receipt.TraceID = %q; want the stored suffixed id preserved on the receipt", rec.TraceID)
	}

	// The exact suffixed key must keep resolving too (no regression on the precise lookup).
	if _, ok := c.LastAdmissionDecision(rec.TraceID); !ok {
		t.Fatalf("exact suffixed key %q no longer resolves", rec.TraceID)
	}

	// A different base trace must NOT be answered by gw-7's receipt.
	if _, ok := c.LastAdmissionDecision("gw-8"); ok {
		t.Fatal("LastAdmissionDecision(gw-8) returned gw-7's receipt; base-trace matching is not scoped")
	}
}

// TestLastAdmissionDecisionPrefersLatestSuffixedForBase pins that when a base trace has
// several suffixed decisions (a queued request later promoted, a retry), the read-back
// returns the LATEST one, not an arbitrary map-order pick.
func TestLastAdmissionDecisionPrefersLatestSuffixedForBase(t *testing.T) {
	c := NewAdmissionController(AdmissionPolicy{TokenBudget: 1, MaxWaiting: 8, AgingRounds: 1})

	l1, err := c.Acquire(context.Background(), SeqRequest{TraceID: "gw-9", SessionID: "gw-9", Tokens: 1, Priority: 1})
	if err != nil {
		t.Fatalf("first Acquire: %v", err)
	}
	defer l1.Release()

	// Second request on the same base trace queues behind the first (budget exhausted);
	// an already-cancelled context makes it return promptly after recording #2.
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := c.Acquire(canceled, SeqRequest{TraceID: "gw-9", SessionID: "gw-9", Tokens: 1, Priority: 1}); err == nil {
		t.Fatal("second Acquire with a cancelled context returned no error")
	}

	rec, ok := c.LastAdmissionDecision("gw-9")
	if !ok {
		t.Fatal("no bare-trace decision resolved")
	}
	// #2 is the later decision; its receipt must win.
	if rec.TraceID != "gw-9#2" {
		t.Fatalf("resolved receipt = %q, want the latest gw-9#2", rec.TraceID)
	}
}

// TestLastAdmissionDecisionHandlesClientTraceWithHash pins the adversarial client id: an
// X-Trace-Id that itself contains "#" is unsanitized by the served path, so Acquire stores
// "foo#bar#1". A bare read for "foo#bar" must still resolve it (prefix form), and this must
// not leak across bases.
func TestLastAdmissionDecisionHandlesClientTraceWithHash(t *testing.T) {
	c := NewAdmissionController(DefaultAdmissionPolicy())
	lease, err := c.Acquire(context.Background(), SeqRequest{TraceID: "foo#bar", SessionID: "foo#bar", Tokens: 32, Priority: 1})
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	defer lease.Release()

	if rec, ok := c.LastAdmissionDecision("foo#bar"); !ok || rec.TraceID != "foo#bar#1" {
		t.Fatalf("hash-bearing client trace unresolved: rec=%+v ok=%v", rec, ok)
	}
	// A different id must not borrow it.
	if _, ok := c.LastAdmissionDecision("foo"); ok {
		t.Fatal("LastAdmissionDecision(foo) borrowed the foo#bar receipt")
	}
}

// TestDualPlannerForwardsNativePhaseReporter pins the DUAL-serve reachability gap the
// first cut of this fix left open: in dual mode s.planner is *DualPlanner, so unless
// DualPlanner itself satisfies agent.NativePhaseReporter the debug handler's type
// assertion fails and the native_phase half is silently omitted even though the local
// planner recorded it. DualPlanner must forward to its local side; the proxy side has no
// native phase and must never answer.
func TestDualPlannerForwardsNativePhaseReporter(t *testing.T) {
	local := &dualPhaseSide{
		dualSide: dualSide{id: "local-model"},
		byTrace: map[string]agent.NativePhaseObservation{
			"gw-dual": {TraceID: "gw-dual", Phase: agent.NativePhasePrefill},
		},
	}
	proxy := &dualSide{id: "api-model"}

	d, err := NewDualPlanner(proxy, local, "local-model")
	if err != nil {
		t.Fatalf("NewDualPlanner: %v", err)
	}

	reporter, ok := agent.Planner(d).(agent.NativePhaseReporter)
	if !ok {
		t.Fatal("*DualPlanner does not satisfy NativePhaseReporter; the native_phase half cannot render in dual mode")
	}
	obs, ok := reporter.NativePhaseObservation("gw-dual")
	if !ok || obs.Phase != agent.NativePhasePrefill {
		t.Fatalf("observation = %+v/%v, want the local side's prefill observation", obs, ok)
	}
	if _, ok := reporter.NativePhaseObservation("absent"); ok {
		t.Fatal("DualPlanner reported a phase for an unrecorded trace")
	}
}

// dualPhaseSide is a dual-side planner that also implements the NativePhaseReporter seam.
type dualPhaseSide struct {
	dualSide
	byTrace map[string]agent.NativePhaseObservation
}

func (p *dualPhaseSide) NativePhaseObservation(traceID string) (agent.NativePhaseObservation, bool) {
	o, ok := p.byTrace[traceID]
	return o, ok
}

// TestBeginServedRequestStampsPlannerTrace drives the REAL served boundary and asserts the
// minted request trace is stamped onto the returned context under the agent's request-trace
// key — the wiring the in-kernel planner's nativePhaseTraceID reads. Before the fix the
// boundary set no such key, so every native-phase observation keyed "" and the debug block
// never rendered.
func TestBeginServedRequestStampsPlannerTrace(t *testing.T) {
	srv := newTestServer(t)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("X-Trace-Id", "gw-served-42")

	ctx, trace, _, _, _ := srv.beginServedRequest(rec, req)
	if trace != "gw-served-42" {
		t.Fatalf("minted trace = %q, want gw-served-42", trace)
	}
	if got := agent.RequestTraceID(ctx); got != trace {
		t.Fatalf("ctx request trace = %q, want the minted %q stamped at the served boundary", got, trace)
	}
	// And the planner's getter — read through the SAME seam — resolves it. (nativePhaseTraceID
	// is unexported in package agent, so this asserts the public accessor half here and the
	// planner half in internal/agent/reachability_13120_test.go.)
	if got := agent.RequestTraceID(ctx); got == "" {
		t.Fatal("served context carries no request trace id")
	}
}
