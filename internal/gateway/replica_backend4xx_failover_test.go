package gateway

// Reconnaissance witness for issue #13498: "a backend 4xx (405) becomes the
// router's own response instead of failing over."
//
// Finding: the BUFFERED router path (ReplicaDispatch.completeReserved) DOES fail
// over on a backend 405 once the live FleetMembership is ARMED — which production
// `fak serve` does in runFleetHealthLoop after its first probe. The defect is the
// STREAMING path (ReplicaDispatch.CompleteStream), which has no retarget loop at
// all, so a 405 from the selected replica is terminal even with the fleet armed.
//
// The unarmed case (a plain New(Config)+Handler() test, as prescribed in the issue)
// also surfaces terminally, but only because route.reservation == nil short-circuits
// completeReserved until the Serve-time health loop arms membership.

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

func registerBackend4xxTestABI() {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})
}

const fleetCompletionBody = `{"model":"fleet-model","choices":[{"message":{"role":"assistant","content":"secondary-ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`

func newBackend4xxGateway(t *testing.T, primary, secondary *httptest.Server) *Server {
	t.Helper()
	registerBackend4xxTestABI()
	srv, err := New(Config{
		EngineID:        "test",
		Model:           "fleet-model",
		BaseURL:         primary.URL,
		ReplicaBaseURLs: []string{secondary.URL},
		Provider:        "openai",
		VDSO:            true,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv
}

func armFleet(t *testing.T, srv *Server) {
	t.Helper()
	if srv.fleet == nil || srv.fleetRouter == nil {
		t.Fatal("setup: fleet/fleetRouter not wired by New")
	}
	srv.fleet.ProbeOnce(context.Background())
	srv.fleetRouter.WithMembership(srv.fleet)
}

// TestReplicaDispatchBackend405FailsOverUnarmed is the issue's prescribed scenario
// (Config{BaseURL, ReplicaBaseURLs} behind Handler(), no Serve fence). It asserts the
// ACTUAL observation: the reservation is nil, so completeReserved does not retarget and
// the 405 is surfaced terminally. Production arms membership at Serve, so this is the
// startup-race window only — see the ARMED test for the steady-state behavior.
func TestReplicaDispatchBackend405FailsOverUnarmed(t *testing.T) {
	var primaryHits, secondaryHits atomic.Int32
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		primaryHits.Add(1)
		w.WriteHeader(http.StatusMethodNotAllowed)
	}))
	defer primary.Close()
	secondary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		secondaryHits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, fleetCompletionBody)
	}))
	defer secondary.Close()

	srv := newBackend4xxGateway(t, primary, secondary)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var resp ChatResponse
	code := postJSON(t, ts.URL+"/v1/chat/completions", ChatRequest{
		Model:    "fleet-model",
		Messages: []agent.Message{{Role: agent.RoleUser, Content: "hi"}},
	}, &resp)

	t.Logf("OBSERVED(unarmed) status=%d primaryHits=%d secondaryHits=%d",
		code, primaryHits.Load(), secondaryHits.Load())

	if code != http.StatusMethodNotAllowed || primaryHits.Load() != 1 || secondaryHits.Load() != 0 {
		t.Fatalf("unarmed expectation changed: status=%d primaryHits=%d secondaryHits=%d (want 405/1/0)",
			code, primaryHits.Load(), secondaryHits.Load())
	}
}

// TestReplicaDispatchBackend405FailsOverArmed proves the BUFFERED path is already
// correct: with the fleet armed exactly as runFleetHealthLoop does, the 405 retargets
// the reserved turn to the secondary replica and the client sees 200.
func TestReplicaDispatchBackend405FailsOverArmed(t *testing.T) {
	var primaryHits, secondaryHits atomic.Int32
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		primaryHits.Add(1)
		w.WriteHeader(http.StatusMethodNotAllowed)
	}))
	defer primary.Close()
	secondary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		secondaryHits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, fleetCompletionBody)
	}))
	defer secondary.Close()

	srv := newBackend4xxGateway(t, primary, secondary)
	armFleet(t, srv)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var resp ChatResponse
	code := postJSON(t, ts.URL+"/v1/chat/completions", ChatRequest{
		Model:    "fleet-model",
		Messages: []agent.Message{{Role: agent.RoleUser, Content: "hi"}},
	}, &resp)

	t.Logf("OBSERVED(armed) status=%d primaryHits=%d secondaryHits=%d",
		code, primaryHits.Load(), secondaryHits.Load())

	if code != http.StatusOK || secondaryHits.Load() == 0 {
		t.Fatalf("DEFECT: armed buffered 405 did not fail over (status=%d primaryHits=%d secondaryHits=%d)",
			code, primaryHits.Load(), secondaryHits.Load())
	}
}

// TestReplicaDispatchCompleteStreamBackend405FailsOver is the defect witness. The
// streaming router CompleteStream has no retarget loop, so with the fleet ARMED a
// backend 405 on the selected replica is still terminal. Expected to FAIL until
// CompleteStream mirrors completeReserved's retarget loop.
func TestReplicaDispatchCompleteStreamBackend405FailsOver(t *testing.T) {
	sseOK := func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"secondary-ok\"},\"finish_reason\":null}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}

	var primaryHits, secondaryHits atomic.Int32
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		primaryHits.Add(1)
		w.WriteHeader(http.StatusMethodNotAllowed)
	}))
	defer primary.Close()
	secondary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		secondaryHits.Add(1)
		sseOK(w)
	}))
	defer secondary.Close()

	srv := newBackend4xxGateway(t, primary, secondary)
	armFleet(t, srv)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := []byte(`{"model":"fleet-model","messages":[{"role":"user","content":"hi"}],"stream":true}`)
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)

	t.Logf("OBSERVED(stream armed) status=%d primaryHits=%d secondaryHits=%d body=%s",
		resp.StatusCode, primaryHits.Load(), secondaryHits.Load(), raw)

	if resp.StatusCode == http.StatusMethodNotAllowed || strings.Contains(string(raw), "upstream_request_rejected") {
		t.Fatalf("DEFECT: streaming backend 405 surfaced terminally (status=%d primaryHits=%d secondaryHits=%d): %s",
			resp.StatusCode, primaryHits.Load(), secondaryHits.Load(), raw)
	}
	if secondaryHits.Load() == 0 {
		t.Fatalf("streaming 405 did not retarget to secondary (primaryHits=%d)", primaryHits.Load())
	}
}

// failingAfterFragmentPlanner emits one content fragment to the sink, then fails
// with a retriable error — the mid-stream shape where retargeting would duplicate
// content already on the client's wire.
type failingAfterFragmentPlanner struct {
	name   string
	stream bool
	err    error
}

func (p *failingAfterFragmentPlanner) Model() string            { return p.name }
func (p *failingAfterFragmentPlanner) StreamingSupported() bool { return p.stream }
func (p *failingAfterFragmentPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return nil, p.err
}
func (p *failingAfterFragmentPlanner) CompleteStream(_ context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	if sink != nil {
		if err := sink("partial-answer"); err != nil {
			return nil, err
		}
	}
	return nil, p.err
}

// TestReplicaDispatchCompleteStreamDoesNotRetargetAfterEmit is the SAFETY half of the
// #13498 streaming fix: once a single content fragment has reached the caller's sink,
// a later streaming failure must NOT retarget to another replica — replaying the turn
// would duplicate the already-delivered prefix on the wire. The retarget loop is armed
// only while the sink is still clean.
func TestReplicaDispatchCompleteStreamDoesNotRetargetAfterEmit(t *testing.T) {
	boom := errors.New("upstream died mid-stream")
	primary := &failingAfterFragmentPlanner{name: "primary", stream: true, err: boom}
	secondary := &replicaDispatchTestPlanner{name: "secondary", streaming: true, streamingSupported: true}
	router, err := NewReplicaDispatch("fleet", []PlannerReplica{
		{Name: "primary", Planner: primary},
		{Name: "secondary", Planner: secondary},
	})
	if err != nil {
		t.Fatalf("NewReplicaDispatch: %v", err)
	}

	var deltas []string
	_, err = router.CompleteStream(context.Background(), func(delta string) error {
		deltas = append(deltas, delta)
		return nil
	}, nil, nil)
	if !errors.Is(err, boom) {
		t.Fatalf("mid-stream error = %v, want the original %v (no retarget after emit)", err, boom)
	}
	if len(deltas) != 1 || deltas[0] != "partial-answer" {
		t.Fatalf("sink deltas = %v, want exactly the one emitted fragment", deltas)
	}
	if _, streams := secondary.counts(); streams != 0 {
		t.Fatalf("secondary was retried after a partial emit (streamN=%d) — this duplicates content", streams)
	}
}
