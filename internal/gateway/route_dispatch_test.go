package gateway

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// markerEngine returns a fixed engine-id marker in the result Meta so a test can
// prove WHICH engine the kernel dispatched to (vs the default echo engine).
type markerEngine struct{ id string }

func (markerEngine) Caps() []abi.Capability { return nil }
func (e markerEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	return &abi.Result{Call: c, Status: abi.StatusOK, Payload: c.Args,
		Meta: map[string]string{"engine": e.id}}, nil
}

// routeServer wires an isolated chain WITH the real engine-residency gate (rank 12)
// installed plus the optional routing manifest, and returns a ready Server. Mirrors
// newTestServer but adds the residency PDP so a routed-to-remote call is adjudicated
// by the SAME gate production runs. Not parallel-safe (mutates the global registry).
func routeServer(t *testing.T, m *modelroute.Manifest) *Server {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{}) // the kernel-default local engine
	abi.RegisterEngine("routed2", markerEngine{"routed2"})
	abi.RegisterAdjudicator(0, toolAdj{}) // allow*/deny* by tool-name prefix
	engine.RegisterResidencyGate()        // the REAL residency floor (rank 12)
	srv, err := New(Config{EngineID: "test", Model: "m", VDSO: true, RouteManifest: m})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv
}

// pickManifest routes one tool to a single-model PICK, with a local default.
func pickManifest(tool, model string) *modelroute.Manifest {
	return &modelroute.Manifest{
		Default: modelroute.Plan{Members: []modelroute.Member{{Model: "test"}}},
		Rules: []modelroute.Rule{{
			Name:  "rule-" + tool,
			Match: modelroute.Match{Tool: tool},
			Plan:  modelroute.Plan{Members: []modelroute.Member{{Model: model}}},
		}},
	}
}

// A matched single-model rule writes the chosen model to ToolCall.Engine, and it is
// set by buildCall — i.e. BEFORE the call is ever submitted (the residency contract).
func TestRoutePickSetsEnginePreSubmit(t *testing.T) {
	s := routeServer(t, pickManifest("read_doc", "remote:openai"))
	tc, err := s.buildCall(context.Background(), "read_doc", `{"q":1}`, true, "", "")
	if err != nil {
		t.Fatalf("buildCall: %v", err)
	}
	if tc.Engine != "remote:openai" {
		t.Fatalf("route not written to Engine pre-submit: got %q, want remote:openai", tc.Engine)
	}
}

// THE load-bearing test: a sensitive-labeled call routed (pre-submit) to a remote
// model is DENIED by the real residency gate. A non-sensitive call on the same route
// is NOT denied by residency — proving the gate keys on sensitivity, not on the
// route alone, and that the route reached the adjudication fold.
func TestRouteSensitiveRemoteDeniedByResidency(t *testing.T) {
	s := routeServer(t, pickManifest("fetch", "remote:openai"))
	ctx := context.Background()

	// Sensitive: tenant payload bound for a remote model -> DENY at adjudication.
	tc, err := s.buildCall(ctx, "fetch", `{"id":7}`, true, "", "")
	if err != nil {
		t.Fatalf("buildCall: %v", err)
	}
	if tc.Engine != "remote:openai" {
		t.Fatalf("route precondition: Engine=%q, want remote:openai", tc.Engine)
	}
	tc.Meta["sensitivity"] = "tenant" // the sensitive-labeled subject
	_, v := s.k.Syscall(ctx, tc)
	if v.Kind != abi.VerdictDeny || v.By != "engine-residency" {
		t.Fatalf("sensitive->remote must be denied by engine-residency, got kind=%v by=%q", v.Kind, v.By)
	}
	if v.Reason != abi.ReasonTrustViolation {
		t.Fatalf("residency deny reason = %v, want TRUST_VIOLATION", v.Reason)
	}

	// Non-sensitive on the SAME remote route: residency must NOT be the denier.
	tc2, err := s.buildCall(ctx, "fetch", `{"id":8}`, true, "", "")
	if err != nil {
		t.Fatalf("buildCall: %v", err)
	}
	_, v2 := s.k.Syscall(ctx, tc2)
	if v2.By == "engine-residency" {
		t.Fatalf("a non-sensitive call must not be denied by residency, got %+v", v2)
	}
}

// With no manifest the gateway leaves Engine unset (the kernel default) — byte-for-
// byte the pre-routing behavior.
func TestRouteBackCompatNoManifest(t *testing.T) {
	s := routeServer(t, nil)
	tc, err := s.buildCall(context.Background(), "anything", `{}`, false, "", "")
	if err != nil {
		t.Fatalf("buildCall: %v", err)
	}
	if tc.Engine != "" {
		t.Fatalf("no manifest must leave Engine empty (kernel default), got %q", tc.Engine)
	}
}

// An ENSEMBLE plan is NOT collapsed to a single member here — the gateway leaves
// Engine unset and defers the N-submit fan-out to issue #597. Collapsing would be a
// silent wrong route, so the route stays the kernel default until #597 lands.
func TestRouteEnsembleNotCollapsed(t *testing.T) {
	m := &modelroute.Manifest{
		Default: modelroute.Plan{Members: []modelroute.Member{{Model: "test"}}},
		Rules: []modelroute.Rule{{
			Name:  "guard-write",
			Match: modelroute.Match{Tool: "risky_write"},
			Plan: modelroute.Plan{
				Members: []modelroute.Member{{Model: "guard-a"}, {Model: "guard-b"}},
				Reduce:  modelroute.ReduceVote,
			},
		}},
	}
	s := routeServer(t, m)
	tc, err := s.buildCall(context.Background(), "risky_write", `{}`, false, "", "")
	if err != nil {
		t.Fatalf("buildCall: %v", err)
	}
	if tc.Engine != "" {
		t.Fatalf("ensemble must not collapse to one member; Engine=%q, want empty (deferred to #597)", tc.Engine)
	}
}

// End-to-end: an allowed, non-sensitive call routed to a registered model dispatches
// to THAT engine (not the kernel default) — proving the pre-submit route drives the
// kernel's engine selection, not just the adjudication.
func TestRouteDispatchesToRoutedEngine(t *testing.T) {
	s := routeServer(t, pickManifest("allow_run", "routed2"))
	ctx := context.Background()
	tc, err := s.buildCall(ctx, "allow_run", `{"x":1}`, false, "", "")
	if err != nil {
		t.Fatalf("buildCall: %v", err)
	}
	if tc.Engine != "routed2" {
		t.Fatalf("route precondition: Engine=%q, want routed2", tc.Engine)
	}
	r, v := s.k.Syscall(ctx, tc)
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("expected ALLOW, got %v (%v)", v.Kind, v.Reason)
	}
	if r == nil || r.Meta["engine"] != "routed2" {
		t.Fatalf("call must dispatch to the routed engine; result meta = %v", r.Meta)
	}
}
