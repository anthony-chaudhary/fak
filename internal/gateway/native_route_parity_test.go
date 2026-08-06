package gateway

// native_route_parity_test.go — the #5644 witness: a `serve --native` turn's in-loop tool
// calls resolve their engine route through the SAME manifest→roster→principal chain the
// proxying path resolves it through, so two serving modes of one binary can no longer
// bind two different routes for the same call.
//
// Before this landed, nativeRunOptions passed WithRouteManifest ONLY: the owned loop bound
// the abstract plan member ("guard-a") while the proxy path bound the account-resolved
// Target.EngineRoute() ("openai:openai-work/gpt-5.5"). Nothing errored — the divergence was
// silent, and a residency PDP adjudicating a native turn read a manifest route where the
// proxy path would have shown it an account-bound one.
//
// Both sides are the REAL seams. The proxy route is whatever buildCall binds to
// abi.ToolCall.Engine pre-submit. The native route is captured by a recording ADJUDICATOR
// from the live agent.RunArm the served turn actually drives — the same pre-Submit seam the
// residency PDP reads, so this witnesses the route the kernel saw rather than re-deriving
// one the test hopes the loop would compute.
//
// Deliberately out of frame: these manifests Match on TOOL only. The gateway classifies with
// Subject.Labels (read_only / sensitivity / tenant) while the agent loop classifies with no
// labels at all, so a LABEL-matching manifest still diverges at the manifest stage. That is a
// separate defect from the roster one #5644 names, and pinning it here would silently widen
// this witness past what it proves.

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// routedTool is a tau2 airline tool the deterministic MockPlanner calls, so the owned loop
// reaches adjudication under the real agent policy rather than a hand-allowed stub. It is
// deliberately WRITE-shaped: the vDSO's tier-2 store is process-global and the agent loop
// does not lower an isolation principal onto its calls, so a read-shaped tool
// (get_user_details) is deduped across tests in this package and the second test's call
// never reaches the adjudicator chain at all. A write-shaped tool is never served from
// cache, so every test here observes its own live route binding.
const routedTool = "book_flight"

// engineRecorder captures the Engine bound to each tool call at ADJUDICATION time — the
// pre-Submit seam the residency PDP reads. It always DEFERs so the real policy rung still
// decides the verdict; this rung only observes.
type engineRecorder struct {
	mu   sync.Mutex
	seen map[string]string
}

func newEngineRecorder() *engineRecorder { return &engineRecorder{seen: map[string]string{}} }

func (*engineRecorder) Caps() []abi.Capability { return nil }

func (r *engineRecorder) Adjudicate(_ context.Context, c *abi.ToolCall) abi.Verdict {
	r.mu.Lock()
	if _, dup := r.seen[c.Tool]; !dup {
		r.seen[c.Tool] = c.Engine
	}
	r.mu.Unlock()
	return abi.Verdict{Kind: abi.VerdictDefer, By: "engine-recorder"}
}

// route returns the Engine recorded for a tool and whether the call ever reached the
// kernel at all. A refused route never dispatches, so absent is the fail-closed witness.
func (r *engineRecorder) route(tool string) (string, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.seen[tool]
	return e, ok
}

// all renders every (tool -> engine) the chain observed, for failure messages.
func (r *engineRecorder) all() map[string]string {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make(map[string]string, len(r.seen))
	for k, v := range r.seen {
		out[k] = v
	}
	return out
}

// nativeParityServer wires a `--native` Server over an optional manifest + roster, plus a
// recording adjudicator and an engine registered at each route a resolved call may land on.
// Config.Native is the bit under test: ownsSessionLoop() is what decides a served turn runs
// the owned loop at all. Not parallel-safe (mutates the global registry).
func nativeParityServer(t *testing.T, m *modelroute.Manifest, roster *modelroute.Roster, routes ...string) (*Server, *engineRecorder) {
	t.Helper()
	abi.ResetForTest()
	// Configure registers the localtools engine + the agent policy RunArm(fak=true) runs
	// under; the region backend is what the syscall Ref resolver needs after a reset.
	agent.Configure()
	abi.RegisterRegionBackend(inlineBackend{})
	for _, route := range routes {
		abi.RegisterEngine(route, markerEngine{route})
	}
	rec := newEngineRecorder()
	abi.RegisterAdjudicator(0, rec)

	srv, err := New(Config{
		EngineID: "localtools", Model: "test-model", VDSO: true,
		Native: true, NativeMaxTurns: 8,
		RouteManifest: m, RouteAccounts: roster,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	srv.planner = agent.NewMockPlanner("test-model")
	return srv, rec
}

// driveNativeTurn runs one served turn through the OWNED loop — the same runNativeArm the
// --native /v1/messages handler calls — so the recorder observes the routes that loop bound.
func driveNativeTurn(t *testing.T, s *Server, ctx context.Context) {
	t.Helper()
	req := &agent.AnthropicMessagesRequest{
		Model:     "test-model",
		MaxTokens: 256,
		Messages:  []agent.Message{{Role: agent.RoleUser, Content: "Book me a direct flight."}},
	}
	if _, err := s.runNativeArm(ctx, req, "parity-trace"); err != nil {
		t.Fatalf("runNativeArm: %v", err)
	}
}

// proxyRoute is the engine route the PROXY path binds pre-submit for the same call —
// literally buildCall's tc.Engine, the value the residency PDP reads inside the fold.
func proxyRoute(s *Server, ctx context.Context, tool string) (string, error) {
	tc, err := s.buildCall(ctx, tool, `{"user_id":"mia_li_3668","flight_id":"HAT001"}`, false, "", "")
	if err != nil {
		return "", err
	}
	return tc.Engine, nil
}

// THE headline witness: one roster, one tool call, both paths, equal resolved routes — and
// the shared answer is the ACCOUNT-resolved route, not the bare plan member. The last
// assertion is the one that goes red on the pre-#5644 tree, where the native loop bound
// "guard-a".
func TestNativeLoopResolvesRouteLikeProxyPath(t *testing.T) {
	s, rec := nativeParityServer(t, pickManifest(routedTool, "guard-a"), parityRoster(), routedRoute, defaultRoute)
	const want = routedRoute
	ctx := WithPrincipal(context.Background(), "tenant-parity")

	gotProxy, err := proxyRoute(s, ctx, routedTool)
	if err != nil {
		t.Fatalf("proxy path: %v", err)
	}
	driveNativeTurn(t, s, ctx)
	gotNative, dispatched := rec.route(routedTool)
	if !dispatched {
		t.Fatalf("the owned loop never dispatched %q — no route to compare; recorder saw %v", routedTool, rec.all())
	}
	if gotNative != gotProxy {
		t.Fatalf("route divergence between serving modes: native=%q proxy=%q — a --native turn "+
			"must resolve the same engine route the proxying path resolves (#5644)", gotNative, gotProxy)
	}
	if gotProxy != want {
		t.Fatalf("proxy route = %q, want %q (the account-resolved EngineRoute)", gotProxy, want)
	}
	if gotNative == "guard-a" {
		t.Fatalf("the owned loop bound the bare plan member %q — the account roster was skipped, "+
			"which is exactly the #5644 defect", gotNative)
	}
}

// The compatibility arm the done-condition names: with NO roster configured both paths
// behave exactly as they do at HEAD — the abstract plan member IS the route.
func TestNativeLoopRouteParityWithoutRoster(t *testing.T) {
	s, rec := nativeParityServer(t, pickManifest(routedTool, "guard-a"), nil, "guard-a", "test")
	ctx := WithPrincipal(context.Background(), "tenant-noroster")

	gotProxy, err := proxyRoute(s, ctx, routedTool)
	if err != nil {
		t.Fatalf("proxy path: %v", err)
	}
	driveNativeTurn(t, s, ctx)
	gotNative, dispatched := rec.route(routedTool)
	if !dispatched {
		t.Fatalf("the owned loop never dispatched %q; recorder saw %v", routedTool, rec.all())
	}
	if gotNative != gotProxy || gotProxy != "guard-a" {
		t.Fatalf("no-roster parity broken: native=%q proxy=%q, want both %q (the pre-roster route)",
			gotNative, gotProxy, "guard-a")
	}
}

// PRECEDENCE, asserted rather than only documented: the manifest decides WHICH abstract id,
// the roster decides WHICH account binds it. They are not alternatives — a roster with no
// manifest binds nothing, because the roster is only consulted for an id the manifest
// already PICKed, so both paths fall through to the kernel default ("").
func TestNativeLoopRoutePrecedenceRosterNeedsManifest(t *testing.T) {
	s, rec := nativeParityServer(t, nil, parityRoster())
	ctx := WithPrincipal(context.Background(), "tenant-precedence")

	gotProxy, err := proxyRoute(s, ctx, routedTool)
	if err != nil {
		t.Fatalf("proxy path: %v", err)
	}
	driveNativeTurn(t, s, ctx)
	gotNative, dispatched := rec.route(routedTool)
	if !dispatched {
		t.Fatalf("the owned loop never dispatched %q; recorder saw %v", routedTool, rec.all())
	}
	if gotNative != "" || gotProxy != "" {
		t.Fatalf("a roster with no manifest must bind nothing on either path: native=%q proxy=%q, want both \"\"",
			gotNative, gotProxy)
	}
}

// The two routes a parityRoster resolves to, and the manifest ids that bind to them.
const (
	routedRoute  = "openai:openai-work/gpt-5.5" // what routedTool ("guard-a") resolves to
	defaultRoute = "local:local-box/up-test"    // what the manifest's DEFAULT plan resolves to
)

// parityRoster binds BOTH the routed id and the manifest's default plan member. Binding the
// default matters: pickManifest sends every UNMATCHED tool to the "test" member, and a
// roster is fail-loud on an id it cannot resolve — so a roster that bound only "guard-a"
// would refuse every other tool in the MockPlanner's flow and the loop would never reach
// routedTool at all. principals (when given) restrict the OpenAI account only, arming the
// residency arm of the keyset (#5332) that resolveRoute enforces and the loop's mirror must
// enforce too; the local account stays unrestricted so the surrounding flow still runs.
func parityRoster(principals ...string) *modelroute.Roster {
	return &modelroute.Roster{
		Version: modelroute.RosterVersion,
		Accounts: []modelroute.Account{
			{ID: "openai-work", Kind: modelroute.KindOpenAI, CredEnv: "OPENAI_WORK_API_KEY", Principals: principals},
			{ID: "local-box", Kind: modelroute.KindLocal, BaseURL: "http://127.0.0.1:11434/v1"},
		},
		Bindings: []modelroute.Binding{
			{Model: "guard-a", Account: "openai-work", UpstreamModel: "gpt-5.5"},
			{Model: "test", Account: "local-box", UpstreamModel: "up-test"},
		},
	}
}

// A principal the account ADMITS resolves identically on both paths — the roster binding is
// honored, not merely tolerated.
func TestNativeLoopRouteParityAdmittedPrincipal(t *testing.T) {
	s, rec := nativeParityServer(t, pickManifest(routedTool, "guard-a"), parityRoster("acme"), routedRoute, defaultRoute)
	const want = routedRoute
	ctx := WithPrincipal(context.Background(), "acme")

	gotProxy, err := proxyRoute(s, ctx, routedTool)
	if err != nil {
		t.Fatalf("proxy path: %v", err)
	}
	driveNativeTurn(t, s, ctx)
	gotNative, dispatched := rec.route(routedTool)
	if !dispatched {
		t.Fatalf("the owned loop never dispatched %q for an ADMITTED principal; recorder saw %v", routedTool, rec.all())
	}
	if gotNative != gotProxy || gotProxy != want {
		t.Fatalf("admitted principal: native=%q proxy=%q, want both %q", gotNative, gotProxy, want)
	}
}

// The arm that makes wiring the roster SAFE rather than merely consistent: a principal the
// account does NOT admit is refused on both paths, and the owned loop's call never reaches
// the kernel at all. Resolving the roster WITHOUT carrying the principal would have let a
// native turn bind a route the proxy path refuses — handing one tenant a route through
// another tenant's account, a worse divergence than the one #5644 set out to close.
func TestNativeLoopRouteParityRefusesUnadmittedPrincipal(t *testing.T) {
	// "evil-corp" is a principal no other test in this package dispatches under, which
	// matters: the vDSO's tier-2 store is process-global and scoped per principal, so a
	// shared principal could serve this call from a SIBLING test's fill and produce
	// "never dispatched" for a reason that has nothing to do with the route refusal.
	// A private principal makes the absence below mean what it claims to mean.
	const unadmitted = "evil-corp"
	s, rec := nativeParityServer(t, pickManifest(routedTool, "guard-a"), parityRoster("acme"), routedRoute, defaultRoute)
	ctx := WithPrincipal(context.Background(), unadmitted)

	gotProxy, proxyErr := proxyRoute(s, ctx, routedTool)
	if proxyErr == nil {
		t.Fatalf("proxy path admitted principal %q to a restricted account (route %q)", unadmitted, gotProxy)
	}
	if !strings.Contains(proxyErr.Error(), "not admitted") || !strings.Contains(proxyErr.Error(), "openai-work") {
		t.Fatalf("proxy refusal should name the account and read as a principal refusal, got: %v", proxyErr)
	}

	// The owned loop reports a route refusal IN-BAND (the model sees a structured error and
	// may adapt), so the run itself still completes; the load-bearing assertion is that the
	// refused call never reached the kernel.
	driveNativeTurn(t, s, ctx)
	if gotNative, dispatched := rec.route(routedTool); dispatched {
		t.Fatalf("the owned loop dispatched %q as principal %q with route %q — it must fail "+
			"closed exactly where the proxy path does (#5332/#5644)", routedTool, unadmitted, gotNative)
	}
}

// The EMPTY principal — an unattributed caller (no keyset, or the single --require-key-env
// bearer) — must not inherit a restricted account's credential either. This is asserted on
// the proxy path only, and deliberately so: the "never dispatched" witness the test above
// uses is not sound for principal "", because the process-global vDSO is scoped per
// principal and every sibling test in this package that drives the MockPlanner calls the
// same tool with the same args under the same empty principal. Proving the loop ENFORCES
// Admits is the job of the test above; this pins the empty-principal verdict itself, which
// both paths read from the one shared modelroute.Target.Admits.
func TestNativeLoopRouteParityRefusesUnattributedCaller(t *testing.T) {
	s, _ := nativeParityServer(t, pickManifest(routedTool, "guard-a"), parityRoster("acme"), routedRoute, defaultRoute)

	gotProxy, err := proxyRoute(s, context.Background(), routedTool)
	if err == nil {
		t.Fatalf("an unattributed caller must not inherit a restricted account (route %q)", gotProxy)
	}
	if !strings.Contains(err.Error(), "<unattributed>") {
		t.Fatalf("the refusal should distinguish an unattributed caller from a wrong tenant, got: %v", err)
	}
}
