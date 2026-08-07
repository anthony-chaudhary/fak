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
	"github.com/anthony-chaudhary/fak/internal/vdso"
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
	// ledger appends EVERY adjudicated call in order, where seen keeps only the first
	// binding per tool. The count is the load-bearing part for #5707: a call served from
	// a vDSO tier-2 entry never reaches the chain at all, so "how many times did this
	// (tool,args) reach the kernel" is exactly the cache-scoping witness.
	ledger []recordedCall
}

// recordedCall is one adjudicated tool call as the kernel saw it: the route bound
// pre-Submit, and the tenant ISOLATION principal the caller lowered onto the call.
type recordedCall struct {
	Tool      string
	Engine    string
	Principal string
}

func newEngineRecorder() *engineRecorder { return &engineRecorder{seen: map[string]string{}} }

func (*engineRecorder) Caps() []abi.Capability { return nil }

func (r *engineRecorder) Adjudicate(_ context.Context, c *abi.ToolCall) abi.Verdict {
	r.mu.Lock()
	if _, dup := r.seen[c.Tool]; !dup {
		r.seen[c.Tool] = c.Engine
	}
	r.ledger = append(r.ledger, recordedCall{Tool: c.Tool, Engine: c.Engine, Principal: c.Meta[vdso.MetaPrincipal]})
	r.mu.Unlock()
	return abi.Verdict{Kind: abi.VerdictDefer, By: "engine-recorder"}
}

// count reports how many times a tool actually reached the adjudicator chain. A vDSO
// tier-2 hit is served without a syscall, so a flat count across two callers means the
// second was served from the first's fill.
func (r *engineRecorder) count(tool string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	n := 0
	for _, rc := range r.ledger {
		if rc.Tool == tool {
			n++
		}
	}
	return n
}

// principalsOf returns, in order, the isolation principal carried by each adjudicated
// call for one tool — the direct read of what the loop lowered onto the call.
func (r *engineRecorder) principalsOf(tool string) []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []string
	for _, rc := range r.ledger {
		if rc.Tool == tool {
			out = append(out, rc.Principal)
		}
	}
	return out
}

// records renders the full ordered ledger for failure messages.
func (r *engineRecorder) records() []recordedCall {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]recordedCall, len(r.ledger))
	copy(out, r.ledger)
	return out
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

// ---------------------------------------------------------------------------
// #5707 — the CACHE half of the same principal boundary. Routing is out of frame
// below: the seam under test is abi.ToolCall.Meta, not abi.ToolCall.Engine.
// ---------------------------------------------------------------------------

// scopedReadTool is the READ-shaped tau2 tool the #5707 witness drives, and scopedReadArgs
// the one byte-identical arg set both tenants call it with — the (tool,args) pair the
// vDSO's tier-2 store is built to dedup, and therefore the pair a missing principal leaks
// across tenants.
const (
	scopedReadTool = "get_user_details"
	scopedReadArgs = `{"user_id":"mia_li_3668"}`
)

// readOnlyPlanner drives ONE read-shaped tool twice with identical args, then finishes.
//
// It replaces the MockPlanner for the #5707 witness for a load-bearing reason: the
// MockPlanner's scripted flow ENDS in a write (book_flight), and a write advances the
// world version that invalidates every pooled read — so each run's tier-2 fill is
// destroyed by that same run before the next tenant's turn begins, and the cross-tenant
// leak cannot be observed through it at all. A read-only flow is also the shape the leak
// actually matters in: a read-heavy multi-tenant serving session, where one tenant's fill
// is exactly what the next tenant would be served.
//
// The second, identical read is the intra-run dedup the fak arm is measured on, so one
// planner witnesses both halves: the dedup must survive, partitioned per principal.
type readOnlyPlanner struct{}

func (readOnlyPlanner) Model() string { return "test-model" }

func (readOnlyPlanner) Complete(_ context.Context, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	reads := 0
	for _, m := range messages {
		if m.Role == agent.RoleTool && m.Name == scopedReadTool {
			reads++
		}
	}
	if reads >= 2 {
		return &agent.Completion{
			Message:      agent.Message{Role: agent.RoleAssistant, Content: "Read the account twice; nothing to change."},
			FinishReason: "stop",
		}, nil
	}
	return &agent.Completion{
		Message: agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{
			ID: "call_" + itoa(uint64(reads)), Type: "function",
			Function: agent.Func{Name: scopedReadTool, Arguments: scopedReadArgs},
		}}},
		FinishReason: "tool_calls",
	}, nil
}

// vdsoScopeServer is a --native Server whose owned loop drives the read-only flow above.
// Routing is deliberately unwired (no manifest, no roster): the seam under test is
// abi.ToolCall.Meta, not abi.ToolCall.Engine.
func vdsoScopeServer(t *testing.T) (*Server, *engineRecorder) {
	t.Helper()
	s, rec := nativeParityServer(t, nil, nil)
	s.planner = readOnlyPlanner{}
	return s, rec
}

// THE #5707 witness: one tool, one arg set, two principals, both driven through the OWNED
// loop — and the second principal's read must reach the kernel rather than be served from
// the first's fill.
//
// The counting witness is deliberately a DELTA, not an absolute. The vDSO's tier-2 store
// is process-global and survives abi.ResetForTest, so whether tenantA's own read reaches
// the kernel depends on what sibling tests in this package filled first. The delta does
// not: on the pre-#5707 tree the loop lowers NO principal, so every tenant shares ONE
// entry and tenantB's read adds nothing to the ledger whether or not tenantA's did —
// which is precisely the leak. With the principal lowered, each tenant fills and reads
// its own scope, so the delta is exactly one dispatch per tenant.
func TestNativeLoopScopesVdsoPerPrincipal(t *testing.T) {
	s, rec := vdsoScopeServer(t)
	// Principals no sibling test in this package dispatches under, so the scopes these
	// turns fill are this test's alone.
	const tenantA, tenantB = "vdso-scope-alpha", "vdso-scope-beta"

	driveNativeTurn(t, s, WithPrincipal(context.Background(), tenantA))
	afterA := rec.count(scopedReadTool)

	driveNativeTurn(t, s, WithPrincipal(context.Background(), tenantB))
	afterB := rec.count(scopedReadTool)

	if afterB == afterA {
		t.Fatalf("CROSS-TENANT vDSO LEAK (#5707): principal %q read %q with the same args as %q and "+
			"the call never reached the kernel — it was served from a tier-2 entry filled under a "+
			"DIFFERENT principal (%d adjudications before %q's turn, %d after). The owned loop must "+
			"lower its principal onto ToolCall.Meta[%q] exactly as the proxy path's buildCall does, "+
			"so one tenant can neither be served from nor fill another's entry. Ledger: %v",
			tenantB, scopedReadTool, tenantA, afterA, tenantB, afterB, vdso.MetaPrincipal, rec.records())
	}
	// Scoping must not become "cache nothing". The MockPlanner reads scopedReadTool TWICE
	// per run with identical args; the second is the vDSO hit the fak arm is measured on.
	// One dispatch per tenant per run is the fix; two would mean the dedup was disabled
	// rather than partitioned.
	if got := afterB - afterA; got != 1 {
		t.Fatalf("principal %q dispatched %q %d times in one turn, want exactly 1: the loop reads it "+
			"twice with identical args, so a second dispatch means per-principal scoping DISABLED "+
			"the tier-2 dedup instead of partitioning it. Ledger: %v", tenantB, scopedReadTool, got, rec.records())
	}
	// The direct read of the fix: every dispatch this test observed is attributed, and the
	// one tenantB's turn added is attributed to tenantB — not inherited from tenantA.
	principals := rec.principalsOf(scopedReadTool)
	if len(principals) == 0 || principals[len(principals)-1] != tenantB {
		t.Fatalf("the call %q dispatched during %q's turn carried principal %v, want the last to be %q — "+
			"an unattributed or wrong-tenant call is what makes one tier-2 entry readable by two tenants",
			scopedReadTool, tenantB, principals, tenantB)
	}
	for i, p := range principals {
		if p != tenantA && p != tenantB {
			t.Fatalf("dispatch %d of %q carried principal %q, want one of {%q,%q}: an unscoped call "+
				"re-opens the shared entry for every later tenant. Ledger: %v",
				i, scopedReadTool, p, tenantA, tenantB, rec.records())
		}
	}
	// REFUTATION: tenantA drives the same read a THIRD time. If per-principal scoping is
	// real, A's own fill still serves A and nothing new reaches the kernel. A rise here
	// would mean the key had gone per-CALL (correctness by cache-busting, which would also
	// destroy the hit/miss timing property the scope exists to protect).
	driveNativeTurn(t, s, WithPrincipal(context.Background(), tenantA))
	if afterA2 := rec.count(scopedReadTool); afterA2 != afterB {
		t.Fatalf("principal %q re-read %q and dispatched again (%d -> %d): its own tier-2 entry must "+
			"still serve it, or the scope is busting the cache rather than partitioning it. Ledger: %v",
			tenantA, scopedReadTool, afterB, afterA2, rec.records())
	}
}

// The compatibility arm the #5707 done-condition names: with NO principal configured the
// loop behaves exactly as it does at HEAD — single-tenant, every caller sharing one scope,
// and no principal key written onto the call at all.
func TestNativeLoopVdsoScopeUnchangedWithoutPrincipal(t *testing.T) {
	s, rec := vdsoScopeServer(t)

	driveNativeTurn(t, s, context.Background())
	afterFirst := rec.count(scopedReadTool)
	driveNativeTurn(t, s, context.Background())

	if afterSecond := rec.count(scopedReadTool); afterSecond != afterFirst {
		t.Fatalf("unattributed caller: %q dispatched again on the second turn (%d -> %d) — with no "+
			"principal every caller shares ONE scope (the documented v0.1 single-tenant behavior), "+
			"so the second read must still be served from the first's fill. Ledger: %v",
			scopedReadTool, afterFirst, afterSecond, rec.records())
	}
	for i, p := range rec.principalsOf(scopedReadTool) {
		if p != "" {
			t.Fatalf("dispatch %d of %q carried principal %q for an unattributed caller, want \"\": "+
				"the empty principal must write NO key, keeping the no-principal path byte-for-byte HEAD",
				i, scopedReadTool, p)
		}
	}
}
