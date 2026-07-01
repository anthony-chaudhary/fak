package gateway

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/fusedturn"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// outEngine returns a FIXED output string as the result payload, so an ensemble test
// can assert WHICH member produced WHICH output and how Combine folded them. Distinct
// from markerEngine (which echoes the args back): an ensemble fold reads the member
// OUTPUTS, so each member needs a controllable, distinct output.
type outEngine struct{ id, out string }

func (outEngine) Caps() []abi.Capability { return nil }
func (e outEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	ref, err := abi.ActiveResolver().Put(ctx, []byte(e.out))
	if err != nil {
		return nil, err
	}
	return &abi.Result{Call: c, Status: abi.StatusOK, Payload: ref,
		Meta: map[string]string{"engine": e.id}}, nil
}

// ensembleServer wires an isolated chain WITH the real engine-residency gate plus a
// routing manifest, registering one outEngine per (id,out) member so a fan-out produces
// distinct, assertable outputs. The kernel default is the echo engine. Mirrors
// routeServer but for the multi-member path. Not parallel-safe (mutates the registry).
func ensembleServer(t *testing.T, m *modelroute.Manifest, members map[string]string) *Server {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{}) // the kernel-default local engine
	for id, out := range members {
		abi.RegisterEngine(id, outEngine{id: id, out: out})
	}
	abi.RegisterAdjudicator(0, toolAdj{}) // allow*/deny* by tool-name prefix
	engine.RegisterResidencyGate()        // the REAL residency floor (rank 12)
	srv, err := New(Config{EngineID: "test", Model: "m", VDSO: true, RouteManifest: m})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv
}

// ensembleManifest routes one tool to a multi-member Plan folded by the given reduction,
// with a local single-model default for everything else.
func ensembleManifest(tool string, reduce modelroute.Reduction, models ...string) *modelroute.Manifest {
	mems := make([]modelroute.Member, len(models))
	for i, mdl := range models {
		mems[i] = modelroute.Member{Model: mdl}
	}
	return &modelroute.Manifest{
		Default: modelroute.Plan{Members: []modelroute.Member{{Model: "test"}}},
		Rules: []modelroute.Rule{{
			Name:  "ensemble-" + tool,
			Match: modelroute.Match{Tool: tool},
			Plan:  modelroute.Plan{Members: mems, Reduce: reduce},
		}},
	}
}

// A vote ensemble folds the members' outputs to the weighted-majority answer, in member
// order, and surfaces the reduce + member count on the result. The tool is write-shaped
// (no read-only prefix) so the vDSO never dedups the members — every member's engine runs.
func TestEnsembleVoteFold(t *testing.T) {
	m := ensembleManifest("allow_vote", modelroute.ReduceVote, "m1", "m2", "m3")
	s := ensembleServer(t, m, map[string]string{"m1": "yes", "m2": "yes", "m3": "no"})

	wv, env, err := s.syscall(context.Background(), "allow_vote", `{}`, false, "", "")
	if err != nil {
		t.Fatalf("syscall: %v", err)
	}
	if wv.Kind != "ALLOW" || wv.By != "modelroute-ensemble" {
		t.Fatalf("verdict = %+v, want ALLOW by modelroute-ensemble", wv)
	}
	if env == nil || env.Content != "yes" {
		t.Fatalf("vote output = %q, want %q", envContent(env), "yes")
	}
	if env.Meta["reduce"] != "vote" {
		t.Fatalf("reduce meta = %q, want vote", env.Meta["reduce"])
	}
	if env.Meta["ensemble_members"] != "3" {
		t.Fatalf("ensemble_members = %q, want 3", env.Meta["ensemble_members"])
	}
	if env.Meta["winner"] != "m1" { // smallest model id voting for the winning output
		t.Fatalf("winner = %q, want m1", env.Meta["winner"])
	}
}

// A concat ensemble joins the members' outputs in member order (deterministic).
func TestEnsembleConcatFold(t *testing.T) {
	m := ensembleManifest("allow_concat", modelroute.ReduceConcat, "ca", "cb")
	s := ensembleServer(t, m, map[string]string{"ca": "alpha", "cb": "beta"})

	_, env, err := s.syscall(context.Background(), "allow_concat", `{}`, false, "", "")
	if err != nil {
		t.Fatalf("syscall: %v", err)
	}
	if env == nil || env.Content != "alpha\nbeta" {
		t.Fatalf("concat output = %q, want %q", envContent(env), "alpha\nbeta")
	}
	if env.Meta["reduce"] != "concat" {
		t.Fatalf("reduce meta = %q, want concat", env.Meta["reduce"])
	}
}

// A first ensemble returns the first member's output (fastest-wins / fallback head).
func TestEnsembleFirstFold(t *testing.T) {
	m := ensembleManifest("allow_first", modelroute.ReduceFirst, "fa", "fb")
	s := ensembleServer(t, m, map[string]string{"fa": "from-a", "fb": "from-b"})

	_, env, err := s.syscall(context.Background(), "allow_first", `{}`, false, "", "")
	if err != nil {
		t.Fatalf("syscall: %v", err)
	}
	if env == nil || env.Content != "from-a" {
		t.Fatalf("first output = %q, want %q", envContent(env), "from-a")
	}
	if env.Meta["winner"] != "fa" {
		t.Fatalf("winner = %q, want fa", env.Meta["winner"])
	}
}

// Each ensemble member is its OWN kernel call: a 3-member fan-out submits 3 calls and
// dispatches 3 engines, so every member shows in the adjudication counters (the #597
// acceptance: never one fan-out that bypasses the floor for the members).
func TestEnsembleMembersAdjudicatedIndividually(t *testing.T) {
	m := ensembleManifest("allow_count", modelroute.ReduceConcat, "k1", "k2", "k3")
	s := ensembleServer(t, m, map[string]string{"k1": "1", "k2": "2", "k3": "3"})

	before := s.k.Counters()
	if _, _, err := s.syscall(context.Background(), "allow_count", `{}`, false, "", ""); err != nil {
		t.Fatalf("syscall: %v", err)
	}
	after := s.k.Counters()
	if got := after.Submits - before.Submits; got != 3 {
		t.Fatalf("ensemble submitted %d kernel calls, want 3 (one per member)", got)
	}
	if got := after.EngineCalls - before.EngineCalls; got != 3 {
		t.Fatalf("ensemble dispatched %d engine calls, want 3 (all members allowed + local)", got)
	}
}

func TestGatewayStampsFusedTurnFamilies(t *testing.T) {
	s := newTestServer(t)
	ctx := context.Background()

	toolCall, err := s.buildCall(ctx, "allow_read", `{}`, true, "", "turn-fused")
	if err != nil {
		t.Fatalf("buildCall: %v", err)
	}
	member := memberCall(toolCall, "test")

	if got := fusedturn.Classify(toolCall); got != fusedturn.ClassClassical {
		t.Fatalf("gateway tool call class = %v, want classical", got)
	}
	if got := fusedturn.Classify(member); got != fusedturn.ClassWeight {
		t.Fatalf("gateway ensemble member class = %v, want weight", got)
	}

	ft := fusedturn.Fuse([]*abi.ToolCall{member, toolCall})
	if !ft.Fused() {
		t.Fatalf("fused turn was not recognized: summary=%+v", ft.Summary())
	}
	rows := ft.Adjudicate(ctx, s.k)
	families := fusedturn.GovernedFamilies(rows)
	if len(families) != 2 || families[0] != fusedturn.ClassClassical || families[1] != fusedturn.ClassWeight {
		t.Fatalf("governed families = %v, want [classical weight]; rows=%+v", families, rows)
	}
}

// THE load-bearing test: an all-remote ensemble carrying a sensitive (tenant) payload
// has EVERY member cross the residency floor and get DENIED — the ensemble fan-out does
// NOT bypass the floor for its members. With no surviving vote it fails closed (no silent
// empty success): the residency deny reason reaches the wire and the result is ERROR.
func TestEnsembleResidencyDeniesRemoteMembers(t *testing.T) {
	m := ensembleManifest("allow_remote", modelroute.ReduceVote, "remote:openai", "remote:anthropic")
	s := ensembleServer(t, m, nil) // remote members are denied before dispatch; no engine needed
	ctx := context.Background()

	tc, err := s.buildCall(ctx, "allow_remote", `{}`, false, "", "")
	if err != nil {
		t.Fatalf("buildCall: %v", err)
	}
	tc.Meta["sensitivity"] = "tenant" // mark the payload sensitive (mirrors the #596 residency test)
	plan, ok := s.ensemblePlan("allow_remote", false, tc.Meta)
	if !ok {
		t.Fatalf("expected an ensemble plan for allow_remote")
	}

	wv, env, err := s.dispatchEnsemble(ctx, tc, plan)
	if err != nil {
		t.Fatalf("dispatchEnsemble: %v", err)
	}
	if wv.Kind != "DENY" || wv.By != "engine-residency" {
		t.Fatalf("all-remote sensitive ensemble must fail closed via residency, got %+v", wv)
	}
	if wv.Reason != abi.ReasonName(abi.ReasonTrustViolation) {
		t.Fatalf("deny reason = %q, want TRUST_VIOLATION", wv.Reason)
	}
	if env == nil || env.Status != "ERROR" {
		t.Fatalf("failed-closed ensemble status = %q, want ERROR", envStatus(env))
	}
	if env.Meta["ensemble_refused"] != "2" {
		t.Fatalf("ensemble_refused = %q, want 2", env.Meta["ensemble_refused"])
	}
}

// A mixed ensemble [local, remote, local] on a sensitive payload: the remote member is
// denied by residency and contributes NO vote, while the two local survivors fold in
// member order despite the gap — the "member order preserved, refused member dropped"
// contract.
func TestEnsembleMixedSurvivorsFoldInOrder(t *testing.T) {
	m := ensembleManifest("allow_mixed", modelroute.ReduceConcat, "local-a", "remote:x", "local-b")
	s := ensembleServer(t, m, map[string]string{"local-a": "AAA", "local-b": "BBB"})
	ctx := context.Background()

	tc, err := s.buildCall(ctx, "allow_mixed", `{}`, false, "", "")
	if err != nil {
		t.Fatalf("buildCall: %v", err)
	}
	tc.Meta["sensitivity"] = "tenant"
	plan, ok := s.ensemblePlan("allow_mixed", false, tc.Meta)
	if !ok {
		t.Fatalf("expected an ensemble plan for allow_mixed")
	}

	wv, env, err := s.dispatchEnsemble(ctx, tc, plan)
	if err != nil {
		t.Fatalf("dispatchEnsemble: %v", err)
	}
	if wv.Kind != "ALLOW" {
		t.Fatalf("verdict = %+v, want ALLOW (two local survivors)", wv)
	}
	if env == nil || env.Content != "AAA\nBBB" {
		t.Fatalf("survivor concat = %q, want %q (member order, denied member dropped)", envContent(env), "AAA\nBBB")
	}
	if env.Meta["ensemble_members"] != "2" {
		t.Fatalf("ensemble_members = %q, want 2", env.Meta["ensemble_members"])
	}
	if env.Meta["ensemble_refused"] != "1" {
		t.Fatalf("ensemble_refused = %q, want 1 (the denied remote member)", env.Meta["ensemble_refused"])
	}
}

// envContent / envStatus are nil-safe field reads for failure messages.
func envContent(e *ResultEnvelope) string {
	if e == nil {
		return "<nil>"
	}
	return e.Content
}

func envStatus(e *ResultEnvelope) string {
	if e == nil {
		return "<nil>"
	}
	return e.Status
}
