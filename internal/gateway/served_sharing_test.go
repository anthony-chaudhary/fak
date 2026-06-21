package gateway

// served_sharing_test.go — #4: the cross-agent tier-2 sharing + scoped-invalidation
// demonstration carried onto the LIVE SERVED PATH (the half the FLEET-SWEEP
// benchmark proved in internal/turnbench but `fak serve` had no end-to-end test for).
//
// Both tests drive the REAL kernel→vDSO fast path through Server.syscall (the same
// helper /v1/fak/syscall and the MCP fak_syscall call), with a fresh real *vdso.VDSO
// registered as the FastPath+Emitter. They assert two productionization claims:
//
//   1. CROSS-AGENT SHARING: a read warmed by agent A is served from the tier-2 cache
//      to a DIFFERENT agent B/C through one gateway — a counted vDSO hit, no second
//      engine call — because the tier-2 key (tool:argHash:epoch) is agent-blind (it
//      carries no TraceID). The measured served hit-rate is the cross-agent uplift.
//   2. SCOPED INVALIDATION SPARES A PEER: at Resource granularity a write to one
//      entity bumps only that entity's epoch, so a peer's shared read of a DIFFERENT
//      entity stays warm (still a tier-2 hit), while the written entity's own read is
//      correctly invalidated (a miss). This is the win that turns cross-agent sharing
//      from a net loss under writes into a net gain.
//
// Production wires this identically: cmd/fak `--invalidation` → gateway.New →
// vdso.Default.SetGranularity, with vdso.Default registered as the FastPath+Emitter
// by internal/registrations. Here we use a fresh instance so the assertions are
// hermetic (no shared process-global cache state across the suite).

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// allowAllAdj admits every call, so these tests exercise the vDSO sharing/eviction
// path rather than the adjudicator (which has its own tests).
type allowAllAdj struct{}

func (allowAllAdj) Caps() []abi.Capability { return nil }
func (allowAllAdj) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	return abi.Verdict{Kind: abi.VerdictAllow, By: "test-allow-all"}
}

// newSharingServer wires an isolated chain (inline backend + echo engine +
// allow-all adjudicator) plus a fresh real vDSO at the configured granularity,
// registered as the kernel's FastPath and completion Emitter, and returns a served
// gateway bound to it.
func newSharingServer(t *testing.T, gran vdso.Granularity) (*Server, *vdso.VDSO) {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, allowAllAdj{})

	v := vdso.New(vdso.DefaultCacheSize)
	v.SetGranularity(gran)
	abi.RegisterFastPath(1, v) // *VDSO.Lookup covers all three tiers
	abi.RegisterEmitter(v)     // fills tier-2 from EvComplete + bumps epochs on writes

	srv, err := New(Config{EngineID: "test", Model: "test", VDSO: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv, v
}

// servedRead drives one read through the full served syscall boundary as a named
// agent (its TraceID), returning the admitted result envelope. It fails the test on
// a transport error or a non-ALLOW verdict.
func servedRead(t *testing.T, srv *Server, tool, args, agent string) *ResultEnvelope {
	t.Helper()
	wv, env, err := srv.syscall(context.Background(), tool, args, true /*readOnly*/, "", agent)
	if err != nil {
		t.Fatalf("served read %s by %s: %v", tool, agent, err)
	}
	if wv.Kind != "ALLOW" {
		t.Fatalf("served read %s by %s: verdict=%q, want ALLOW", tool, agent, wv.Kind)
	}
	if env == nil {
		t.Fatalf("served read %s by %s: nil result envelope", tool, agent)
	}
	return env
}

func servedBy(env *ResultEnvelope) (by, tier string) {
	if env == nil || env.Meta == nil {
		return "", ""
	}
	return env.Meta["served_by"], env.Meta["tier"]
}

// TestServed_CrossAgentTier2Hit is the headline #4 demonstration: agent A warms a
// read through `fak serve`, and two DIFFERENT agents (B, C) are served the same
// result from the tier-2 cache — counted vDSO hits, no extra engine calls — proving
// the cache is shared ACROSS agents on the live served path (the key is agent-blind).
func TestServed_CrossAgentTier2Hit(t *testing.T) {
	srv, _ := newSharingServer(t, vdso.Global)
	const tool, args = "get_doc", `{"id":"refund-policy"}`

	// Agent A: cold read -> tier-2 miss -> engine -> cache fill.
	a := servedRead(t, srv, tool, args, "agent-A")
	if by, _ := servedBy(a); by == "vdso" {
		t.Fatalf("agent A first read should MISS (engine), got served_by=%q", by)
	}

	// Agents B and C: the SAME read, DIFFERENT trace ids -> tier-2 hits served from
	// A's fill. This is the cross-agent share (the key carries no TraceID).
	for _, agent := range []string{"agent-B", "agent-C"} {
		env := servedRead(t, srv, tool, args, agent)
		by, tier := servedBy(env)
		if by != "vdso" || tier != "2" {
			t.Fatalf("%s should be a cross-agent tier-2 hit, got served_by=%q tier=%q", agent, by, tier)
		}
	}

	// Measured on the live served path: 1 engine call warmed 3 served reads; the two
	// peers were cross-agent vDSO hits.
	c := srv.k.Counters()
	if c.VDSOHits != 2 {
		t.Errorf("VDSOHits=%d, want 2 (agent-B + agent-C served from A's fill)", c.VDSOHits)
	}
	if c.EngineCalls != 1 {
		t.Errorf("EngineCalls=%d, want 1 (only agent A's cold read reached the engine)", c.EngineCalls)
	}
	hitRate := float64(c.VDSOHits) / float64(c.Submits)
	t.Logf("cross-agent served hit-rate = %d/%d = %.2f (uplift: 3 reads, 1 engine call)", c.VDSOHits, c.Submits, hitRate)
	if hitRate <= 0 {
		t.Errorf("cross-agent hit-rate must be > 0 on the served path, got %.2f", hitRate)
	}
}

// TestServed_ScopedInvalidationSparesPeer proves the finer eraser on the served
// path: at Resource granularity, a write (booking) to one route invalidates only
// that route's shared read, while a PEER's shared read of a different route stays
// warm — and the written route's own read is correctly invalidated.
func TestServed_ScopedInvalidationSparesPeer(t *testing.T) {
	srv, _ := newSharingServer(t, vdso.Resource)
	const (
		route1 = `{"origin":"SFO","destination":"JFK"}` // entity flights:SFO-JFK
		route2 = `{"origin":"LAX","destination":"SEA"}` // entity flights:LAX-SEA
		read   = "search_direct_flight"
		book   = "book_flight"
	)

	// Warm both routes (agent A), then confirm a peer (agent B) shares both.
	servedRead(t, srv, read, route1, "agent-A") // miss -> fill route1
	servedRead(t, srv, read, route2, "agent-A") // miss -> fill route2
	if by, tier := servedBy(servedRead(t, srv, read, route1, "agent-B")); by != "vdso" || tier != "2" {
		t.Fatalf("peer read of route1 should be a tier-2 hit before any write, got served_by=%q tier=%q", by, tier)
	}
	if by, tier := servedBy(servedRead(t, srv, read, route2, "agent-B")); by != "vdso" || tier != "2" {
		t.Fatalf("peer read of route2 should be a tier-2 hit before any write, got served_by=%q tier=%q", by, tier)
	}

	// A booking on route2 (a write) — scoped to entity flights:LAX-SEA.
	wv, _, err := srv.syscall(context.Background(), book, route2, false /*readOnly*/, "", "agent-writer")
	if err != nil {
		t.Fatalf("served booking: %v", err)
	}
	if wv.Kind != "ALLOW" {
		t.Fatalf("served booking verdict=%q, want ALLOW", wv.Kind)
	}

	// THE PEER STAYS WARM: a different agent's shared read of route1 is still a tier-2
	// hit — the write to route2 did not strand it (the win the global flush destroyed).
	if by, tier := servedBy(servedRead(t, srv, read, route1, "agent-C")); by != "vdso" || tier != "2" {
		t.Errorf("route1 must stay warm after a write to route2 (scoped invalidation), got served_by=%q tier=%q", by, tier)
	}

	// SOUNDNESS: the written route's own read IS invalidated (a hit here would be a
	// stale serve).
	if by, _ := servedBy(servedRead(t, srv, read, route2, "agent-C")); by == "vdso" {
		t.Errorf("route2 must be invalidated by its own booking, but was served stale from cache (served_by=%q)", by)
	}
}
