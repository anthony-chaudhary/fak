package gateway

// served_isolation_test.go — the multi-tenant counterpart of served_sharing_test.go.
//
// served_sharing_test proves the WIN: cross-agent tier-2 sharing (the key is agent-blind,
// so agent B is served agent A's warmed read). This file proves the matching PRIVACY
// guarantee on the same live served path: when a request names an isolation PRINCIPAL
// (X-Fak-Principal header / request field, lowered onto ctx by WithPrincipal), the vDSO
// scopes the tier-2 entry per principal, so principal B is NEVER served principal A's
// private cached result — closing the cross-tenant cache leak + hit/miss timing oracle —
// while a tool DECLARED shareable stays cross-tenant shared (the public-read win, made
// an explicit choice rather than an accident).

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// principalEngine echoes the calling principal into its result, so a cross-tenant leak
// is VISIBLE (bob served `{"owner":"alice"}` would be the bug). It models an
// identity-dependent, arg-blind tool: the result depends on WHO asked, not on the args.
type principalEngine struct{}

func (principalEngine) Caps() []abi.Capability { return nil }
func (principalEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	owner := ""
	if c.Meta != nil {
		owner = c.Meta[vdso.MetaPrincipal]
	}
	body := []byte(`{"owner":"` + owner + `"}`)
	return &abi.Result{Call: c, Status: abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body))},
		Meta:    map[string]string{"engine": "principal"}}, nil
}

// newIsolationServer wires an isolated chain (inline backend + the identity-revealing
// engine + allow-all adjudicator) plus a fresh real vDSO (default Global granularity),
// registered as the kernel's FastPath + Emitter, and returns a served gateway bound to it.
func newIsolationServer(t *testing.T) (*Server, *vdso.VDSO) {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", principalEngine{})
	abi.RegisterAdjudicator(0, allowAllAdj{})

	v := vdso.New(vdso.DefaultCacheSize)
	abi.RegisterFastPath(1, v)
	abi.RegisterEmitter(v)

	srv, err := New(Config{EngineID: "test", Model: "test", VDSO: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv, v
}

// servedReadP drives one read through the full served syscall boundary on behalf of a
// named principal, returning the admitted result envelope.
func servedReadP(t *testing.T, srv *Server, tool, args, principal, agent string) *ResultEnvelope {
	t.Helper()
	wv, env, err := srv.syscall(WithPrincipal(context.Background(), principal), tool, args, true /*readOnly*/, "", agent)
	if err != nil {
		t.Fatalf("served read %s by %s/%s: %v", tool, principal, agent, err)
	}
	if wv.Kind != "ALLOW" {
		t.Fatalf("served read %s by %s: verdict=%q, want ALLOW", tool, principal, wv.Kind)
	}
	if env == nil {
		t.Fatalf("served read %s by %s: nil result envelope", tool, principal)
	}
	return env
}

// TestServed_PrincipalIsolation_NoCrossTenantLeak is the headline privacy witness: an
// identity-dependent, arg-blind read warmed by alice is NOT served to bob — bob reaches
// the engine and gets HIS own result, while alice's own re-read still hits her entry.
func TestServed_PrincipalIsolation_NoCrossTenantLeak(t *testing.T) {
	srv, _ := newIsolationServer(t)
	const tool, args = "get_inbox", `{}` // arg-blind: only the principal differs

	// Alice: cold read -> engine -> fills HER entry with HER private bytes.
	a := servedReadP(t, srv, tool, args, "alice", "agent-A")
	if by, _ := servedBy(a); by == "vdso" {
		t.Fatalf("alice's first read should MISS (engine), got served_by=%q", by)
	}
	if a.Content != `{"owner":"alice"}` {
		t.Fatalf("alice served %q, want her own result", a.Content)
	}

	// Bob: the SAME (tool,args), DIFFERENT principal. Must not be served alice's entry —
	// it must miss to the engine and get BOB's own result.
	b := servedReadP(t, srv, tool, args, "bob", "agent-B")
	if by, _ := servedBy(b); by == "vdso" {
		t.Fatalf("cross-tenant LEAK: bob served from alice's cache (served_by=vdso): %s", b.Content)
	}
	if b.Content != `{"owner":"bob"}` {
		t.Fatalf("bob served %q, want his own isolated result {\"owner\":\"bob\"}", b.Content)
	}

	// Alice again: HER warmed entry is still reusable (within-tenant sharing intact).
	a2 := servedReadP(t, srv, tool, args, "alice", "agent-A2")
	if by, tier := servedBy(a2); by != "vdso" || tier != "2" {
		t.Fatalf("alice's re-read should be her own tier-2 hit, got served_by=%q tier=%q", by, tier)
	}
	if a2.Content != `{"owner":"alice"}` {
		t.Fatalf("alice's re-read served %q", a2.Content)
	}

	// Two engine calls (alice cold + bob cold) prove bob did NOT serve from alice's fill.
	c := srv.k.Counters()
	if c.EngineCalls != 2 {
		t.Errorf("EngineCalls=%d, want 2 (alice + bob each reached the engine; no cross-tenant serve)", c.EngineCalls)
	}
}

// TestServed_ShareableToolSharesAcrossPrincipals proves the opt-in: a tool declared
// public/identity-independent IS served across principals on the live path (the
// deliberate cross-tenant cache win — a shared policy/reference doc).
func TestServed_ShareableToolSharesAcrossPrincipals(t *testing.T) {
	srv, v := newIsolationServer(t)
	const tool, args = "get_policy", `{"id":"refund"}`
	v.RegisterShareable(tool)

	// Alice warms the shared-knowledge read (cold -> engine).
	a := servedReadP(t, srv, tool, args, "alice", "agent-A")
	if by, _ := servedBy(a); by == "vdso" {
		t.Fatalf("alice's first read of the shareable tool should MISS (engine), got served_by=%q", by)
	}

	// Bob, a DIFFERENT principal, is served from alice's fill — because the tool is
	// declared identity-independent. A tier-2 cross-principal hit, no second engine call.
	b := servedReadP(t, srv, tool, args, "bob", "agent-B")
	if by, tier := servedBy(b); by != "vdso" || tier != "2" {
		t.Fatalf("a Shareable tool must serve ACROSS principals (tier-2), got served_by=%q tier=%q", by, tier)
	}

	c := srv.k.Counters()
	if c.EngineCalls != 1 {
		t.Errorf("EngineCalls=%d, want 1 (alice warmed; bob shared the public read)", c.EngineCalls)
	}
}

// emitWriteAs publishes a write-shaped completion to the process-global vDSO on behalf of
// a principal (mirroring coherence_test's direct-bus pattern). The mutation lands on every
// live server's change feed, stamped with the issuer principal.
func emitWriteAs(tool, principal, args string) {
	wc := &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(args)},
		Meta: map[string]string{vdso.MetaPrincipal: principal}}
	vdso.Default.Emit(abi.Event{Kind: abi.EvComplete, Call: wc,
		Result: &abi.Result{Call: wc, Status: abi.StatusOK, Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"ok":true}`)}}})
}

// TestChanges_ScopedPerPrincipal_NoCrossTenantTagLeak proves the change-feed counterpart
// of the cache isolation: a tenant draining /v1/fak/changes sees its OWN writes but NOT a
// peer tenant's (whose tool/tags would reveal which entities that tenant changed), while
// an unscoped (admin / single-tenant) drain still sees everything — the v0.1 behavior.
func TestChanges_ScopedPerPrincipal_NoCrossTenantTagLeak(t *testing.T) {
	srv := newTestServer(t) // its feed subscribes to vdso.Default at New()

	// Two tenants each issue a write naming a different entity. Unique tool names keep
	// these distinguishable from any peer event already on the shared bus.
	emitWriteAs("book_flight_zA", "alice", `{"origin":"SFO","destination":"JFK"}`)
	emitWriteAs("book_flight_zB", "bob", `{"origin":"LAX","destination":"SEA"}`)

	// Bob's scoped drain: his own write, never alice's.
	bobEvents, _ := srv.changes("bob", 0)
	if findMutation(bobEvents, "book_flight_zB") == nil {
		t.Errorf("bob should see his own write on the feed")
	}
	if findMutation(bobEvents, "book_flight_zA") != nil {
		t.Errorf("cross-tenant LEAK: bob's feed surfaced alice's write")
	}

	// Alice's scoped drain is the mirror.
	aliceEvents, _ := srv.changes("alice", 0)
	if findMutation(aliceEvents, "book_flight_zA") == nil {
		t.Errorf("alice should see her own write")
	}
	if findMutation(aliceEvents, "book_flight_zB") != nil {
		t.Errorf("cross-tenant LEAK: alice's feed surfaced bob's write")
	}

	// An unscoped (admin / single-tenant) drain still sees BOTH — back-compat.
	all, _ := srv.changes("", 0)
	if findMutation(all, "book_flight_zA") == nil || findMutation(all, "book_flight_zB") == nil {
		t.Errorf("an unscoped drain must see all tenants' writes (single-tenant / admin back-compat)")
	}
}
