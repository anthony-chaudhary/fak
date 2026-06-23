package vdso

// principal_test.go — witnesses for multi-tenant tier-2 isolation (principal.go).
//
// The leak these guard against: the tier-2 key is agent-blind, so for an
// identity-dependent, arg-blind tool (read_inbox{}, whoami) two principals issuing the
// same (tool,args) would share ONE cached result — principal B reading principal A's
// private bytes. The fix scopes the key per principal; these tests prove (1) distinct
// principals are isolated, (2) the SAME principal still reuses its own entry, (3) an
// empty/declared-shareable tool stays cross-principal shared (the key is byte-identical
// to v0.1), and (4) scoping preserves the tool:hash:epoch key grammar.

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// roCallP is a read-only call tagged with an isolation principal (unset for "").
func roCallP(tool, args, principal string) *abi.ToolCall {
	c := roCall(tool, args)
	if principal != "" {
		c.Meta[MetaPrincipal] = principal
	}
	return c
}

// TestPrincipal_IsolatesDistinctPrincipals is the headline leak witness: an
// identity-dependent, arg-blind read warmed by alice must NOT be served to bob.
func TestPrincipal_IsolatesDistinctPrincipals(t *testing.T) {
	v := New(32)
	ctx := context.Background()
	const tool, args = "get_inbox", `{}` // arg-blind: only the principal differs

	// Alice warms her private read.
	v.Emit(completeEvent(roCallP(tool, args, "alice"), `{"owner":"alice"}`))

	// Bob issues the SAME (tool,args) — must MISS (his own engine call), never be
	// served alice's bytes.
	if res, ok := v.Lookup(ctx, roCallP(tool, args, "bob")); ok {
		t.Fatalf("cross-principal LEAK: bob was served alice's entry: %s", resolveBytes(t, res.Payload))
	}
	if v.MissReasons()[MissNotCached] == 0 {
		t.Fatalf("bob's miss should be NOT_CACHED (no entry under his key), reasons=%v", v.MissReasons())
	}

	// Alice re-reads -> HIT, her own bytes (within-tenant reuse intact).
	res, ok := v.Lookup(ctx, roCallP(tool, args, "alice"))
	if !ok {
		t.Fatalf("alice's own re-read must hit her cached entry")
	}
	if got := string(resolveBytes(t, res.Payload)); got != `{"owner":"alice"}` {
		t.Fatalf("alice served %q, want her own entry", got)
	}
}

// TestPrincipal_SamePrincipalShares pins that isolation does not break the legitimate
// within-principal reuse the cache exists for.
func TestPrincipal_SamePrincipalShares(t *testing.T) {
	v := New(32)
	ctx := context.Background()
	const tool, args = "get_inbox", `{}`
	v.Emit(completeEvent(roCallP(tool, args, "alice"), `{"owner":"alice"}`))
	if _, ok := v.Lookup(ctx, roCallP(tool, args, "alice")); !ok {
		t.Fatalf("the same principal must reuse its own warmed entry")
	}
}

// TestPrincipal_EmptyPrincipalSharesUnchanged proves the default (single-tenant)
// behavior is unchanged: an anonymous fill is served to another anonymous caller, and
// the empty-principal key is byte-identical to a never-scoped key.
func TestPrincipal_EmptyPrincipalSharesUnchanged(t *testing.T) {
	v := New(32)
	ctx := context.Background()
	const tool, args = "get_doc", `{"id":"x"}`

	v.Emit(completeEvent(roCall(tool, args), `{"body":"X"}`))
	if _, ok := v.Lookup(ctx, roCall(tool, args)); !ok {
		t.Fatalf("anonymous (no-principal) callers must still SHARE the v0.1 cache")
	}
}

// TestPrincipal_KeyGrammarPreservedAndScoped proves the scoped key (a) diverges per
// principal (isolation), (b) still parses as a cachemeta tool key, and (c) never
// pollutes the tool field — the principal is folded into the hash, not the grammar.
func TestPrincipal_KeyGrammarPreservedAndScoped(t *testing.T) {
	v := New(32)
	const tool, args = "get_inbox", `{}`
	anon := roCall(tool, args)
	alice := roCallP(tool, args, "alice")
	bob := roCallP(tool, args, "bob")

	v.mu.Lock()
	kAnon := v.keyLocked(anon, []byte(args))
	kAlice := v.keyLocked(alice, []byte(args))
	kBob := v.keyLocked(bob, []byte(args))
	v.mu.Unlock()

	if kAlice == kAnon || kBob == kAnon || kAlice == kBob {
		t.Fatalf("principals must produce DISTINCT keys: anon=%q alice=%q bob=%q", kAnon, kAlice, kBob)
	}
	e, err := cachemeta.FromVDSOKey(kAlice, alice.Args)
	if err != nil {
		t.Fatalf("scoped key broke the cachemeta grammar: %v", err)
	}
	if e.Derivation.Tool != tool {
		t.Fatalf("principal leaked into the parsed tool field: %q (want %q)", e.Derivation.Tool, tool)
	}
}

// TestPrincipal_ShareableSharesAcrossPrincipals proves the opt-in: a tool declared
// public/identity-independent is served ACROSS principals (the deliberate cross-tenant
// win, e.g. a shared policy doc).
func TestPrincipal_ShareableSharesAcrossPrincipals(t *testing.T) {
	v := New(32)
	ctx := context.Background()
	const tool, args = "get_policy", `{"id":"refund"}`
	v.RegisterShareable(tool)
	if !v.Shareable(tool) {
		t.Fatalf("RegisterShareable did not take")
	}

	v.Emit(completeEvent(roCallP(tool, args, "alice"), `{"body":"refund-policy"}`))

	res, ok := v.Lookup(ctx, roCallP(tool, args, "bob"))
	if !ok {
		t.Fatalf("a Shareable tool must be served ACROSS principals")
	}
	if res.Meta["tier"] != "2" {
		t.Fatalf("shareable cross-principal serve should be a tier-2 hit, got tier %q", res.Meta["tier"])
	}
	if got := string(resolveBytes(t, res.Payload)); got != `{"body":"refund-policy"}` {
		t.Fatalf("shareable serve got %q", got)
	}
}
