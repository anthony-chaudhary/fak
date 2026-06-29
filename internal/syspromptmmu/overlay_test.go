package syspromptmmu

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/capindex"
	"github.com/anthony-chaudhary/fak/internal/promptmmu"
)

func digestOf(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

// fakeResolver serves n skill capabilities with fixed-size bodies and counts how many
// times a BODY is paged (Resolve), so a test can prove the HIT path re-faults nothing.
type fakeResolver struct {
	bodies   map[capindex.CapRef][]byte
	cards    []capindex.CapCard
	resolves *int
}

func newFakeResolver(n, bodyBytes int, trigger string, resolves *int) *fakeResolver {
	r := &fakeResolver{bodies: map[capindex.CapRef][]byte{}, resolves: resolves}
	for i := 0; i < n; i++ {
		ref := capindex.CapRef{Kind: capindex.CapKindSkill, Name: fmt.Sprintf("cap-%04d", i)}
		body := bytes.Repeat([]byte("x"), bodyBytes)
		r.bodies[ref] = body
		r.cards = append(r.cards, capindex.CapCard{
			Ref:     ref,
			Digest:  digestOf(body), // the body content hash — the at-rest HIT key
			Trigger: trigger,
		})
	}
	return r
}

func (r *fakeResolver) Index() []capindex.CapCard { return r.cards }

func (r *fakeResolver) Fault(ref capindex.CapRef) (capindex.Capability, error) {
	body, ok := r.bodies[ref]
	if !ok {
		return capindex.Capability{}, capindex.ErrNotFound
	}
	cnt := r.resolves
	return capindex.Capability{
		Ref:     ref,
		Digest:  digestOf(body),
		Resolve: func() []byte { *cnt++; return body }, // a body page; counted
	}, nil
}

func newCatalog(t *testing.T, n, bodyBytes int, trigger string, resolves *int) *capindex.Catalog {
	t.Helper()
	cat := capindex.NewCatalog()
	cat.AddResolver(capindex.CapKindSkill, newFakeResolver(n, bodyBytes, trigger, resolves))
	cat.Sync()
	return cat
}

func baseTokens() int64 {
	var s int64
	for _, seg := range BaseContextPlan() {
		s += seg.Tokens
	}
	return s
}

// TestOverlayZeroForInfinity is the headline invariant-4 proof: the resident base is
// independent of catalog size, and the per-turn overlay is budget-bounded — so the
// resident cost stays FLAT as the catalog grows from 10 to 1000 capabilities.
func TestOverlayZeroForInfinity(t *testing.T) {
	if baseTokens() == 0 {
		t.Fatal("resident base has no tokens")
	}
	var ra, rb int
	small := newCatalog(t, 10, 400, "read file grep", &ra) // 100 tokens/body
	big := newCatalog(t, 1000, 400, "read file grep", &rb)

	const budget int64 = 250
	selSmall := SelectOverlay(small, "read", budget)
	selBig := SelectOverlay(big, "read", budget)

	if selSmall.Tokens > budget || selBig.Tokens > budget {
		t.Fatalf("overlay exceeded budget: small=%d big=%d budget=%d", selSmall.Tokens, selBig.Tokens, budget)
	}
	if len(selBig.Faulted) != len(selSmall.Faulted) {
		t.Errorf("faulted count grew with catalog: small=%d big=%d (must be budget-bounded, not catalog-bounded)",
			len(selSmall.Faulted), len(selBig.Faulted))
	}
	if len(selBig.Faulted) >= 1000 {
		t.Errorf("the 1000-card catalog faulted %d bodies — the base must be pointers, not bodies", len(selBig.Faulted))
	}
}

// TestOverlayBudget asserts winners are faulted in rank order until the budget stops the
// run, the boundary card is measured (faulted) once then excluded, and the rest are
// skipped without a fault.
func TestOverlayBudget(t *testing.T) {
	var resolves int
	cat := newCatalog(t, 5, 400, "read file grep", &resolves) // 100 tokens/body

	sel := SelectOverlay(cat, "read", 250) // fits 2 (200), the 3rd (300) overflows
	if len(sel.Faulted) != 2 {
		t.Errorf("faulted %d, want 2", len(sel.Faulted))
	}
	if sel.Tokens != 200 {
		t.Errorf("overlay tokens %d, want 200", sel.Tokens)
	}
	if len(sel.Skipped) != 3 {
		t.Errorf("skipped %d, want 3 (boundary + 2 lower-ranked)", len(sel.Skipped))
	}
	if resolves != 3 {
		t.Errorf("body pages = %d, want 3 (2 included + 1 boundary probe)", resolves)
	}
}

// TestOverlayDeterministic asserts the same (catalog, intent, budget) yields an
// identical selection (segments, faulted set, and digest).
func TestOverlayDeterministic(t *testing.T) {
	var r int
	cat := newCatalog(t, 6, 400, "read file grep", &r)
	a := SelectOverlay(cat, "read", 350)
	b := SelectOverlay(cat, "read", 350)
	if !reflect.DeepEqual(a.Overlay, b.Overlay) || a.Digest != b.Digest {
		t.Fatal("SelectOverlay is not deterministic")
	}
	if len(a.Faulted) == 0 {
		t.Fatal("expected a non-empty overlay")
	}
}

// TestOverlayEmptyIntentAndNilCatalog asserts the degenerate cases produce an empty
// overlay (no query, nothing faulted) rather than an error.
func TestOverlayEmptyIntentAndNilCatalog(t *testing.T) {
	var r int
	cat := newCatalog(t, 3, 400, "read file", &r)

	if sel := SelectOverlay(cat, "", 1000); len(sel.Overlay) != 0 || len(sel.Faulted) != 0 {
		t.Errorf("empty intent must yield an empty overlay, got %d segs", len(sel.Overlay))
	}
	if r != 0 {
		t.Errorf("empty intent faulted %d bodies, want 0", r)
	}
	if sel := SelectOverlay(nil, "read", 1000); len(sel.Overlay) != 0 {
		t.Errorf("nil catalog must yield an empty overlay")
	}
}

// TestOverlayCacheHitNoRefault asserts a re-invocation with an identical rank digest is
// a HIT served from the cache with NO body re-faulted (the SkillContextRecord property).
func TestOverlayCacheHitNoRefault(t *testing.T) {
	var resolves int
	cat := newCatalog(t, 5, 400, "read file grep", &resolves)
	oc := NewOverlayCache()

	sel1, hit1 := oc.GetOrSelect(cat, "read", 250)
	if hit1 {
		t.Fatal("first call must be a MISS")
	}
	n1 := resolves
	if n1 == 0 {
		t.Fatal("first call should have faulted at least one body")
	}

	sel2, hit2 := oc.GetOrSelect(cat, "read", 250)
	if !hit2 {
		t.Fatal("second call with an unchanged catalog must be a HIT")
	}
	if resolves != n1 {
		t.Errorf("HIT re-faulted bodies: %d -> %d", n1, resolves)
	}
	if sel1.Digest != sel2.Digest || !reflect.DeepEqual(sel1.Overlay, sel2.Overlay) {
		t.Error("HIT returned a different selection than the cached one")
	}
}

// TestOverlayDigestTracksBodyChange asserts the HIT key changes when a capability body
// changes — so a stale overlay is a MISS, never a wrong HIT.
func TestOverlayDigestTracksBodyChange(t *testing.T) {
	var a, b int
	catA := newCatalog(t, 3, 400, "read file", &a)
	catB := newCatalog(t, 3, 500, "read file", &b) // different body size ⇒ different digests

	if SelectOverlay(catA, "read", 5000).Digest == SelectOverlay(catB, "read", 5000).Digest {
		t.Fatal("digest must change when capability bodies change")
	}
}

// TestOverlayFeedsSpliceCacheSafe ties Rung 3 to Rung 2: a queried overlay realized
// through BuildSystemValue then swapped via SpliceSystemOverlay leaves the resident
// spine+policy prefix byte-identical (the Rung-2 cache invariant stays green).
func TestOverlayFeedsSpliceCacheSafe(t *testing.T) {
	var r int
	plan := BaseContextPlan()
	cat := newCatalog(t, 5, 200, "read file grep", &r) // 50 tokens/body

	sel1 := SelectOverlay(cat, "read", 120) // fits 2
	body := bodyWith(t, BuildSystemValue(plan, sel1.Overlay), nil)

	_, prefixEnd, _, ok := promptmmu.ArraySplicePoints(body, "system")
	if !ok {
		t.Fatal("could not anchor the cached prefix on the built body")
	}
	prefix := append([]byte(nil), body[:prefixEnd]...)

	sel2 := SelectOverlay(cat, "read", 400) // fits more → a different overlay
	if len(sel2.Faulted) <= len(sel1.Faulted) {
		t.Fatalf("expected a larger overlay at the bigger budget: %d vs %d", len(sel2.Faulted), len(sel1.Faulted))
	}
	res := SpliceSystemOverlay(body, plan, sel2.Overlay, decodeOK)
	if !res.Changed {
		t.Fatalf("expected a splice, got identity (%s)", res.SkipReason)
	}
	if len(res.Body) < len(prefix) || !bytes.Equal(res.Body[:len(prefix)], prefix) {
		t.Fatal("a queried overlay broke the Rung-2 resident-prefix invariant")
	}
}
