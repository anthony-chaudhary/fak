package ctxresidency_test

import (
	"context"
	"testing"

	_ "github.com/anthony-chaudhary/fak/internal/blob" // registers the "blob" PageOut backend the gate pages bodies through.
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/ctxresidency"
)

// TestSpinePolicyNeverEvictedUnderPressure is the Rung-4 headline acceptance
// witness (#1262, invariant 3): no eviction path can remove a TierSpine /
// TierPolicy block regardless of budget pressure. A mix of spine, policy, and
// overlay capabilities is faulted in, then EvictColdest is hammered far more
// times than there are evictable bodies. The property: every drop is an OVERLAY
// block; the spine and policy floor are still resident at the end; and once the
// overlay is exhausted EvictColdest reports nothing left to evict — it never
// reaches into the spine to find one.
func TestSpinePolicyNeverEvictedUnderPressure(t *testing.T) {
	ctx := context.Background()
	mmu := ctxmmu.New()
	cr := ctxresidency.NewCapResidency(mmu)

	// The never-paged floor: identity spine + safety policy blocks. Each is
	// classified from a MEASURED profile and pinned to its tier — not guessed.
	spine := []ctxresidency.CapKey{
		{Kind: "spine", Name: "fak-identity", Version: "v1"},
		{Kind: "spine", Name: "capability-defn", Version: "v1"},
	}
	policy := []ctxresidency.CapKey{
		{Kind: "policy", Name: "deny-floor", Version: "v1"},
		{Kind: "policy", Name: "safety-critical", Version: "v1"},
	}
	overlay := []ctxresidency.CapKey{
		{Kind: "skill", Name: "alpha", Version: "v1"},
		{Kind: "skill", Name: "bravo", Version: "v1"},
		{Kind: "skill", Name: "charlie", Version: "v1"},
		{Kind: "skill", Name: "delta", Version: "v1"},
	}

	for _, k := range spine {
		cr.Fault(k, "sha256:"+k.Name, []byte("identity-load-bearing spine body"), nil)
		// resident iff used-every-turn ∧ small ∧ identity-load-bearing -> TierSpine.
		tier := ctxresidency.ClassifyTier(ctxresidency.BlockProfile{
			UsedEveryTurn: true, Small: true, IdentityLoadBearing: true,
		})
		if tier != ctxresidency.TierSpine {
			t.Fatalf("classifier put identity block in %q, want spine", tier)
		}
		cr.SetTier(k, tier)
	}
	for _, k := range policy {
		cr.Fault(k, "sha256:"+k.Name, []byte("safety-critical policy floor body"), nil)
		// Safety-load-bearing but LARGE and NOT used every turn — the worst case
		// for a coldness heuristic. It must STILL be never-paged (policy floor).
		tier := ctxresidency.ClassifyTier(ctxresidency.BlockProfile{
			UsedEveryTurn: false, Small: false, SafetyLoadBearing: true,
		})
		if tier != ctxresidency.TierPolicy {
			t.Fatalf("classifier put safety block in %q, want policy (never overlay)", tier)
		}
		cr.SetTier(k, tier)
	}
	for _, k := range overlay {
		cr.Fault(k, "sha256:"+k.Name, []byte("a pageable harness overlay card body"), nil)
		// Non-load-bearing -> overlay (the only pageable tier). Leave it as the
		// default/classified overlay; do not pin it resident.
		if tier := ctxresidency.ClassifyTier(ctxresidency.BlockProfile{UsedEveryTurn: true, Small: true}); tier != ctxresidency.TierOverlay {
			t.Fatalf("classifier put non-load-bearing block in %q, want overlay", tier)
		}
		cr.SetTier(overlay[0], ctxresidency.TierOverlay) // explicit on one; the rest stay default-pageable.
	}

	// Hammer eviction far past the number of evictable bodies. A spine/policy
	// block must NEVER be selected, regardless of how many rounds of pressure.
	protected := map[ctxresidency.CapKey]bool{}
	for _, k := range append(append([]ctxresidency.CapKey{}, spine...), policy...) {
		protected[k] = true
	}
	evictions := 0
	for i := 0; i < 100; i++ {
		evicted, radius, ok := cr.EvictColdest(ctx)
		if !ok {
			break
		}
		if protected[evicted] {
			t.Fatalf("EvictColdest selected a never-paged block %+v under pressure — invariant 3 broken", evicted)
		}
		// A blast-radius read precedes every overlay eviction; the cost is real.
		if radius.Tokens <= 0 {
			t.Errorf("overlay eviction of %+v recorded a non-positive blast radius %+v", evicted, radius)
		}
		evictions++
	}
	if evictions != len(overlay) {
		t.Errorf("evicted %d blocks, want exactly the %d overlay blocks (spine/policy must be untouchable)", evictions, len(overlay))
	}

	// Final state: every spine/policy block is still RESIDENT; every overlay
	// block is HELD (paged out through the witness gate).
	snap := cr.Snapshot()
	byKey := map[ctxresidency.CapKey]ctxresidency.CapRow{}
	for _, row := range snap.Caps {
		byKey[row.Key] = row
	}
	for _, k := range append(append([]ctxresidency.CapKey{}, spine...), policy...) {
		if byKey[k].State != ctxresidency.StateResident {
			t.Errorf("never-paged block %+v ended in state %v, want resident", k, byKey[k].State)
		}
		if !byKey[k].Tier.AlwaysResident() {
			t.Errorf("never-paged block %+v carries tier %q (not always-resident)", k, byKey[k].Tier)
		}
	}
	for _, k := range overlay {
		if byKey[k].State != ctxresidency.StateHeld {
			t.Errorf("overlay block %+v ended in state %v, want held (paged out)", k, byKey[k].State)
		}
	}
	if snap.Resident != len(spine)+len(policy) {
		t.Errorf("resident count = %d, want %d (the spine + policy floor)", snap.Resident, len(spine)+len(policy))
	}
}

// TestClassifyTierNeverPagesLoadBearing is the by-construction property over the
// classifier itself: enumerate every BlockProfile and assert that ANY
// safety-load-bearing OR identity-load-bearing block resolves to a never-paged
// tier — it can never land in the pageable overlay, no matter its size or access
// frequency. This is the structural fix for silent under-retrieval: the safety
// floor is excluded from the evictable set by the classifier, not by a heuristic.
func TestClassifyTierNeverPagesLoadBearing(t *testing.T) {
	bools := []bool{false, true}
	for _, used := range bools {
		for _, small := range bools {
			for _, safety := range bools {
				for _, identity := range bools {
					p := ctxresidency.BlockProfile{
						UsedEveryTurn: used, Small: small,
						SafetyLoadBearing: safety, IdentityLoadBearing: identity,
					}
					got := ctxresidency.ClassifyTier(p)
					loadBearing := safety || identity
					if loadBearing && got == ctxresidency.TierOverlay {
						t.Errorf("ClassifyTier(%+v) = overlay — a load-bearing block was made pageable", p)
					}
					if loadBearing && !got.AlwaysResident() {
						t.Errorf("ClassifyTier(%+v) = %q, which is not always-resident", p, got)
					}
					if !loadBearing && got != ctxresidency.TierOverlay {
						t.Errorf("ClassifyTier(%+v) = %q, want overlay for a non-load-bearing block", p, got)
					}
					// The spine is the tight resident slot: identity ∧ small ∧ used.
					if got == ctxresidency.TierSpine && !(identity && small && used) {
						t.Errorf("ClassifyTier(%+v) = spine without identity∧small∧used", p)
					}
				}
			}
		}
	}
}

// TestRefuseIfCrossesSpine proves the explicit eviction guard: measuring the
// blast radius of a spine/policy block reports refused=true (the eviction would
// cross the never-paged head), while an overlay block reports refused=false with
// its real measured cost. This is the first-class refusal a Rung-5 promote/demote
// verb calls before acting on a block.
func TestRefuseIfCrossesSpine(t *testing.T) {
	cr := ctxresidency.NewCapResidency(ctxmmu.New())

	spineKey := ctxresidency.CapKey{Kind: "spine", Name: "id", Version: "v1"}
	overlayKey := ctxresidency.CapKey{Kind: "skill", Name: "card", Version: "v1"}
	cr.Fault(spineKey, "d1", []byte("identity spine body"), nil)
	cr.Fault(overlayKey, "d2", []byte("overlay card body bytes"), nil)
	cr.SetTier(spineKey, ctxresidency.TierSpine)
	cr.SetTier(overlayKey, ctxresidency.TierOverlay)

	if radius, refused := cr.RefuseIfCrossesSpine(spineKey); !refused {
		t.Errorf("spine block eviction not refused (radius %+v) — the spine can be paged out", radius)
	} else if radius.Tokens != len("identity spine body") {
		t.Errorf("refused spine block still measured the wrong cost: %+v", radius)
	}
	if radius, refused := cr.RefuseIfCrossesSpine(overlayKey); refused {
		t.Errorf("overlay block eviction refused (radius %+v) — the pageable tier must be evictable", radius)
	} else if radius.Tokens != len("overlay card body bytes") {
		t.Errorf("overlay block measured the wrong cost: %+v", radius)
	}

	// An unknown key is not refused and reports a zero radius (nothing to cross).
	if radius, refused := cr.RefuseIfCrossesSpine(ctxresidency.CapKey{Kind: "skill", Name: "ghost"}); refused || radius != (ctxresidency.BlastRadius{}) {
		t.Errorf("unknown key: refused=%v radius=%+v, want false/zero", refused, radius)
	}
}

// TestEvictColdestOverlayMeasuresBeforeDrop ties the issue's three asks together
// on the overlay path: EvictColdestOverlay pages out the coldest overlay body
// through the witness gate, never a spine block, and the measured cost it returns
// is the real blast radius (recorded for Rung 6), read BEFORE the drop.
func TestEvictColdestOverlayMeasuresBeforeDrop(t *testing.T) {
	ctx := context.Background()
	mmu := ctxmmu.New()
	cr := ctxresidency.NewCapResidency(mmu)

	spineKey := ctxresidency.CapKey{Kind: "spine", Name: "id", Version: "v1"}
	cold := ctxresidency.CapKey{Kind: "skill", Name: "cold", Version: "v1"}
	cr.Fault(spineKey, "d1", []byte("safety/identity spine body, never paged"), nil)
	cr.Fault(cold, "d2", []byte("cold overlay body"), nil)
	cr.SetTier(spineKey, ctxresidency.TierSpine)

	want := cr.MeasureBlastRadius(cold) // the cost measured before any eviction.
	evicted, radius, ok := cr.EvictColdestOverlay(ctx)
	if !ok {
		t.Fatal("EvictColdestOverlay found nothing to page out, want the cold overlay body")
	}
	if evicted != cold {
		t.Fatalf("paged out %+v, want the overlay block %+v (never the spine)", evicted, cold)
	}
	if radius != want {
		t.Errorf("recorded blast radius %+v != the radius measured before the drop %+v", radius, want)
	}
}
