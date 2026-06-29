package ctxresidency_test

import (
	"context"
	"testing"

	_ "github.com/anthony-chaudhary/fak/internal/blob" // registers the "blob" PageOut backend the gate pages bodies through.
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/ctxresidency"
)

// skillKey builds the capindex.CapRef identity the residency tracker keys on.
func skillKey(name, version string) ctxresidency.CapKey {
	return ctxresidency.CapKey{Kind: "skill", Name: name, Version: version}
}

// TestEvictColdestEvictablePicksColdestNeverPinned is the headline C3 acceptance
// witness (#1106): four capabilities are faulted in, one pinned (in-flight), one
// holding a live dependent (resident), two clean evictable. Under pressure the
// tracker must evict the COLDEST EVICTABLE one — never the pinned one, never the
// resident one, never a warmer evictable one — and report the MEASURED blast
// radius of the drop.
func TestEvictColdestEvictablePicksColdestNeverPinned(t *testing.T) {
	ctx := context.Background()
	mmu := ctxmmu.New()
	cr := ctxresidency.NewCapResidency(mmu)

	cold := skillKey("cold", "v1")   // faulted first -> coldest evictable.
	warm := skillKey("warm", "v1")   // faulted later -> warmer evictable.
	held := skillKey("held", "v1")   // pinned in-flight -> never evicted.
	dep := skillKey("dep", "v1")     // has a live dependent -> resident, not evicted.
	child := skillKey("child", "v1") // the dependent dep would invalidate.

	// Fault order sets the coldness clock (cold is oldest).
	cr.Fault(cold, "sha256:cold", []byte("cold body, 16 bytes!"), nil)
	cr.Fault(dep, "sha256:dep", []byte("dep body"), []ctxresidency.CapKey{child})
	cr.Fault(held, "sha256:held", []byte("held body bytes here"), nil)
	cr.Fault(warm, "sha256:warm", []byte("warm body"), nil)

	cr.Pin(held) // an in-flight invocation pins it.

	// MEASURED, not guessed: the cost reported before the evict must equal the
	// cold body's resident tokens and its dependent count (0).
	measured := cr.MeasureBlastRadius(cold)
	if measured.Tokens != len("cold body, 16 bytes!") || measured.DependentEntries != 0 {
		t.Fatalf("measured blast radius = %+v, want {%d,0}", measured, len("cold body, 16 bytes!"))
	}
	// The resident dep's measured radius reports the dependent an evict WOULD drop.
	depRadius := cr.MeasureBlastRadius(dep)
	if depRadius.DependentEntries != 1 {
		t.Fatalf("dep blast radius dependents = %d, want 1 (the child it invalidates)", depRadius.DependentEntries)
	}

	evicted, radius, ok := cr.EvictColdest(ctx)
	if !ok {
		t.Fatal("EvictColdest found nothing evictable, want the cold capability")
	}
	if evicted != cold {
		t.Errorf("evicted %+v, want the COLDEST evictable %+v (not pinned/resident/warm)", evicted, cold)
	}
	// The returned radius is the measured cost, not a heuristic score.
	if radius != measured {
		t.Errorf("evict radius = %+v, want the measured %+v", radius, measured)
	}

	snap := cr.Snapshot()
	byKey := map[ctxresidency.CapKey]ctxresidency.CapRow{}
	for _, row := range snap.Caps {
		byKey[row.Key] = row
	}
	// cold is now HELD (evicted, paged out through the witness gate).
	if byKey[cold].State != ctxresidency.StateHeld {
		t.Errorf("cold should be HELD after eviction, got %v", byKey[cold].State)
	}
	if byKey[cold].PageID == "" {
		t.Errorf("evicted cold should carry a ctxmmu held id (witnessed page-out), got empty")
	}
	// held (pinned) and dep (resident) are untouched.
	if byKey[held].State != ctxresidency.StateResident {
		t.Errorf("pinned held should be RESIDENT (never evicted), got %v", byKey[held].State)
	}
	if byKey[dep].State != ctxresidency.StateResident {
		t.Errorf("dep (has dependent) should be RESIDENT (never evicted), got %v", byKey[dep].State)
	}
	if byKey[warm].State != ctxresidency.StateEvictable {
		t.Errorf("warm (clean, warmer) should still be EVICTABLE (untouched), got %v", byKey[warm].State)
	}
	if snap.Held != 1 || snap.Resident != 2 || snap.Evictable != 1 {
		t.Errorf("snapshot counts = held %d/resident %d/evictable %d, want 1/2/1", snap.Held, snap.Resident, snap.Evictable)
	}
	// Witness: the gate's own capability page-out counter moved with the evict.
	if snap.MMUCapPaged < 1 {
		t.Errorf("MMUCapPaged=%d, want >=1 (the gate witnessed the page-out)", snap.MMUCapPaged)
	}
}

// TestEvictThenReFaultRoundTripsWitnessGate proves the re-fault round-trip: an
// evicted capability's body pages back IN only through the ctxmmu witness gate
// (a Clear() then PageInBody), and a re-Fault restores it to resident. This is
// the "re-query is a FAULT" half of the working set — an evicted body never
// silently reappears; it re-pages through the witness.
func TestEvictThenReFaultRoundTripsWitnessGate(t *testing.T) {
	ctx := context.Background()
	mmu := ctxmmu.New()
	cr := ctxresidency.NewCapResidency(mmu)

	k := skillKey("paged", "v1")
	body := []byte("a real skill body that pages out to CAS")
	cr.Fault(k, "sha256:paged", body, nil)

	evicted, _, ok := cr.EvictColdest(ctx)
	if !ok || evicted != k {
		t.Fatalf("evict %+v ok=%v, want the only capability", evicted, ok)
	}
	snap := cr.Snapshot()
	var pageID string
	for _, row := range snap.Caps {
		if row.Key == k {
			pageID = row.PageID
		}
	}
	if pageID == "" {
		t.Fatal("evicted capability has no held id — the page-out was not witnessed")
	}

	// The witness gate refuses a page-in until a Clear() — the same gate that
	// keeps quarantine sealed. An uncleared re-fault must NOT round-trip.
	if _, err := mmu.PageInBody(ctx, pageID); err == nil {
		t.Fatal("PageInBody of an un-cleared evicted body succeeded — the witness gate was bypassed")
	}
	mmu.Clear(pageID)
	got, err := mmu.PageInBody(ctx, pageID)
	if err != nil {
		t.Fatalf("witnessed page-in failed after Clear: %v", err)
	}
	if len(got) != len(body) {
		t.Errorf("paged-in body length = %d, want %d (the round-trip lost bytes)", len(got), len(body))
	}

	// A re-Fault makes the capability resident again (the loader re-paged it).
	cr.Fault(k, "sha256:paged", body, nil)
	snap2 := cr.Snapshot()
	for _, row := range snap2.Caps {
		if row.Key == k {
			if row.State != ctxresidency.StateEvictable {
				t.Errorf("re-faulted capability should be resident/evictable, got %v", row.State)
			}
			if row.PageID != "" {
				t.Errorf("re-faulted capability still carries a held id %q — it is resident, not paged out", row.PageID)
			}
		}
	}
}

// TestEvictNothingWhenAllHeldOrPinned pins the never-evict invariant from the
// other side: when every capability is pinned (in-flight) or resident
// (dependent-holding), EvictColdest evicts NOTHING and reports ok=false — the
// loader degrades to "no eviction possible" rather than dropping a held frame.
func TestEvictNothingWhenAllHeldOrPinned(t *testing.T) {
	ctx := context.Background()
	cr := ctxresidency.NewCapResidency(ctxmmu.New())

	pinned := skillKey("pinned", "v1")
	resident := skillKey("resident", "v1")
	cr.Fault(pinned, "d1", []byte("body one"), nil)
	cr.Fault(resident, "d2", []byte("body two"), []ctxresidency.CapKey{skillKey("c", "v1")})
	cr.Pin(pinned)

	evicted, radius, ok := cr.EvictColdest(ctx)
	if ok {
		t.Errorf("EvictColdest evicted %+v (radius %+v) — nothing should be evictable (all pinned/resident)", evicted, radius)
	}
	if radius != (ctxresidency.BlastRadius{}) {
		t.Errorf("no-evict radius = %+v, want zero", radius)
	}

	// Unpinning the pinned one makes it the sole eviction candidate.
	cr.Unpin(pinned)
	evicted2, _, ok2 := cr.EvictColdest(ctx)
	if !ok2 || evicted2 != pinned {
		t.Errorf("after Unpin, evicted %+v ok=%v, want the now-evictable %+v", evicted2, ok2, pinned)
	}
}

// TestTouchKeepsWarmCapabilityResident proves the LRU coldness is honest: a
// capability faulted first but TOUCHED (a procedural-cache HIT) becomes warmer
// than a later-faulted one, so the eviction pick flips to the genuinely coldest.
func TestTouchKeepsWarmCapabilityResident(t *testing.T) {
	ctx := context.Background()
	cr := ctxresidency.NewCapResidency(ctxmmu.New())

	first := skillKey("first", "v1")
	second := skillKey("second", "v1")
	cr.Fault(first, "d1", []byte("first body"), nil)
	cr.Fault(second, "d2", []byte("second body"), nil)

	// first is older, but a HIT re-uses it -> Touch warms it past second.
	cr.Touch(first)

	evicted, _, ok := cr.EvictColdest(ctx)
	if !ok {
		t.Fatal("expected an eviction")
	}
	if evicted != second {
		t.Errorf("evicted %+v, want %+v — the touched capability must outlive the genuinely coldest", evicted, second)
	}
}
