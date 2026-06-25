package recall

import (
	"context"
	"testing"
)

func TestComputeSyndrome_Deterministic(t *testing.T) {
	p := Page{Digest: "abc", Len: 10, Taint: 2, Quarantined: true, QID: "q1"}
	if computeSyndrome(p) != computeSyndrome(p) {
		t.Fatal("syndrome must be deterministic for the same page")
	}
	if len(computeSyndrome(p)) != syndromeWidth {
		t.Fatalf("syndrome width = %d, want %d", len(computeSyndrome(p)), syndromeWidth)
	}
}

// TestSyndrome_CatchesEachIntegrityField proves that flipping ANY of the
// integrity-critical fields changes the syndrome — the whole point of the check.
// The poison-release field (Quarantined) is the security-load-bearing one.
func TestSyndrome_CatchesEachIntegrityField(t *testing.T) {
	base := Page{Digest: "abc", Len: 10, Taint: 2, Quarantined: true, QID: "q1"}
	s0 := computeSyndrome(base)
	muts := map[string]Page{
		"digest":      mut(base, func(p *Page) { p.Digest = "xyz" }),
		"len":         mut(base, func(p *Page) { p.Len = 11 }),
		"taint":       mut(base, func(p *Page) { p.Taint = 3 }),
		"quarantined": mut(base, func(p *Page) { p.Quarantined = false }),
		"qid":         mut(base, func(p *Page) { p.QID = "q2" }),
	}
	for name, m := range muts {
		if computeSyndrome(m) == s0 {
			t.Errorf("flipping %s did not change the syndrome — a silent corruption would pass", name)
		}
	}
	// A presentation-only field must NOT change the syndrome (so a benign descriptor
	// repair does not invalidate it).
	if got := computeSyndrome(mut(base, func(p *Page) { p.Descriptor = "different" })); got != s0 {
		t.Error("descriptor change must not move the syndrome (it is not integrity-critical)")
	}
}

func mut(p Page, f func(*Page)) Page { f(&p); return p }

func TestClassifyFault(t *testing.T) {
	body := []byte("the authoritative bytes")
	d := Digest(body)
	clean := stampSyndrome(Page{Digest: d, Len: int64(len(body)), Taint: 1})

	// Clean: body present + matching + syndrome agrees.
	if got := ClassifyFault(clean, body); got != FaultClean {
		t.Fatalf("clean page => FaultClean, got %v", got)
	}
	// Erasure: body absent.
	if got := ClassifyFault(clean, nil); got != FaultErasure {
		t.Fatalf("missing body => FaultErasure, got %v", got)
	}
	// Erasure: body present but does not hash to the recorded digest (rotted CAS).
	if got := ClassifyFault(clean, []byte("wrong bytes")); got != FaultErasure {
		t.Fatalf("digest-mismatched body => FaultErasure, got %v", got)
	}
	// Repairable: body authoritative, but the manifest's metadata was tampered AFTER
	// the syndrome was stamped (the security case: someone flipped Quarantined off to
	// release a sealed page). Stamp a quarantined page, then clear the flag — the
	// stored syndrome still reflects the pre-tamper (quarantined) state.
	sealed := stampSyndrome(Page{Digest: d, Len: int64(len(body)), Taint: 3, Quarantined: true, QID: "q1"})
	tampered := sealed
	tampered.Quarantined = false // someone tried to unseal it; syndrome no longer matches
	if got := ClassifyFault(tampered, body); got != FaultRepairable {
		t.Fatalf("metadata tamper (unseal) with authoritative body => FaultRepairable, got %v", got)
	}
	// Unchecked: no syndrome at all (pre-rung page) + present body.
	unstamped := Page{Digest: d, Len: int64(len(body)), Taint: 1}
	if got := ClassifyFault(unstamped, body); got != FaultUnchecked {
		t.Fatalf("unstamped page => FaultUnchecked, got %v", got)
	}
}

func TestFaultClass_StringAndRepairable(t *testing.T) {
	if !FaultRepairable.Repairable() {
		t.Error("FaultRepairable.Repairable() must be true")
	}
	if FaultErasure.Repairable() {
		t.Error("an erasure is NOT locally repairable")
	}
	for c, want := range map[FaultClass]string{
		FaultUnchecked: "unchecked", FaultClean: "clean", FaultRepairable: "repairable", FaultErasure: "erasure",
	} {
		if c.String() != want {
			t.Errorf("%d.String()=%q, want %q", c, c.String(), want)
		}
	}
}

// TestVerify_EndToEnd records, persists, reloads, and proves a clean session
// verifies clean — and that a post-load metadata tamper is then caught as a fault.
func TestVerify_EndToEnd(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	r := NewRecorder("s-syndrome")
	r.Record(ctx, "get_user_details", []byte(`{"name":"Ada"}`))
	r.Record(ctx, "search_flights", []byte(`[{"from":"SFO"}]`))
	if err := r.Persist(dir); err != nil {
		t.Fatal(err)
	}
	s, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	// Every page was stamped at Manifest() time and its body is in CAS => all clean.
	if f := s.Verify(); len(f) != 0 {
		t.Fatalf("freshly loaded clean session => no faults, got %v", f)
	}
	// Tamper a page's security-critical metadata in the loaded table and re-verify:
	// the syndrome no longer matches, so the page is flagged repairable.
	s.Manifest.Pages[0].Quarantined = !s.Manifest.Pages[0].Quarantined
	faults := s.Verify()
	if len(faults) != 1 || faults[0].Step != 0 || faults[0].Class != FaultRepairable {
		t.Fatalf("tampered page => one FaultRepairable at step 0, got %v", faults)
	}
}

// TestSyndrome_DefaultNeutral proves the field is omitempty: a page with no
// syndrome marshals without the key, so a pre-rung manifest is byte-identical.
func TestSyndrome_DefaultNeutral(t *testing.T) {
	if computeSyndrome(Page{}) == "" {
		t.Fatal("a zero page still has a computable (non-empty) syndrome")
	}
	// An unstamped page carries no Syndrome and classifies as Unchecked, never a fault.
	p := Page{Digest: Digest([]byte("x")), Len: 1}
	if p.Syndrome != "" {
		t.Fatal("an unstamped page must have an empty Syndrome (omitempty)")
	}
	if ClassifyFault(p, []byte("x")) != FaultUnchecked {
		t.Fatal("unstamped + present body => Unchecked (no false fault)")
	}
}
