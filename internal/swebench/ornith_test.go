package swebench

import (
	"testing"
	"time"
)

// A freshly-built record holds only the vendor's OBSERVED number; nothing fak
// served has touched it, so it must report OBSERVED-ONLY and an unset witness.
func TestNewOrnithRecordIsObservedOnly(t *testing.T) {
	r := NewOrnithRecord("astropy__astropy-12907", true)
	if r.FakWitnessedPass != nil {
		t.Fatalf("fresh record has a witnessed result: %+v", r.FakWitnessedPass)
	}
	if r.Witnessed() {
		t.Fatal("fresh record reports Witnessed() true")
	}
	if got := r.Provenance(); got != OrnithObservedOnly {
		t.Fatalf("fresh record provenance = %q, want %q", got, OrnithObservedOnly)
	}
	if !r.VendorObservedPass {
		t.Fatal("vendor observed pass not preserved")
	}
}

// Recording a real fak-served outcome flips the record to WITNESSED without
// disturbing the OBSERVED vendor field.
func TestSetWitnessedFlipsProvenance(t *testing.T) {
	r := NewOrnithRecord("django__django-11099", true)
	at := time.Unix(1_700_000_000, 0).UTC()
	r.SetWitnessed(false, at, "runs/ornith-9b/preds.jsonl")

	if got := r.Provenance(); got != OrnithWitnessed {
		t.Fatalf("after SetWitnessed provenance = %q, want %q", got, OrnithWitnessed)
	}
	if !r.Witnessed() {
		t.Fatal("after SetWitnessed Witnessed() is false")
	}
	if r.FakWitnessedPass == nil || *r.FakWitnessedPass != false {
		t.Fatalf("witnessed pass = %v, want a stored false", r.FakWitnessedPass)
	}
	// The witnessed result is independent of the vendor's: vendor said pass,
	// fak's run said fail — both are preserved, neither overwrites the other.
	if !r.VendorObservedPass {
		t.Fatal("vendor observed result was clobbered by the witnessed run")
	}
	if r.WitnessedAt == nil || !r.WitnessedAt.Equal(at) {
		t.Fatalf("witnessed-at = %v, want %v", r.WitnessedAt, at)
	}
	if r.RunRef != "runs/ornith-9b/preds.jsonl" {
		t.Fatalf("run ref = %q, not recorded", r.RunRef)
	}
}

// A reproduction set reports an honest OBSERVED-vs-WITNESSED headline: zero
// witnessed until a real run lands, then it climbs.
func TestOrnithReproStatus(t *testing.T) {
	o := NewOrnithRepro("ornith-9b", "swebench-verified")
	o.Add("t-1", true)
	r2 := o.Add("t-2", false)
	o.Add("t-3", true)

	st := o.Status()
	if st.Total != 3 {
		t.Fatalf("total = %d, want 3", st.Total)
	}
	if st.Witnessed != 0 {
		t.Fatalf("witnessed = %d before any real run, want 0", st.Witnessed)
	}
	if st.ObservedOnly != 3 {
		t.Fatalf("observed-only = %d, want 3", st.ObservedOnly)
	}
	if st.VendorObservedOK != 2 {
		t.Fatalf("vendor observed ok = %d, want 2", st.VendorObservedOK)
	}
	if st.FakWitnessedOK != 0 {
		t.Fatalf("fak witnessed ok = %d before any run, want 0", st.FakWitnessedOK)
	}
	if o.FullyWitnessed() {
		t.Fatal("FullyWitnessed true with no witnessed records")
	}

	// One real fak-served run lands a pass on t-2.
	r2.SetWitnessed(true, time.Now(), "runs/ornith-9b/preds.jsonl")
	st = o.Status()
	if st.Witnessed != 1 {
		t.Fatalf("witnessed = %d after one run, want 1", st.Witnessed)
	}
	if st.ObservedOnly != 2 {
		t.Fatalf("observed-only = %d after one witness, want 2", st.ObservedOnly)
	}
	if st.FakWitnessedOK != 1 {
		t.Fatalf("fak witnessed ok = %d, want 1", st.FakWitnessedOK)
	}
	if got := st.WitnessedFrac; got < 0.33 || got > 0.34 {
		t.Fatalf("witnessed frac = %v, want ~0.333", got)
	}
}

func TestOrnithTicketIDsSorted(t *testing.T) {
	o := NewOrnithRepro("ornith-9b", "swebench-verified")
	o.Add("zeta", true)
	o.Add("alpha", false)
	o.Add("mu", true)
	got := o.TicketIDs()
	want := []string{"alpha", "mu", "zeta"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("ticket ids = %v, want %v", got, want)
		}
	}
}
