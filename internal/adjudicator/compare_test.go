package adjudicator

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestCompareLocalKeepsExternalEnginesExplicit(t *testing.T) {
	report := CompareLocal(10)
	if report.Schema != ComparisonSchema || report.Complete {
		t.Fatalf("schema/complete=%q/%v", report.Schema, report.Complete)
	}
	if len(report.Arms) != 4 {
		t.Fatalf("arms=%d, want native, baseline, OPA, Cedar", len(report.Arms))
	}
	for _, arm := range report.Arms[:2] {
		if !arm.Available || arm.Correctness != 1 {
			t.Fatalf("local arm=%+v", arm)
		}
	}
	for _, arm := range report.Arms[2:] {
		if arm.Available || arm.UnavailableReason == "" {
			t.Fatalf("external arm must remain honestly unavailable: %+v", arm)
		}
	}
}

func TestComparisonCorpusExercisesDifferentPolicyRungs(t *testing.T) {
	report := CompareLocal(1)
	if got, want := report.Arms[0].Calls, len(ComparisonCorpus()); got != want {
		t.Fatalf("calls=%d, want %d", got, want)
	}
}

// TestTunedBaselineIsDerivedFromThePolicy — the no-engine baseline reads its verdicts
// out of the SAME Policy the native arm evaluates, so it cannot drift when that policy
// grows an entry. A hand-maintained switch would answer Deny for a newly allowed tool
// and quietly report the BASELINE as wrong when it was the fixture that moved.
func TestTunedBaselineIsDerivedFromThePolicy(t *testing.T) {
	p := comparisonPolicy()
	// read_ticket is allowed only through AllowPrefix "read_", never by an exact entry.
	if p.Allow["read_ticket"] {
		t.Fatal("fixture drift: read_ticket must be prefix-allowed, not exact-allowed")
	}
	if got := directLookupKind(p, "read_ticket"); got != abi.VerdictAllow {
		t.Errorf("prefix-allowed tool: got %v, want Allow", got)
	}
	if got := directLookupKind(p, "refund_payment"); got != abi.VerdictDeny {
		t.Errorf("denied tool: got %v, want Deny", got)
	}

	// A policy that grows an allow entry is followed, with no baseline edit.
	widened := comparisonPolicy()
	widened.Allow["newly_allowed"] = true
	if got := directLookupKind(widened, "newly_allowed"); got != abi.VerdictAllow {
		t.Errorf("widened policy: got %v, want Allow", got)
	}
	if got := directLookupKind(p, "newly_allowed"); got != abi.VerdictDeny {
		t.Errorf("original policy must be unaffected: got %v, want Deny", got)
	}

	// The precomputed table covers every corpus tool, so the timed loop is a bare hit.
	corpus := ComparisonCorpus()
	table := tunedBaselineLookup(p, corpus)
	for i := range corpus {
		if _, ok := table[corpus[i].Call.Tool]; !ok {
			t.Errorf("corpus tool %q missing from the precomputed baseline table", corpus[i].Call.Tool)
		}
	}
}

func BenchmarkPolicyAdjudicationComparison(b *testing.B) {
	a := New(comparisonPolicy())
	corpus := ComparisonCorpus()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := range corpus {
			_ = a.Adjudicate(b.Context(), &corpus[j].Call)
		}
	}
}
