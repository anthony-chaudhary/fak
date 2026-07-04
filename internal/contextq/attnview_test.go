package contextq

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/kvmmu"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// attnview_test.go — acceptance for issue #2617 (queryable attention side-car views).
//
// The fixture drives a real kvmmu.Context (Append lays out real spans over a synthetic
// model session, the same machinery the kvmmu bridge tests use) and then sets the
// per-span attention scalars to KNOWN values. The view is a pure read over those
// scalars, so the assertions are hand-computable: the tests pin exactly which spans a
// threshold/hottest/dead-weight predicate returns and which MaterializationVerdict each
// carries — never running a forward pass to produce the numbers.

func attnViewCfg() model.Config {
	return model.Config{
		HiddenSize: 32, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 48, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1, ModelType: "llama",
	}
}

// drivenCtx builds a Context with four spans A/B/C/D over a fresh synthetic session.
// The model is returned so a test can witness that the view never installs an observer.
func drivenCtx(t *testing.T) (*model.Model, *kvmmu.Context) {
	t.Helper()
	m := model.NewSynthetic(attnViewCfg())
	c := kvmmu.New(m.NewSession())
	c.Append("A", "toolA", []int{1, 2})
	c.Append("B", "toolB", []int{3, 4})
	c.Append("C", "toolC", []int{5, 6})
	c.Append("D", "toolD", []int{7, 8})
	return m, c
}

// setSeg applies mut to the segment with the given id (exported scalars are set
// directly so the fixture values are hand-chosen, not attention-derived).
func setSeg(c *kvmmu.Context, id string, mut func(*kvmmu.Segment)) {
	for _, s := range c.Segments() {
		if s.ID == id {
			mut(s)
		}
	}
}

func refIDs(refs []SpanScalarRef) []string {
	ids := make([]string, len(refs))
	for i, r := range refs {
		ids[i] = r.SpanID
	}
	return ids
}

func eqIDs(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestThresholdViewExactSpans is the core acceptance: a threshold view (a_s > θ) over
// a driven Context returns EXACTLY the spans above θ, strongest-first, each stamped HIT
// while resident. Attended fixture: A=0.9, B=0.1, C=0.5, D=0.0; θ=0.4 selects {A, C}.
func TestThresholdViewExactSpans(t *testing.T) {
	m, c := drivenCtx(t)
	setSeg(c, "A", func(s *kvmmu.Segment) { s.Attended = 0.9 })
	setSeg(c, "B", func(s *kvmmu.Segment) { s.Attended = 0.1 })
	setSeg(c, "C", func(s *kvmmu.Segment) { s.Attended = 0.5 })
	setSeg(c, "D", func(s *kvmmu.Segment) { s.Attended = 0.0 })

	refs := ThresholdView(c, ScalarAttended, 0.4)
	if got := refIDs(refs); !eqIDs(got, []string{"A", "C"}) {
		t.Fatalf("threshold θ=0.4 spans = %v, want [A C] (strongest-first)", got)
	}
	if refs[0].Value != 0.9 || refs[1].Value != 0.5 {
		t.Fatalf("values = %v/%v, want 0.9/0.5", refs[0].Value, refs[1].Value)
	}
	for _, r := range refs {
		if r.Verdict.Kind != MaterializationHit {
			t.Fatalf("resident span %s verdict = %q, want HIT", r.SpanID, r.Verdict.Kind)
		}
	}
	// Zero-forward-pass invariant: the view read scalars only — it never installed an
	// attention observer on the model (no AttnObserver re-run).
	if m.AttnObserverSet() {
		t.Fatalf("view installed an AttnObserver — the view must be a pure scalar read")
	}
}

// TestHeldSpanFaults: a span whose K/V was evicted (Held, Len 0) still matches by its
// surviving rolling mass, but its ref is FAULT — rendering it must page the bytes back.
func TestHeldSpanFaults(t *testing.T) {
	_, c := drivenCtx(t)
	setSeg(c, "A", func(s *kvmmu.Segment) { s.EMA = 0.8 })
	setSeg(c, "C", func(s *kvmmu.Segment) {
		s.EMA = 0.6
		s.Held = true
		s.Len = 0
	})

	refs := ThresholdView(c, ScalarEMA, 0.5)
	if got := refIDs(refs); !eqIDs(got, []string{"A", "C"}) {
		t.Fatalf("EMA θ=0.5 spans = %v, want [A C]", got)
	}
	byID := map[string]SpanScalarRef{refs[0].SpanID: refs[0], refs[1].SpanID: refs[1]}
	if byID["A"].Verdict.Kind != MaterializationHit || !byID["A"].Resident {
		t.Fatalf("A verdict = %q resident=%v, want HIT/resident", byID["A"].Verdict.Kind, byID["A"].Resident)
	}
	if byID["C"].Verdict.Kind != MaterializationFault || byID["C"].Resident {
		t.Fatalf("held C verdict = %q resident=%v, want FAULT/not-resident", byID["C"].Verdict.Kind, byID["C"].Resident)
	}
}

// TestQuarantinedSpanRefused: a span whose KV entry is TaintQuarantined is REFUSED for
// free via taint inheritance, even when resident and above the threshold.
func TestQuarantinedSpanRefused(t *testing.T) {
	_, c := drivenCtx(t)
	qid := cachemeta.EntryID{Digest: "poison-span", Length: 2}
	c.TrackEntry(cachemeta.Entry{ID: qid, Security: cachemeta.Security{Taint: abi.TaintQuarantined}})
	setSeg(c, "A", func(s *kvmmu.Segment) { s.Attended = 0.9 })
	setSeg(c, "C", func(s *kvmmu.Segment) {
		s.Attended = 0.7
		s.KV = qid
	})

	refs := ThresholdView(c, ScalarAttended, 0.4)
	byID := map[string]SpanScalarRef{}
	for _, r := range refs {
		byID[r.SpanID] = r
	}
	if _, ok := byID["C"]; !ok {
		t.Fatalf("quarantined C should still appear in the view (as REFUSE), got %v", refIDs(refs))
	}
	if byID["C"].Verdict.Kind != MaterializationRefuse {
		t.Fatalf("quarantined C verdict = %q, want REFUSE", byID["C"].Verdict.Kind)
	}
	if byID["A"].Verdict.Kind != MaterializationHit {
		t.Fatalf("benign resident A verdict = %q, want HIT", byID["A"].Verdict.Kind)
	}
}

// TestHottestView returns the top-n by the named scalar, descending.
func TestHottestView(t *testing.T) {
	_, c := drivenCtx(t)
	setSeg(c, "A", func(s *kvmmu.Segment) { s.Cumulative = 0.2 })
	setSeg(c, "B", func(s *kvmmu.Segment) { s.Cumulative = 0.8 })
	setSeg(c, "C", func(s *kvmmu.Segment) { s.Cumulative = 0.5 })
	setSeg(c, "D", func(s *kvmmu.Segment) { s.Cumulative = 0.0 }) // zero mass — never a candidate

	refs := HottestView(c, ScalarCumulative, 2)
	if got := refIDs(refs); !eqIDs(got, []string{"B", "C"}) {
		t.Fatalf("hottest top-2 by Cumulative = %v, want [B C]", got)
	}
}

// TestDeadWeightView returns cold-but-resident spans coldest-first and excludes Held.
func TestDeadWeightView(t *testing.T) {
	_, c := drivenCtx(t)
	setSeg(c, "A", func(s *kvmmu.Segment) { s.Attended = 0.9 })                            // hot, excluded
	setSeg(c, "B", func(s *kvmmu.Segment) { s.Attended = 0.10 })                           // cold, resident
	setSeg(c, "C", func(s *kvmmu.Segment) { s.Attended = 0.00 })                           // coldest, resident
	setSeg(c, "D", func(s *kvmmu.Segment) { s.Attended = 0.05; s.Held = true; s.Len = 0 }) // cold but Held — excluded

	refs := DeadWeightView(c, ScalarAttended, 0.1)
	if got := refIDs(refs); !eqIDs(got, []string{"C", "B"}) {
		t.Fatalf("dead-weight θ=0.1 spans = %v, want [C B] (coldest-first, Held excluded)", got)
	}
	for _, r := range refs {
		if r.Verdict.Kind != MaterializationHit {
			t.Fatalf("resident dead-weight %s verdict = %q, want HIT", r.SpanID, r.Verdict.Kind)
		}
	}
}

// TestNilContext is a defensive no-op.
func TestNilContext(t *testing.T) {
	if refs := ThresholdView(nil, ScalarAttended, 0); refs != nil {
		t.Fatalf("nil context = %v, want nil", refs)
	}
}
