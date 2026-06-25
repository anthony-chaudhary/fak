package shipgate

// evidence_profile_test.go closes the graduated keep-bit acceptance criteria of
// issue #680: the zero-value ClassFull default is byte-identical to the legacy
// all-three AND (criterion 1), ClassDocsOnly keeps on truth-clean alone (2),
// ClassProofCarrying keeps on gain + truth-clean (3), no class sets the keep-bit
// from an unmeasured input (4), and an unprovable narrower class falls back to
// ClassFull via the harness gate ClassifyPaths (7).

import (
	"math/rand"
	"testing"
	"testing/quick"
)

// criterion 1: the zero-value Witness (Class unset => ClassFull) yields a keep
// decision byte-identical to the legacy all-three AND for every input. Proven as
// an equivalence over a fixed-seed sweep AND as a direct equality with the literal
// legacy expression.
func TestClassFullDefaultIsLegacyAllThree(t *testing.T) {
	r := rand.New(rand.NewSource(0x680_680))
	pick := func() float64 {
		switch r.Intn(5) {
		case 0:
			return r.Float64() * 2000
		case 1:
			return 0
		default:
			return r.NormFloat64()
		}
	}
	for i := 0; i < 5000; i++ {
		before, after := pick(), pick()
		if r.Intn(4) == 0 {
			after = before
		}
		lower, green, clean := r.Intn(2) == 0, r.Intn(2) == 0, r.Intn(2) == 0
		var w Witness // Class is the zero value ClassFull
		w.Before, w.After, w.LowerBetter, w.SuiteGreen, w.TruthClean = before, after, lower, green, clean
		d, out := Evaluate(w)
		legacy := w.improved() && w.SuiteGreen && w.TruthClean
		if out.Kept() != legacy {
			t.Fatalf("case %d: ClassFull default != legacy AND: kept=%v legacy=%v for %+v", i, out.Kept(), legacy, w)
		}
		if (d == KEEP) != legacy {
			t.Fatalf("case %d: ClassFull decision != legacy AND: d=%v legacy=%v for %+v", i, d, legacy, w)
		}
	}
}

// criterion 2: a ClassDocsOnly witness KEEPs on truth-clean ALONE - even with a red
// suite and no metric gain - and REVERTs when truth is dirty.
func TestClassDocsOnlyKeepsOnTruthCleanAlone(t *testing.T) {
	keep := Witness{Class: ClassDocsOnly, Before: 1000, After: 1000, SuiteGreen: false, TruthClean: true}
	if d, out := Evaluate(keep); d != KEEP || !out.Kept() {
		t.Fatalf("docs-only + truth-clean must KEEP (suite/gain irrelevant), got %v kept=%v", d, out.Kept())
	}
	revert := Witness{Class: ClassDocsOnly, Before: 1, After: 2, SuiteGreen: true, TruthClean: false}
	if d, out := Evaluate(revert); d != REVERT || out.Kept() {
		t.Fatalf("docs-only + dirty truth must REVERT, got %v kept=%v", d, out.Kept())
	}
}

// criterion 3: a ClassProofCarrying witness KEEPs on a strict gain AND truth-clean
// - even with a red suite - and REVERTs if either required signal is missing.
func TestClassProofCarryingKeepsOnGainAndTruthClean(t *testing.T) {
	keep := Witness{Class: ClassProofCarrying, Before: 1000, After: 800, LowerBetter: true, SuiteGreen: false, TruthClean: true}
	if d, out := Evaluate(keep); d != KEEP || !out.Kept() {
		t.Fatalf("proof-carrying + gain + truth-clean must KEEP (suite irrelevant), got %v kept=%v", d, out.Kept())
	}
	noGain := Witness{Class: ClassProofCarrying, Before: 1000, After: 1000, SuiteGreen: false, TruthClean: true}
	if d, _ := Evaluate(noGain); d != REVERT {
		t.Fatalf("proof-carrying with NO gain must REVERT, got %v", d)
	}
	dirtyTruth := Witness{Class: ClassProofCarrying, Before: 1000, After: 800, LowerBetter: true, SuiteGreen: false, TruthClean: false}
	if d, _ := Evaluate(dirtyTruth); d != REVERT {
		t.Fatalf("proof-carrying with dirty truth must REVERT, got %v", d)
	}
}

// criterion 4: no class can produce Kept()==true from an unmeasured input - Class
// alone never sets the keep-bit. With every measured signal false, EVERY class
// (including an unrecognized value) must REVERT, because every profile still
// requires the truth-clean signal.
func TestNoClassKeepsFromUnmeasuredInput(t *testing.T) {
	for _, c := range []EvidenceClass{ClassFull, ClassDocsOnly, ClassProofCarrying, EvidenceClass(99)} {
		w := Witness{Class: c} // Before==After (no gain), SuiteGreen=false, TruthClean=false
		d, out := Evaluate(w)
		if d != REVERT || out.Kept() {
			t.Fatalf("class %s: keep-bit set from unmeasured input: d=%v kept=%v", c, d, out.Kept())
		}
	}
}

// criterion 4 (property form): the keep-bit is a pure function of the profile's
// named measured signals. For every class, Kept() == "all signals the profile
// requires hold", computed independently from ProfileFor - Class (the only
// non-measured field) never moves the bit on its own.
func TestKeptIsPureFunctionOfProfileSignals(t *testing.T) {
	prop := func(classByte uint8, gain, suite, truth bool) bool {
		c := EvidenceClass(classByte % 4) // 0..3 => Full/DocsOnly/ProofCarrying/(unknown)
		before, after := 0.0, 0.0
		if gain {
			after = 1.0 // After>Before under LowerBetter=false => improved()
		}
		w := Witness{Class: c, Before: before, After: after, SuiteGreen: suite, TruthClean: truth}
		_, out := Evaluate(w)
		p := ProfileFor(c)
		want := (!p.needGain || w.improved()) && (!p.needSuite || w.SuiteGreen) && (!p.needTruth || w.TruthClean)
		return out.Kept() == want
	}
	cfg := &quick.Config{MaxCount: 5000, Rand: rand.New(rand.NewSource(0xC1a55))}
	if err := quick.Check(prop, cfg); err != nil {
		t.Fatalf("keep-bit not a pure function of profile signals: %v", err)
	}
}

// An unrecognized EvidenceClass value falls back to ClassFull: it produces the same
// (decision, keep-bit) as an explicit ClassFull for the same measured witness, and
// the same profile (all-three) is returned by ProfileFor.
func TestUnknownClassFallsBackToFull(t *testing.T) {
	unknown := EvidenceClass(99)
	if pf, pfull := ProfileFor(unknown), ProfileFor(ClassFull); pf != pfull {
		t.Fatalf("ProfileFor(unknown) != ProfileFor(ClassFull): %+v != %+v", pf, pfull)
	}
	w := Witness{Class: unknown, Before: 1000, After: 800, LowerBetter: true, SuiteGreen: true, TruthClean: true}
	dUnk, outUnk := Evaluate(w)
	w.Class = ClassFull
	dFull, outFull := Evaluate(w)
	if dUnk != dFull || outUnk.Kept() != outFull.Kept() {
		t.Fatalf("unknown class did not fall back to ClassFull: unk=(%v,%v) full=(%v,%v)",
			dUnk, outUnk.Kept(), dFull, outFull.Kept())
	}
}

// criterion 7 (the gate): ClassifyPaths declares ClassDocsOnly ONLY when every
// touched path is a doc path; any code path, a mixed change, an empty path set, or
// a nil predicate falls back to ClassFull - the harness refuses to apply a
// docs-only profile to a candidate it cannot prove is docs-only.
func TestClassifyPathsRefusesUnprovableDocsOnly(t *testing.T) {
	isDoc := func(p string) bool {
		return p == "docs/a.md" || p == "README.md"
	}
	cases := []struct {
		name  string
		paths []string
		want  EvidenceClass
	}{
		{"all docs", []string{"docs/a.md", "README.md"}, ClassDocsOnly},
		{"single doc", []string{"README.md"}, ClassDocsOnly},
		{"one code path", []string{"docs/a.md", "internal/shipgate/shipgate.go"}, ClassFull},
		{"all code", []string{"a.go", "b.go"}, ClassFull},
		{"mixed", []string{"README.md", "main.go"}, ClassFull},
		{"empty path set", nil, ClassFull},
	}
	for _, tc := range cases {
		if got := ClassifyPaths(tc.paths, isDoc); got != tc.want {
			t.Fatalf("ClassifyPaths(%s)=%s want %s", tc.name, got, tc.want)
		}
	}
	if got := ClassifyPaths([]string{"README.md"}, nil); got != ClassFull {
		t.Fatalf("ClassifyPaths with nil predicate must fall back to ClassFull, got %s", got)
	}
}
