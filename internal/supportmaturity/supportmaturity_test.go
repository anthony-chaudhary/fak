package supportmaturity

import (
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/covmatrix"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// allSupport mirrors the closed covmatrix.Support vocabulary (covmatrix.go). If a fifth
// value is ever added there, this roster must grow with it — and TestEverySupportValueMaps
// will fail until it does, which is the point: the ladder must stay total over covmatrix.
var allSupport = []covmatrix.Support{
	covmatrix.Undefined,
	covmatrix.Fenced,
	covmatrix.ProofPathOnly,
	covmatrix.Supported,
}

// allPreflightVerdicts mirrors the closed ggufload preflight verdict vocabulary
// (preflight.go). Same contract as allSupport: a new verdict must be added here.
var allPreflightVerdicts = []string{
	ggufload.PreflightReady,
	ggufload.PreflightRefuseTooBig,
	ggufload.PreflightRefuseArch,
	ggufload.PreflightRefuseHeader,
}

// TestEverySupportValueMaps asserts every covmatrix.Support value lowers to exactly one
// VALID rung, and that the four values are distinct (covmatrix lowers losslessly — its
// own ordering is preserved with no two values collapsed onto one rung).
func TestEverySupportValueMaps(t *testing.T) {
	want := map[covmatrix.Support]Rung{
		covmatrix.Undefined:     M0None,
		covmatrix.Fenced:        M1Fenced,
		covmatrix.ProofPathOnly: M3Runs,
		covmatrix.Supported:     M4Correct,
	}
	seen := map[Rung]covmatrix.Support{}
	for _, s := range allSupport {
		got := FromSupport(s)
		if !got.Valid() {
			t.Fatalf("FromSupport(%q) = %v, not a closed M0–M7 rung", s, got)
		}
		if exp, ok := want[s]; !ok {
			t.Fatalf("Support value %q has no expected rung pinned — extend want", s)
		} else if got != exp {
			t.Fatalf("FromSupport(%q) = %s (%s), want %s (%s)", s, got, got.Label(), exp, exp.Label())
		}
		if prev, dup := seen[got]; dup {
			t.Fatalf("lossy lowering: %q and %q both map to %s — covmatrix must lower losslessly", prev, s, got)
		}
		seen[got] = s
	}
	if len(seen) != len(allSupport) {
		t.Fatalf("mapped %d distinct rungs for %d Support values", len(seen), len(allSupport))
	}
}

// TestEveryPreflightVerdictMaps asserts every ggufload preflight verdict lowers to
// exactly one VALID rung. Preflight verdicts MAY share a rung (the two REFUSE_* fences
// both land on M1) — the contract is totality (every verdict maps), not injectivity.
func TestEveryPreflightVerdictMaps(t *testing.T) {
	want := map[string]Rung{
		ggufload.PreflightReady:        M2Loads,
		ggufload.PreflightRefuseTooBig: M1Fenced,
		ggufload.PreflightRefuseArch:   M1Fenced,
		ggufload.PreflightRefuseHeader: M0None,
	}
	for _, v := range allPreflightVerdicts {
		got := FromPreflightVerdict(v)
		if !got.Valid() {
			t.Fatalf("FromPreflightVerdict(%q) = %v, not a closed M0–M7 rung", v, got)
		}
		if exp, ok := want[v]; !ok {
			t.Fatalf("verdict %q has no expected rung pinned — extend want", v)
		} else if got != exp {
			t.Fatalf("FromPreflightVerdict(%q) = %s (%s), want %s (%s)", v, got, got.Label(), exp, exp.Label())
		}
	}
}

// TestCorrectnessClassMaps asserts both compute.CorrectnessClass values witness M4
// (the class is the M4 BAR — bit-exact vs cosine — not a separate rung).
func TestCorrectnessClassMaps(t *testing.T) {
	for _, c := range []compute.CorrectnessClass{compute.Reference, compute.Approx} {
		if got := FromCorrectnessClass(c); got != M4Correct {
			t.Fatalf("FromCorrectnessClass(%s) = %s, want %s (M4 is the correctness bar)", c, got, M4Correct)
		}
	}
}

// TestLoweringBands pins the doctrine's claim about WHERE each sibling vocabulary sits
// on the ladder: covmatrix.Support spans the M0–M4 band (M2 'loads' is owned by the
// preflight verdict, not by covmatrix), and the preflight verdicts span the narrower
// M0–M2 sub-band. If a future mapping drifts out of its declared band, this fails.
func TestLoweringBands(t *testing.T) {
	supLo, supHi := rungSpan(t, func() []Rung {
		out := make([]Rung, 0, len(allSupport))
		for _, s := range allSupport {
			out = append(out, FromSupport(s))
		}
		return out
	}())
	if supLo != M0None || supHi != M4Correct {
		t.Fatalf("covmatrix.Support band = [%s,%s], want [M0,M4]", supLo, supHi)
	}

	preLo, preHi := rungSpan(t, func() []Rung {
		out := make([]Rung, 0, len(allPreflightVerdicts))
		for _, v := range allPreflightVerdicts {
			out = append(out, FromPreflightVerdict(v))
		}
		return out
	}())
	if preLo != M0None || preHi != M2Loads {
		t.Fatalf("preflight band = [%s,%s], want [M0,M2]", preLo, preHi)
	}
}

// rungSpan returns the lowest and highest rung in rs (rs must be non-empty).
func rungSpan(t *testing.T, rs []Rung) (lo, hi Rung) {
	t.Helper()
	if len(rs) == 0 {
		t.Fatalf("rungSpan: empty rung set")
	}
	lo, hi = rs[0], rs[0]
	for _, r := range rs[1:] {
		if r.Less(lo) {
			lo = r
		}
		if hi.Less(r) {
			hi = r
		}
	}
	return lo, hi
}

// TestRungRender asserts every closed rung renders a distinct "M<n>" id matching its
// ordinal and a distinct non-empty doctrine label, and that an out-of-range rung is both
// not-Valid and rendered as the explicit unknown form — the closed-vocabulary guard the
// From* defaults rely on (an unrecognized input floors to M0None, never to a bogus rung).
func TestRungRender(t *testing.T) {
	seenID, seenLabel := map[string]bool{}, map[string]bool{}
	for i, r := range Rungs {
		id := r.String()
		if id != fmt.Sprintf("M%d", i) {
			t.Fatalf("Rungs[%d].String() = %q, want M%d", i, id, i)
		}
		if seenID[id] {
			t.Fatalf("duplicate rung id %q", id)
		}
		seenID[id] = true
		lbl := r.Label()
		if lbl == "" || lbl == "unknown" {
			t.Fatalf("rung %s has no doctrine label", id)
		}
		if seenLabel[lbl] {
			t.Fatalf("duplicate rung label %q", lbl)
		}
		seenLabel[lbl] = true
	}
	bad := Rung(len(Rungs))
	if bad.Valid() {
		t.Fatalf("out-of-range rung %d reports Valid", uint8(bad))
	}
	if got := bad.Label(); got != "unknown" {
		t.Fatalf("out-of-range rung Label() = %q, want \"unknown\"", got)
	}
}

// TestRungOrderIsTotal asserts the ladder is a closed, total order: exactly 8 rungs
// M0..M7, the Rungs roster is strictly increasing, every rung is Valid, and Less is a
// strict total order (irreflexive + trichotomy over every ordered pair).
func TestRungOrderIsTotal(t *testing.T) {
	if len(Rungs) != 8 {
		t.Fatalf("ladder has %d rungs, want a closed 8 (M0..M7)", len(Rungs))
	}
	for i, r := range Rungs {
		if !r.Valid() {
			t.Fatalf("Rungs[%d] = %v is not a valid rung", i, r)
		}
		if int(r) != i {
			t.Fatalf("Rungs[%d] = %s has ordinal %d — roster must be M0..M7 in order", i, r, uint8(r))
		}
		if i > 0 && !Rungs[i-1].Less(r) {
			t.Fatalf("ladder not strictly increasing at %s → %s", Rungs[i-1], r)
		}
	}
	// Trichotomy: for every ordered pair exactly one of a<b, b<a, a==b holds.
	for _, a := range Rungs {
		if a.Less(a) {
			t.Fatalf("Less is not irreflexive: %s < %s", a, a)
		}
		for _, b := range Rungs {
			lt, gt, eq := a.Less(b), b.Less(a), a == b
			n := 0
			for _, p := range []bool{lt, gt, eq} {
				if p {
					n++
				}
			}
			if n != 1 {
				t.Fatalf("trichotomy broken for (%s,%s): lt=%v gt=%v eq=%v", a, b, lt, gt, eq)
			}
		}
	}
}
