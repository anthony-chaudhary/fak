package steerpr

import (
	"fmt"
	"testing"
)

// bandOrderWorstFirst is the ordering the fold must honor, RESIDUAL >
// UNVERIFIABLE > CLEARED (worst first), written out INDEPENDENTLY of the
// production bandRank so the cross-product oracle below cannot silently agree
// with a broken rank map.
var bandOrderWorstFirst = []Band{BandResidual, BandUnverifiable, BandCleared}

// referenceWorst returns the worst band in bands using bandOrderWorstFirst — a
// hand-rolled oracle that never calls the code under test. A cross-product table
// whose expected column was computed by FoldBand would prove nothing; this is
// the independent witness FoldBand is checked against.
func referenceWorst(bands []Band) Band {
	worstIdx := len(bandOrderWorstFirst) - 1 // start at the best (CLEARED)
	for _, b := range bands {
		for i, ref := range bandOrderWorstFirst {
			if b == ref && i < worstIdx {
				worstIdx = i
			}
		}
	}
	return bandOrderWorstFirst[worstIdx]
}

// commitsWithBands builds members carrying exactly the given cached bands and no
// verdict, so commitBand returns each band verbatim (the Verdict==Unknown path).
func commitsWithBands(bands ...Band) []Commit {
	out := make([]Commit, len(bands))
	for i, b := range bands {
		out[i] = Commit{SHA: fmt.Sprintf("s%d", i), Band: b}
	}
	return out
}

// bandTuples enumerates every ordered length-n tuple over the closed band set —
// the literal cross-product the issue's done-condition names.
func bandTuples(n int) [][]Band {
	if n == 0 {
		return [][]Band{{}}
	}
	var out [][]Band
	for _, head := range bandOrderWorstFirst {
		for _, tail := range bandTuples(n - 1) {
			out = append(out, append([]Band{head}, tail...))
		}
	}
	return out
}

// TestFoldBandFullCrossProduct is the issue's headline: over the FULL
// cross-product of member bands, the unit band equals the worst member. Arity
// 1..3 enumerates every mixed pair and triple in every order, which also proves
// the fold is order-independent (a set fold, not a sequence fold). The expected
// column comes from referenceWorst — an oracle independent of FoldBand.
func TestFoldBandFullCrossProduct(t *testing.T) {
	total := 0
	for n := 1; n <= 3; n++ {
		for _, tuple := range bandTuples(n) {
			want := referenceWorst(tuple)
			if got := FoldBand(commitsWithBands(tuple...)); got != want {
				t.Errorf("FoldBand(%v) = %q, want %q (unit band must be the worst member)", tuple, got, want)
			}
			total++
		}
	}
	// 3 + 9 + 27: a silent drop to a subset would let an untested combination
	// fold optimistically, so the count is pinned.
	if total != 39 {
		t.Fatalf("cross-product covered %d tuples, want 39 (arity 1..3 over 3 bands)", total)
	}
}

// TestFoldBandMixedCasesFoldToWorse pins the exact mixed cases the issue calls
// out, each asserted in both member orders so the worse band wins from either
// side — the fold is pessimistic, never a majority vote or a first-wins.
func TestFoldBandMixedCasesFoldToWorse(t *testing.T) {
	cases := []struct {
		name    string
		members []Band
		want    Band
	}{
		{"{CLEARED, RESIDUAL} -> RESIDUAL", []Band{BandCleared, BandResidual}, BandResidual},
		{"{CLEARED, UNVERIFIABLE} -> UNVERIFIABLE", []Band{BandCleared, BandUnverifiable}, BandUnverifiable},
		{"{UNVERIFIABLE, RESIDUAL} -> RESIDUAL", []Band{BandUnverifiable, BandResidual}, BandResidual},
		{"{CLEARED x N} -> CLEARED", []Band{BandCleared, BandCleared, BandCleared}, BandCleared},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := FoldBand(commitsWithBands(tc.members...)); got != tc.want {
				t.Errorf("FoldBand(%v) = %q, want %q", tc.members, got, tc.want)
			}
			rev := make([]Band, len(tc.members))
			for i, b := range tc.members {
				rev[len(tc.members)-1-i] = b
			}
			if got := FoldBand(commitsWithBands(rev...)); got != tc.want {
				t.Errorf("FoldBand(%v reversed) = %q, want %q (fold must be order-independent)", rev, got, tc.want)
			}
		})
	}
}

// TestFoldBandEmptyUnitIsNotVacuouslyCleared pins the trap the issue names: a
// unit with no members must NOT fold to CLEARED by vacuous truth. "All zero
// members were witnessed" is technically true and operationally a lie — nothing
// was shown to the operator, so nothing was cleared.
func TestFoldBandEmptyUnitIsNotVacuouslyCleared(t *testing.T) {
	for _, empty := range [][]Commit{nil, {}} {
		got := FoldBand(empty)
		if got == BandCleared {
			t.Fatalf("FoldBand(empty) = CLEARED — vacuous clearing of an empty unit is the exact trap this pins against")
		}
		if got != BandUnverifiable {
			t.Errorf("FoldBand(empty) = %q, want %q (an empty unit is UNVERIFIABLE, not CLEARED)", got, BandUnverifiable)
		}
	}
}

// TestFoldBandDerivesOnlyFromWitnessRungs proves the band's input set is closed
// over witness rungs: nothing but the members' verdicts can move it.
func TestFoldBandDerivesOnlyFromWitnessRungs(t *testing.T) {
	// (a) Every field OTHER than the witness rung is not an input. Two members
	// sharing a verdict but disagreeing on SHA, Subject, Leaf, Type, Resolves,
	// Mentions and Files must fold identically. Commit carries no ack, comment,
	// or operator-state field at all, so those cannot move the band by
	// construction; this pins the fields it DOES carry.
	plain := []Commit{{SHA: "a", Verdict: VerdictUnwitnessed}}
	decorated := []Commit{{
		SHA: "z", Subject: "fix(x): a busy subject (#9) (fak x)", Leaf: "x", Type: "fix",
		Resolves: []string{"#9"}, Mentions: []string{"#8"},
		Files:   []string{"a.go", "b.go"},
		Verdict: VerdictUnwitnessed,
	}}
	if a, b := FoldBand(plain), FoldBand(decorated); a != b {
		t.Errorf("band moved with non-witness fields: plain=%q decorated=%q (must be equal)", a, b)
	}

	// (b) Unit SIZE cannot move the band. N cleared members stay CLEARED for any
	// N, and a lone residual still reds a unit that is 99% cleared — pessimistic,
	// never majority-vote.
	for _, n := range []int{1, 2, 5, 50} {
		members := make([]Band, n)
		for i := range members {
			members[i] = BandCleared
		}
		if got := FoldBand(commitsWithBands(members...)); got != BandCleared {
			t.Errorf("FoldBand(%d cleared) = %q, want CLEARED (size must not move it)", n, got)
		}
	}
	mostlyCleared := make([]Band, 99)
	for i := range mostlyCleared {
		mostlyCleared[i] = BandCleared
	}
	mostlyCleared = append(mostlyCleared, BandResidual)
	if got := FoldBand(commitsWithBands(mostlyCleared...)); got != BandResidual {
		t.Errorf("FoldBand(99 cleared + 1 residual) = %q, want RESIDUAL (one unwitnessed member reds the unit)", got)
	}

	// (c) The witness rung ALONE determines the band: members expressed via the
	// raw Verdict fold to the same value as members expressed via the equivalent
	// cached Band. If any non-rung input leaked in, the two paths would diverge.
	viaVerdict := []Commit{
		{SHA: "1", Verdict: VerdictWitnessed},
		{SHA: "2", Verdict: VerdictAbstain},
		{SHA: "3", Verdict: VerdictUnwitnessed},
	}
	viaBand := commitsWithBands(BandCleared, BandUnverifiable, BandResidual)
	if a, b := FoldBand(viaVerdict), FoldBand(viaBand); a != b {
		t.Errorf("verdict path=%q band path=%q — the witness rung must fully determine the fold", a, b)
	}
}
