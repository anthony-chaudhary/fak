package steerpr

import (
	"reflect"
	"testing"
)

// The honesty floor for the operator overlay: a human's OPINION must never be
// laundered into a machine WITNESS.
//
// The overlay puts two facts side by side that look similar and are not:
//
//   - diff-witnessed — a non-forgeable machine bit meaning the diff PROVES the
//     claim (internal/dispatchtick/witness.go grades it; this package only ever
//     reads it, as a supplied Verdict).
//   - acked — a human said "I looked at this and it seems fine".
//
// The second is vastly weaker. The pressure to collapse them is real and will
// grow: an acked RESIDUAL unit still counts toward osp_residual, and the
// obvious "fix" for a stubborn residual pile is to let an ack clear the band.
// That would let any human — or any agent with a shell — launder an unwitnessed
// claim into a witnessed one, defeating the premise that a self-report is not a
// fact.
//
// The counter-argument "but then the residual pile never goes down" is expected
// and correct. The pile goes down when work gets WITNESSED, not when a human
// looks at it. A chronically high pile is a true signal about the fleet's
// witness discipline, not a rendering problem to be acked away.
//
// These tests are structural on purpose. `fak steer ack` (#5028) is not built
// yet, so there is no ack API to call; proving the property by calling the
// affordance would only be possible after the affordance exists, which is
// exactly when it is too late to design the fence. Instead they pin the SHAPE
// of the band's input set, so an ack that ever reaches the band reds this file
// on the commit that wires it.

// TestAntiGamingClearedIsReachableOnlyFromTheWitnessBit is the core anti-forge
// property: of the whole Verdict space, exactly ONE value opens the CLEARED
// band, and it is the machine's witness bit. Everything else — an unrecognized
// rung, an ack-flavoured string someone hopes is special-cased, a near-miss on
// the witnessed literal — lands strictly worse.
func TestAntiGamingClearedIsReachableOnlyFromTheWitnessBit(t *testing.T) {
	forgeAttempts := []Verdict{
		// An operator's opinion, spelled every way someone might hope is honored.
		"ACK", "ACKED", "acked", "ack", "ACKNOWLEDGED",
		"OK", "ok", "APPROVED", "approved", "operator-approved",
		"LGTM", "lgtm", "reviewed", "REVIEWED", "seems-fine",
		"CLEARED", "cleared", "CLAIM_CLEARED",
		// The witness bit's own name, as prose rather than the rung.
		"diff-witnessed", "DIFF_WITNESSED", "witnessed", "WITNESSED",
		// Near-misses on the real rung: no trimming, no case-folding, no
		// prefix match may open the band.
		"CLAIM_WITNESSED ", " CLAIM_WITNESSED", "claim_witnessed",
		"CLAIM_WITNESSED_BY_OPERATOR", "CLAIM_WITNESSEDX", "CLAIM_WITNESS",
		// Structural junk.
		"", "-", "0", "1", "true", "null",
	}
	for _, v := range forgeAttempts {
		if got := BandFor(v); got == BandCleared {
			t.Errorf("BandFor(%q) = %q: only the %q witness rung may open the CLEARED band",
				v, got, VerdictWitnessed)
		}
	}
	// The one legitimate door still opens, or the band would be useless.
	if got := BandFor(VerdictWitnessed); got != BandCleared {
		t.Errorf("BandFor(%q) = %q, want %q", VerdictWitnessed, got, BandCleared)
	}
}

// TestAntiGamingBandForTakesOnlyAWitnessVerdict proves the band's input set is
// CLOSED over the witness rungs: the only thing that can reach BandFor is a
// Verdict. Ack/steer state is structurally not an input — not "is not passed
// today", but "has nowhere to be passed".
//
// If someone widens this to BandFor(v Verdict, acked bool) to let an ack clear
// a band, this reds on that commit. That is the point.
func TestAntiGamingBandForTakesOnlyAWitnessVerdict(t *testing.T) {
	fn := reflect.TypeOf(BandFor)
	if got := fn.NumIn(); got != 1 {
		t.Fatalf("BandFor takes %d inputs, want exactly 1: the band's input set must stay closed "+
			"over witness rungs — a second input is where an ack would enter", got)
	}
	if got, want := fn.In(0), reflect.TypeOf(VerdictUnknown); got != want {
		t.Errorf("BandFor input is %v, want %v: the band may read the witness rung and nothing else", got, want)
	}
	if got := fn.NumOut(); got != 1 {
		t.Fatalf("BandFor returns %d values, want exactly 1", got)
	}
	if got, want := fn.Out(0), reflect.TypeOf(BandCleared); got != want {
		t.Errorf("BandFor output is %v, want %v", got, want)
	}
}

// TestAntiGamingCommitBandInputSetIsPinned pins the field set FoldBand reads a
// commit through. FoldBand takes []Commit, so a Commit field is the other way
// an ack could reach the band — a future `Acked bool` wired into the fold would
// launder every acked commit.
//
// Carrying ack state for RENDERING is legitimate (#5028 may want to show
// "RESIDUAL (acked)"). Carrying it into the BAND is not. This test does not
// forbid the field; it forbids adding one SILENTLY. When ack lands, this reds,
// and whoever widens it must come here and prove the new field cannot improve a
// band — which is the review this fence exists to force.
func TestAntiGamingCommitBandInputSetIsPinned(t *testing.T) {
	pinned := map[string][]string{
		"Commit": {"SHA", "Subject", "Leaf", "Type", "Resolves", "Mentions", "Files", "Verdict", "Band"},
		// Partial (#5027) is membership completeness — N of M expected commits
		// landed. It is RENDER-ONLY state and provably cannot reach the band; see
		// TestAntiGamingPartialCannotImproveABand.
		"Unit": {"Leaf", "Title", "Commits", "Types", "Resolves", "Mentions", "Files", "Band", "Curve", "Partial"},
	}
	got := map[string][]string{
		"Commit": fieldNames(reflect.TypeOf(Commit{})),
		"Unit":   fieldNames(reflect.TypeOf(Unit{})),
	}
	for name, want := range pinned {
		if !equalStrings(got[name], want) {
			t.Errorf("%s fields = %v, want %v\n"+
				"A field changed shape. If you are adding operator ack/steer state: it must NOT be able to\n"+
				"improve a band. Add it to this pin, then extend\n"+
				"TestAntiGamingAckCannotLaunderAnUnwitnessedCommitToCleared to prove the fold ignores it.",
				name, got[name], want)
		}
	}
}

// TestAntiGamingAckCannotLaunderAnUnwitnessedCommitToCleared is the deliberate
// laundering attempt the acceptance gate names: it tries to clear a band via an
// ack and asserts the band is unchanged.
//
// The forge under test is the cheapest one available to anything holding a
// Commit: write the CLEARED band straight onto a commit the machine graded
// CLAIM_UNWITNESSED. A supplied Band is a CACHE of an earlier fold, not
// evidence; the Verdict is the machine bit. So when the two disagree, the
// witness must floor the result — otherwise "acked" silently becomes
// "diff-witnessed".
func TestAntiGamingAckCannotLaunderAnUnwitnessedCommitToCleared(t *testing.T) {
	forged := Commit{
		SHA:     "forged1",
		Subject: "feat(steerpr): a claim nobody proved (fak steerpr)",
		Leaf:    "steerpr",
		Type:    "feat",
		Verdict: VerdictUnwitnessed, // the machine: the diff did NOT prove it
		Band:    BandCleared,        // the ack: "I looked, it's fine"
	}

	if got := FoldBand([]Commit{forged}); got != BandResidual {
		t.Errorf("FoldBand() = %q, want %q: a CLEARED band written onto a %q commit must not outrank "+
			"the witness — that is an ack laundered into diff-witnessed",
			got, BandResidual, VerdictUnwitnessed)
	}

	// The same forge must not survive the unit fold either, which is the surface
	// an operator actually reads.
	units, _ := FoldUnits([]Commit{forged})
	if len(units) != 1 {
		t.Fatalf("FoldUnits() = %d units, want 1", len(units))
	}
	if units[0].Band != BandResidual {
		t.Errorf("unit band = %q, want %q: the forge must not survive FoldUnits either", units[0].Band, BandResidual)
	}
	// And it must still be counted in the posted number.
	if got := Residual(units); got != 1 {
		t.Errorf("Residual() = %d, want 1: an acked residual still owes attention", got)
	}

	// An ack must not rescue a unit through a sibling, either: one unwitnessed
	// member reds the unit no matter how thoroughly its neighbours were acked.
	mixed := []Commit{
		{SHA: "a", Leaf: "x", Verdict: VerdictWitnessed, Band: BandCleared},
		{SHA: "b", Leaf: "x", Verdict: VerdictUnwitnessed, Band: BandCleared},
	}
	if got := FoldBand(mixed); got != BandResidual {
		t.Errorf("FoldBand(mixed) = %q, want %q: an acked unwitnessed member still reds its unit", got, BandResidual)
	}

	// The floor is one-directional: an ack may make a unit look WORSE (an
	// operator flagging something the machine cleared is real signal, and
	// pessimism is always safe), never better.
	pessimistic := Commit{SHA: "c", Verdict: VerdictWitnessed, Band: BandResidual}
	if got := FoldBand([]Commit{pessimistic}); got != BandResidual {
		t.Errorf("FoldBand(pessimistic) = %q, want %q: a worse-than-witnessed band is allowed to stand", got, BandResidual)
	}
}

// TestAntiGamingResidualIsAckIndependent proves the posted number cannot be
// deflated by acking. osp_residual counts RESIDUAL units regardless of ack
// state, so acking a pile does not shrink it — only witnessing does.
func TestAntiGamingResidualIsAckIndependent(t *testing.T) {
	units := []Unit{
		{Leaf: "a", Band: BandResidual, Commits: []Commit{{SHA: "a1", Verdict: VerdictUnwitnessed}}},
		{Leaf: "b", Band: BandResidual, Commits: []Commit{{SHA: "b1", Verdict: VerdictUnwitnessed}}},
		{Leaf: "c", Band: BandCleared, Commits: []Commit{{SHA: "c1", Verdict: VerdictWitnessed}}},
	}
	before := Residual(units)
	if before != 2 {
		t.Fatalf("Residual() = %d, want 2", before)
	}

	// "Ack" every residual unit the only way anything can today — by writing the
	// cleared band onto its members — then re-derive the bands from the witness
	// rungs the way any honest re-tick does. The number must not move.
	for i := range units {
		for j := range units[i].Commits {
			units[i].Commits[j].Band = BandCleared
		}
	}
	for i := range units {
		units[i].Band = FoldBand(units[i].Commits)
	}
	if after := Residual(units); after != before {
		t.Errorf("Residual() = %d after acking every unit, want %d: an ack must not deflate osp_residual — "+
			"the pile falls when work is WITNESSED, not when a human looks at it", after, before)
	}
}

// TestAntiGamingAckedResidualRendersHonestly is the render-level proof: an
// acked residual unit must never render as CLEARED. Rendering is where the
// laundering would actually deceive an operator — a forged band that reads
// "CLEARED" on screen has done its damage whatever the struct says.
//
// The "(acked)" suffix itself belongs to `fak steer ack` (#5028), which is not
// built; what is pinned here is the half that must hold BEFORE it lands — the
// rendered band of an acked residual stays RESIDUAL, so the suffix can only
// ever be additive decoration on an honest band.
func TestAntiGamingAckedResidualRendersHonestly(t *testing.T) {
	acked := []Commit{{
		SHA:     "r1",
		Subject: "fix(steerpr): unproven (fak steerpr)",
		Leaf:    "steerpr",
		Type:    "fix",
		Verdict: VerdictUnwitnessed,
		Band:    BandCleared, // acked
	}}
	units, _ := FoldUnits(acked)
	if len(units) != 1 {
		t.Fatalf("FoldUnits() = %d units, want 1", len(units))
	}
	u := units[0]
	if u.Band != BandResidual {
		t.Fatalf("unit band = %q, want %q", u.Band, BandResidual)
	}
	// The title is the operator-facing string; it must not have absorbed a
	// cleared-looking band from the ack.
	if got := UnitTitle(u); got != acked[0].Subject {
		t.Errorf("UnitTitle() = %q, want the bare subject %q", got, acked[0].Subject)
	}
	// The band an operator would read renders RESIDUAL, never CLEARED.
	if string(u.Band) != string(BandResidual) {
		t.Errorf("rendered band = %q, want %q: an acked residual must render RESIDUAL, never CLEARED", u.Band, BandResidual)
	}
}

// TestAntiGamingPartialCannotImproveABand discharges the obligation the pin
// above imposes on #5027's new Unit.Partial field: proving the band fold ignores
// it.
//
// The laundering path this closes is specific and tempting. A "complete" unit
// reads as finished, and the obvious bad inference is that a finished intent
// deserves a cleaner band — collapsing "all the work arrived" into "all the work
// was proven". Those are different facts. A 12-of-12 complete unit made entirely
// of CLAIM_UNWITNESSED commits is fully assembled and fully unproven, and it must
// still render RESIDUAL.
//
// The proof is structural: FoldBand takes []Commit, and Partial lives on Unit,
// so it is not even in the fold's input type. This asserts the property end to
// end anyway, because the structural argument would quietly stop holding if a
// future refactor moved the fold to take a Unit.
func TestAntiGamingPartialCannotImproveABand(t *testing.T) {
	unwitnessed := []Commit{
		{SHA: "p1", Subject: "feat(steerpr): unproven (fak steerpr)", Leaf: "steerpr", Type: "feat", Verdict: VerdictUnwitnessed},
		{SHA: "p2", Subject: "feat(steerpr): also unproven (fak steerpr)", Leaf: "steerpr", Type: "feat", Verdict: VerdictUnwitnessed},
	}
	units, _ := FoldUnits(unwitnessed)
	if len(units) != 1 {
		t.Fatalf("FoldUnits() = %d units, want 1", len(units))
	}
	want := units[0].Band
	if want != BandResidual {
		t.Fatalf("baseline band = %q, want %q", want, BandResidual)
	}

	// Every partial state, including the most "finished"-looking one, must leave
	// the band exactly where the witness rungs put it.
	forges := []struct {
		name string
		exp  Expectation
		ok   bool
	}{
		{"complete 2 of 2", Expectation{Total: 2, Source: SourceFanout}, true},
		{"over-complete 2 of 1", Expectation{Total: 1, Source: SourceFanout}, true},
		{"complete via cohort", Expectation{Total: 2, Source: SourceCohort}, true},
		{"forming 2 of 9", Expectation{Total: 9, Source: SourceFanout}, true},
		{"unknown denominator", Expectation{}, false},
	}
	for _, f := range forges {
		t.Run(f.name, func(t *testing.T) {
			u := units[0].WithPartial(f.exp, f.ok)
			if u.Partial == nil {
				t.Fatal("WithPartial produced no partial state")
			}
			if u.Band != want {
				t.Errorf("unit band = %q after binding partial %+v, want %q: membership completeness "+
					"must not move the band — \"all the work arrived\" is not \"all the work was proven\"",
					u.Band, u.Partial, want)
			}
			// Re-derive the band the way any honest re-tick does; a Partial must not
			// leak in through the fold either.
			if got := FoldBand(u.Commits); got != want {
				t.Errorf("FoldBand() = %q with partial %+v bound, want %q", got, u.Partial, want)
			}
			// And the posted number must not deflate.
			if got := Residual([]Unit{u}); got != 1 {
				t.Errorf("Residual() = %d, want 1: a complete unit of unwitnessed commits still owes attention", got)
			}
		})
	}

	// AttachPartials is the bulk path the CLI uses; it must be equally inert.
	bulk := []Unit{units[0]}
	AttachPartials(bulk, func(Unit) (Expectation, bool) {
		return Expectation{Total: 1, Source: SourceFanout}, true // maximally "complete"
	})
	if bulk[0].Band != want {
		t.Errorf("AttachPartials moved the band to %q, want %q", bulk[0].Band, want)
	}
	if got := Residual(bulk); got != 1 {
		t.Errorf("Residual() = %d after AttachPartials, want 1", got)
	}
}

func fieldNames(t reflect.Type) []string {
	out := make([]string, 0, t.NumField())
	for i := 0; i < t.NumField(); i++ {
		out = append(out, t.Field(i).Name)
	}
	return out
}

func equalStrings(a, b []string) bool {
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
