package policy

// orgprecedence_fold_test.go covers the parts of the #5322 fold that
// org_precedence_test.go deliberately does not reach.
//
// That file is the research note's truth table, transcribed: it asserts the
// resolved VALUE for every (compiled-in, central, operator) row and nothing else.
// This file asserts the two things the fold has to get right for `fak org status`
// to be worth reading:
//
//  1. PROVENANCE — not just what the value is, but which channel it came from,
//     and what a channel asked for that the fold quietly absorbed. A resolution
//     that lands on the right number while blaming the wrong channel sends an
//     operator to argue with the wrong person.
//  2. ATTRIBUTION over real assembled floors — that FoldOrgProvenance reads three
//     adjudicator.Policy snapshots and says who moved each knob, including the two
//     refusal cases (central reaching at a FROZEN knob, the operator widening past
//     a central grant) that the assembly site has to be able to act on.

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// noteFor returns the note recorded against channel, if any.
func noteFor(res OrgPrecedenceResolution, channel string) (OrgPrecedenceNote, bool) {
	for _, n := range res.Notes {
		if n.Channel == channel {
			return n, true
		}
	}
	return OrgPrecedenceNote{}, false
}

// TestOrgPrecedenceProvenanceNamesTheOwningChannel pins the Channel field for
// each class. The truth table proves the VALUE; this proves an operator reading
// `fak org status` is pointed at the channel that can actually change it.
func TestOrgPrecedenceProvenanceNamesTheOwningChannel(t *testing.T) {
	cases := []struct {
		name string
		in   OrgPrecedenceInput
		want string
	}{
		{
			// Nobody restricts: the floor owns it, and "compiled-in" is the honest
			// answer rather than an empty string a renderer would have to guess at.
			"ratchet-unrestricted-belongs-to-the-floor",
			OrgPrecedenceInput{Class: AmendRatchet},
			ChannelCompiledIn,
		},
		{
			// Both tighten, so the value is the same either way — but the HIGHEST
			// authority denier is the one an operator cannot talk their way past.
			"ratchet-names-the-highest-authority-denier",
			OrgPrecedenceInput{
				Class:    AmendRatchet,
				Central:  OrgKnobContribution{Deny: true},
				Operator: OrgKnobContribution{Deny: true},
			},
			ChannelCentral,
		},
		{
			// Only the operator denies: naming central here would send them to IT
			// over a lock they put on their own box.
			"ratchet-operator-only-deny-belongs-to-the-operator",
			OrgPrecedenceInput{Class: AmendRatchet, Operator: OrgKnobContribution{Deny: true}},
			ChannelOperatorOverlay,
		},
		{
			// Central restates the compiled ceiling exactly. Restating a value is
			// not setting it: ownership stays with the channel that established it.
			"gated-widen-restating-a-ceiling-does-not-take-ownership",
			OrgPrecedenceInput{
				Class:      AmendGatedWiden,
				CompiledIn: OrgKnobContribution{Set: true, Cap: 200},
				Central:    OrgKnobContribution{Set: true, Cap: 200},
			},
			ChannelCompiledIn,
		},
		{
			// Central overreaches; it is clamped to the compiled cap and therefore
			// did NOT lower the ceiling — so the floor still owns the running value.
			"gated-widen-clamped-central-does-not-own-the-value",
			OrgPrecedenceInput{
				Class:      AmendGatedWiden,
				CompiledIn: OrgKnobContribution{Set: true, Cap: 200},
				Central:    OrgKnobContribution{Set: true, Cap: 300},
			},
			ChannelCompiledIn,
		},
		{
			// A FROZEN knob is compiled-in no matter who else spoke.
			"frozen-always-belongs-to-the-floor",
			OrgPrecedenceInput{
				Class:    AmendFrozen,
				Central:  OrgKnobContribution{WidenAttempt: true},
				Operator: OrgKnobContribution{Deny: true},
			},
			ChannelCompiledIn,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ResolveOrgPrecedence(tc.in).Channel; got != tc.want {
				t.Fatalf("resolved channel: got %q, want %q", got, tc.want)
			}
		})
	}
}

// TestOrgPrecedenceClampNoteCarriesTheNumbers is the "your grant did not land
// whole" signal. An org admin who writes 300 against a compiled cap of 200 gets
// 200; without the asked/got pair the report can only say "clamped", which does
// not tell them what to change.
func TestOrgPrecedenceClampNoteCarriesTheNumbers(t *testing.T) {
	res := ResolveOrgPrecedence(OrgPrecedenceInput{
		Class:      AmendGatedWiden,
		CompiledIn: OrgKnobContribution{Set: true, Cap: 200},
		Central:    OrgKnobContribution{Set: true, Cap: 300},
	})
	note, ok := noteFor(res, ChannelCentral)
	if !ok {
		t.Fatalf("a clamped central grant recorded no note: %+v", res)
	}
	if note.Effect != OrgNoteClamped || note.Asked != 300 || note.Got != 200 {
		t.Fatalf("clamp note: got %+v, want effect=%s asked=300 got=200", note, OrgNoteClamped)
	}
}

// TestOrgPrecedenceOperatorClampIsAttributedToTheOperator guards the direction of
// the clamp note. The operator asked to climb back past a central grant; the note
// must land on the operator, not on the central channel that capped them.
func TestOrgPrecedenceOperatorClampIsAttributedToTheOperator(t *testing.T) {
	res := ResolveOrgPrecedence(OrgPrecedenceInput{
		Class:      AmendGatedWiden,
		CompiledIn: OrgKnobContribution{Set: true, Cap: 200},
		Central:    OrgKnobContribution{Set: true, Cap: 100},
		Operator:   OrgKnobContribution{Set: true, Cap: 150},
	})
	if _, ok := noteFor(res, ChannelCentral); ok {
		t.Fatalf("central granted 100 under a cap of 200 and was not clamped, but carries a note: %+v", res.Notes)
	}
	note, ok := noteFor(res, ChannelOperatorOverlay)
	if !ok || note.Effect != OrgNoteClamped || note.Asked != 150 || note.Got != 100 {
		t.Fatalf("operator clamp note: got %+v (ok=%v), want effect=%s asked=150 got=100", note, ok, OrgNoteClamped)
	}
}

// TestOrgPrecedenceRatchetWidenAttemptIsRecorded proves a refused loosening is
// reported rather than dropped. The resolved value is unchanged either way, which
// is exactly why silence here is dangerous: a central plane could believe for
// months that it had relaxed a ratchet.
func TestOrgPrecedenceRatchetWidenAttemptIsRecorded(t *testing.T) {
	res := ResolveOrgPrecedence(OrgPrecedenceInput{
		Class:   AmendRatchet,
		Central: OrgKnobContribution{WidenAttempt: true},
	})
	if res.Verdict != OrgVerdictAllow {
		t.Fatalf("a widen attempt on an unrestricted ratchet must not change the verdict: %+v", res)
	}
	note, ok := noteFor(res, ChannelCentral)
	if !ok || note.Effect != OrgNoteRefusedWiden {
		t.Fatalf("refused-widen note: got %+v (ok=%v), want effect=%s", note, ok, OrgNoteRefusedWiden)
	}
}

// TestOrgPrecedenceFrozenRecordsNoAuthority is the auditor's row. A central
// manifest touching the FROZEN floor changes nothing — and is precisely the event
// worth surfacing, because it is either a misconfigured org or an attempt on the
// one surface that is supposed to be unreachable.
func TestOrgPrecedenceFrozenRecordsNoAuthority(t *testing.T) {
	res := ResolveOrgPrecedence(OrgPrecedenceInput{
		Class:      AmendFrozen,
		CompiledIn: OrgKnobContribution{Deny: true},
		Central:    OrgKnobContribution{WidenAttempt: true},
		Operator:   OrgKnobContribution{WidenAttempt: true},
	})
	if res.Verdict != OrgVerdictDeny {
		t.Fatalf("frozen knob moved: %+v", res)
	}
	for _, ch := range []string{ChannelCentral, ChannelOperatorOverlay} {
		note, ok := noteFor(res, ch)
		if !ok || note.Effect != OrgNoteNoAuthority {
			t.Fatalf("%s note: got %+v (ok=%v), want effect=%s", ch, note, ok, OrgNoteNoAuthority)
		}
	}
	// The compiled-in channel is the one channel that MAY move a frozen knob, so
	// it must never appear in the notes — a report that accused the floor of
	// overreaching against itself would be nonsense.
	if _, ok := noteFor(res, ChannelCompiledIn); ok {
		t.Fatalf("compiled-in must not be noted as overreaching on its own knob: %+v", res.Notes)
	}
}

// TestOrgPrecedenceUnsetGatedWidenIsTightest covers the case the truth table has
// no row for: nobody sets a ceiling at all. Set=false means INHERIT everywhere
// else in the fold, so with nothing to inherit the knob must land on its zero
// value (its tightest posture) and not on "unbounded".
func TestOrgPrecedenceUnsetGatedWidenIsTightest(t *testing.T) {
	res := ResolveOrgPrecedence(OrgPrecedenceInput{Class: AmendGatedWiden})
	if res.Cap != 0 || res.Channel != ChannelCompiledIn {
		t.Fatalf("an unset gated-widen knob: got cap=%d channel=%q, want cap=0 channel=%q",
			res.Cap, res.Channel, ChannelCompiledIn)
	}
}

// TestOrgPrecedenceUnknownClassFallsBackToFrozen is the fail-closed backstop. A
// knob whose class this fold does not understand must resolve to the compiled-in
// floor, never to whichever channel happened to speak last.
func TestOrgPrecedenceUnknownClassFallsBackToFrozen(t *testing.T) {
	for _, class := range []AmendmentClass{AmendSelfAmendable, AmendmentClass("SOMETHING_NEW")} {
		res := ResolveOrgPrecedence(OrgPrecedenceInput{
			Class:      class,
			CompiledIn: OrgKnobContribution{Deny: true},
			Central:    OrgKnobContribution{WidenAttempt: true},
		})
		if res.Verdict != OrgVerdictDeny || res.Channel != ChannelCompiledIn {
			t.Fatalf("class %q: got verdict=%q channel=%q, want DENY from %s",
				class, res.Verdict, res.Channel, ChannelCompiledIn)
		}
	}
}

// ---------------------------------------------------------------------------
// FoldOrgProvenance — attribution over three real assembled floors
// ---------------------------------------------------------------------------

// allowPolicy builds a floor admitting exactly the named tools. Allow is a
// GATED_WIDEN knob, so adding a name is the simplest real widening there is.
func allowPolicy(names ...string) adjudicator.Policy {
	allow := map[string]bool{}
	for _, n := range names {
		allow[n] = true
	}
	return adjudicator.Policy{Allow: allow}
}

// knobProvenance finds one field's attributed provenance.
func knobProvenance(t *testing.T, f OrgFold, field string) OrgKnobProvenance {
	t.Helper()
	for _, k := range f.Knobs {
		if k.Field == field {
			return k
		}
	}
	t.Fatalf("field %q is absent from the fold's knob list (%d knobs)", field, len(f.Knobs))
	return OrgKnobProvenance{}
}

// TestFoldOrgProvenanceAttributesEachStage walks one knob through all three
// channels and checks the fold names the right one each time. Last-writer down
// the authority order is the honest reading of a snapshot chain: the value
// running is the one the last stage left.
func TestFoldOrgProvenanceAttributesEachStage(t *testing.T) {
	base := allowPolicy("read_docs")
	cases := []struct {
		name    string
		stages  OrgStages
		want    string
		widened bool
	}{
		{
			"untouched-knob-stands-at-the-floor",
			OrgStages{CompiledIn: base, Central: base, Operator: base, CentralApplied: true},
			ChannelCompiledIn, false,
		},
		{
			"central-widening-belongs-to-central",
			OrgStages{
				CompiledIn: base, Central: allowPolicy("read_docs", "deploy_stage"),
				Operator: allowPolicy("read_docs", "deploy_stage"), CentralApplied: true,
			},
			ChannelCentral, true,
		},
		{
			// The operator tightened BELOW the central grant, which the lattice
			// permits. Attribution follows the last stage that moved it.
			"operator-tightening-below-central-belongs-to-the-operator",
			OrgStages{
				CompiledIn: base, Central: allowPolicy("read_docs", "deploy_stage"),
				Operator: allowPolicy("read_docs"), CentralApplied: true,
			},
			ChannelOperatorOverlay, false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := knobProvenance(t, FoldOrgProvenance(tc.stages), "Allow")
			if got.Channel != tc.want {
				t.Fatalf("Allow channel: got %q, want %q", got.Channel, tc.want)
			}
			if got.Widened != tc.widened {
				t.Fatalf("Allow widened: got %v, want %v", got.Widened, tc.widened)
			}
			if got.Class != AmendGatedWiden {
				t.Fatalf("Allow class: got %q, want %q", got.Class, AmendGatedWiden)
			}
		})
	}
}

// TestFoldOrgProvenanceBucketsCentralWidening is the journal feed: #5322 requires
// every central widening recorded with issuer and envelope version, and this is
// the bucket the caller iterates to write those rows.
func TestFoldOrgProvenanceBucketsCentralWidening(t *testing.T) {
	f := FoldOrgProvenance(OrgStages{
		CompiledIn:     allowPolicy("read_docs"),
		Central:        allowPolicy("read_docs", "deploy_stage"),
		Operator:       allowPolicy("read_docs", "deploy_stage"),
		CentralApplied: true,
	})
	if len(f.CentralWiden) != 1 || f.CentralWiden[0].New != "deploy_stage" {
		t.Fatalf("central widen bucket: got %+v, want one added_allow=deploy_stage", f.CentralWiden)
	}
	if !f.CentralMoved() {
		t.Fatal("CentralMoved must be true when central widened a knob")
	}
	if len(f.OperatorPastCentral) != 0 {
		t.Fatalf("the operator changed nothing but is reported as widening past central: %+v", f.OperatorPastCentral)
	}
}

// TestFoldOrgProvenanceFlagsOperatorWideningPastCentral is the enforcement signal
// the assembly site consumes: under a central grant, a local operator may tighten
// further but may not climb back past it.
func TestFoldOrgProvenanceFlagsOperatorWideningPastCentral(t *testing.T) {
	stages := OrgStages{
		CompiledIn:     allowPolicy("read_docs"),
		Central:        allowPolicy("read_docs"),
		Operator:       allowPolicy("read_docs", "deploy_prod"),
		CentralApplied: true,
	}
	f := FoldOrgProvenance(stages)
	if len(f.OperatorPastCentral) != 1 || f.OperatorPastCentral[0].New != "deploy_prod" {
		t.Fatalf("operator-past-central: got %+v, want one added_allow=deploy_prod", f.OperatorPastCentral)
	}
}

// TestFoldOrgProvenanceUnenrolledOperatorWideningIsOrdinary is the epic's
// load-bearing compatibility invariant: an un-enrolled box behaves byte-for-byte
// as it does today. The same operator widening that is a violation under a
// central grant is ordinary local configuration without one — so the fold must
// not report it, or every un-enrolled box would light up with a refusal it has no
// way to resolve.
func TestFoldOrgProvenanceUnenrolledOperatorWideningIsOrdinary(t *testing.T) {
	base := allowPolicy("read_docs")
	f := FoldOrgProvenance(OrgStages{
		CompiledIn:     base,
		Central:        base,
		Operator:       allowPolicy("read_docs", "deploy_prod"),
		CentralApplied: false,
	})
	if len(f.OperatorPastCentral) != 0 {
		t.Fatalf("an un-enrolled box reported an operator widening as past-central: %+v", f.OperatorPastCentral)
	}
	if f.CentralMoved() {
		t.Fatalf("an un-enrolled box reported central movement: %+v", f)
	}
	if got := knobProvenance(t, f, "Allow"); got.Channel != ChannelOperatorOverlay {
		t.Fatalf("Allow channel on an un-enrolled box: got %q, want %q", got.Channel, ChannelOperatorOverlay)
	}
}

// TestFoldOrgProvenanceCentralRatchetLooseningIsAWiden proves the fold reads
// DIRECTION from the knob's class rather than from the shape of the edit. On a
// RATCHET knob, ADDING is the safe direction and REMOVING is the loosening — the
// opposite of the GATED_WIDEN allowlist above. A central manifest dropping a
// self-modify glob is the org plane removing protection, and it has to land in
// the widen bucket (the one #5322 journals) and not read as an ordinary tighten
// because the field got shorter.
//
// This is the closest reachable case to a central overreach: no field-backed knob
// is FROZEN today, so CentralRefused is fed only by DiffAmendment's fail-closed
// route for an unclassified field — unreachable from a test without adding one.
func TestFoldOrgProvenanceCentralRatchetLooseningIsAWiden(t *testing.T) {
	compiled := adjudicator.Policy{SelfModifyGlobs: []string{".fak/guard/*.json"}}
	central := adjudicator.Policy{}
	f := FoldOrgProvenance(OrgStages{
		CompiledIn: compiled, Central: central, Operator: central, CentralApplied: true,
	})
	if len(f.CentralWiden) != 1 || f.CentralWiden[0].Field != "SelfModifyGlobs" {
		t.Fatalf("central widen bucket: got %+v, want the removed self-modify glob", f.CentralWiden)
	}
	if len(f.CentralTighten) != 0 {
		t.Fatalf("dropping a self-modify glob was read as a tighten: %+v", f.CentralTighten)
	}
	if !f.CentralMoved() {
		t.Fatal("CentralMoved must be true when central dropped a self-modify glob")
	}
	if got := knobProvenance(t, f, "SelfModifyGlobs"); got.Channel != ChannelCentral || !got.Widened {
		t.Fatalf("SelfModifyGlobs provenance: got channel=%q widened=%v, want %q/true",
			got.Channel, got.Widened, ChannelCentral)
	}
}

// TestFoldOrgProvenanceListsEveryFieldBackedKnob guards the completeness the
// report depends on. Listing only the MOVED knobs would make "this knob is at the
// shipped floor" and "this knob is not in the registry" render identically, and
// the second is a bug an operator has no way to see.
func TestFoldOrgProvenanceListsEveryFieldBackedKnob(t *testing.T) {
	f := FoldOrgProvenance(OrgStages{})
	want := 0
	for _, k := range PolicyKnobRegistry {
		if k.Field != "" {
			want++
		}
	}
	if len(f.Knobs) != want {
		t.Fatalf("fold listed %d knobs, want every field-backed registry entry (%d)", len(f.Knobs), want)
	}
	for _, k := range f.Knobs {
		if k.Channel != ChannelCompiledIn {
			t.Fatalf("an empty three-stage fold attributed %s to %q, want %q", k.Field, k.Channel, ChannelCompiledIn)
		}
	}
}
