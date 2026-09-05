package policy

// Direction tests for the per-knob amendment rules (#5414). Two things are
// pinned here and they are not the same thing:
//
//   - Every exported adjudicator.Policy field produces a NON-none delta, and the
//     delta names that field. This is the fail-open closure: a knob that no rule
//     analyzes used to fold to AmendmentNone, which both admission gates admit
//     unconditionally.
//   - A genuine tighten of each knob is admitted as a tighten and a genuine
//     widen is gated — and, critically, an UNPROVABLE direction falls back to
//     WIDEN, never tighten.

import (
	"fmt"
	"reflect"
	"regexp"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// secretProbe is a SYNTHETIC extra-secret shape used only to move the
// SecretPatterns knob. It matches nothing real and carries no credential.
var secretProbe = regexp.MustCompile(`probe-` + `[0-9]{4}`)

// amendmentDirectionCase is one knob's pair of transitions: `tighten` must
// classify AmendmentTighten with nothing gated, `widen` must classify
// AmendmentWiden. Both must attribute their change to `field`.
type amendmentDirectionCase struct {
	field       string
	tightenOld  adjudicator.Policy
	tightenNext adjudicator.Policy
	widenOld    adjudicator.Policy
	widenNext   adjudicator.Policy
}

// amendmentDirectionCases covers EVERY exported adjudicator.Policy field. The
// coverage test below reflects over the struct and fails if a field is missing
// a case, so a knob added tomorrow cannot slip in without declaring how it
// moves.
func amendmentDirectionCases() []amendmentDirectionCase {
	rungA := &adjudicator.RungProfile{}
	return []amendmentDirectionCase{
		{
			field:       "Posture",
			tightenOld:  adjudicator.Policy{Posture: adjudicator.PostureAdmitAndLog},
			tightenNext: adjudicator.Policy{Posture: adjudicator.PostureFailClosed},
			widenOld:    adjudicator.Policy{Posture: adjudicator.PostureFailClosed},
			widenNext:   adjudicator.Policy{Posture: adjudicator.PostureAdmitAndLog},
		},
		{
			field:       "Allow",
			tightenOld:  adjudicator.Policy{Allow: map[string]bool{"t": true}},
			tightenNext: adjudicator.Policy{},
			widenOld:    adjudicator.Policy{},
			widenNext:   adjudicator.Policy{Allow: map[string]bool{"t": true}},
		},
		{
			field:       "AllowPrefix",
			tightenOld:  adjudicator.Policy{AllowPrefix: []string{"read_"}},
			tightenNext: adjudicator.Policy{},
			widenOld:    adjudicator.Policy{},
			widenNext:   adjudicator.Policy{AllowPrefix: []string{"read_"}},
		},
		{
			field:       "Deny",
			tightenOld:  adjudicator.Policy{},
			tightenNext: adjudicator.Policy{Deny: map[string]abi.ReasonCode{"t": abi.ReasonPolicyBlock}},
			widenOld:    adjudicator.Policy{Deny: map[string]abi.ReasonCode{"t": abi.ReasonPolicyBlock}},
			widenNext:   adjudicator.Policy{},
		},
		{
			field:       "SelfModifyGlobs",
			tightenOld:  adjudicator.Policy{},
			tightenNext: adjudicator.Policy{SelfModifyGlobs: []string{"kernel/**"}},
			widenOld:    adjudicator.Policy{SelfModifyGlobs: []string{"kernel/**"}},
			widenNext:   adjudicator.Policy{},
		},
		{
			field:       "BlockedPathGlobs",
			tightenOld:  adjudicator.Policy{},
			tightenNext: adjudicator.Policy{BlockedPathGlobs: []string{".env"}},
			widenOld:    adjudicator.Policy{BlockedPathGlobs: []string{".env"}},
			widenNext:   adjudicator.Policy{},
		},
		{
			field:       "ArgPredicates",
			tightenOld:  adjudicator.Policy{},
			tightenNext: adjudicator.Policy{ArgPredicates: []adjudicator.ArgPredicate{{Tool: "write_file", Arg: "path", Glob: "./out/**"}}},
			widenOld:    adjudicator.Policy{ArgPredicates: []adjudicator.ArgPredicate{{Tool: "write_file", Arg: "path", Glob: "./out/**"}}},
			widenNext:   adjudicator.Policy{},
		},
		{
			field:       "SecretPatterns",
			tightenOld:  adjudicator.Policy{},
			tightenNext: adjudicator.Policy{SecretPatterns: []*regexp.Regexp{secretProbe}},
			widenOld:    adjudicator.Policy{SecretPatterns: []*regexp.Regexp{secretProbe}},
			widenNext:   adjudicator.Policy{},
		},
		{
			field:       "InlineEval",
			tightenOld:  adjudicator.Policy{},
			tightenNext: adjudicator.Policy{InlineEval: []adjudicator.InlineEvalSpec{{Interp: "perl", Flags: []string{"-e"}}}},
			widenOld:    adjudicator.Policy{InlineEval: []adjudicator.InlineEvalSpec{{Interp: "perl", Flags: []string{"-e"}}}},
			widenNext:   adjudicator.Policy{},
		},
		{
			field:       "EgressBlockHosts",
			tightenOld:  adjudicator.Policy{},
			tightenNext: adjudicator.Policy{EgressBlockHosts: []string{"ads.example.invalid"}},
			widenOld:    adjudicator.Policy{EgressBlockHosts: []string{"ads.example.invalid"}},
			widenNext:   adjudicator.Policy{},
		},

		// ---- #5414: the knobs that used to fold to AmendmentNone ----
		{
			field:       "RedactFields",
			tightenOld:  adjudicator.Policy{},
			tightenNext: adjudicator.Policy{RedactFields: []string{"authorization"}},
			widenOld:    adjudicator.Policy{RedactFields: []string{"authorization"}},
			widenNext:   adjudicator.Policy{},
		},
		{
			field:       "LintWrites",
			tightenOld:  adjudicator.Policy{},
			tightenNext: adjudicator.Policy{LintWrites: true},
			widenOld:    adjudicator.Policy{LintWrites: true},
			widenNext:   adjudicator.Policy{},
		},
		{
			field:       "Profile",
			tightenOld:  adjudicator.Policy{Profile: rungA},
			tightenNext: adjudicator.Policy{},
			widenOld:    adjudicator.Policy{},
			widenNext:   adjudicator.Policy{Profile: rungA},
		},
		{
			field:       "Complain",
			tightenOld:  adjudicator.Policy{Complain: map[string]bool{"t": true}},
			tightenNext: adjudicator.Policy{},
			widenOld:    adjudicator.Policy{},
			widenNext:   adjudicator.Policy{Complain: map[string]bool{"t": true}},
		},
		{
			field:       "AdvisoryReasons",
			tightenOld:  adjudicator.Policy{AdvisoryReasons: map[abi.ReasonCode]bool{abi.ReasonDefaultDeny: true}},
			tightenNext: adjudicator.Policy{},
			widenOld:    adjudicator.Policy{},
			widenNext:   adjudicator.Policy{AdvisoryReasons: map[abi.ReasonCode]bool{abi.ReasonDefaultDeny: true}},
		},
		{
			field:       "SecretPosture",
			tightenOld:  adjudicator.Policy{SecretPosture: adjudicator.SecretAdmitAndLog},
			tightenNext: adjudicator.Policy{SecretPosture: adjudicator.SecretFailClosed},
			widenOld:    adjudicator.Policy{SecretPosture: adjudicator.SecretFailClosed},
			widenNext:   adjudicator.Policy{SecretPosture: adjudicator.SecretAdmitAndLog},
		},
		{
			field:       "EgressExtraDenyHosts",
			tightenOld:  adjudicator.Policy{},
			tightenNext: adjudicator.Policy{EgressExtraDenyHosts: []string{"metadata.internal.invalid"}},
			widenOld:    adjudicator.Policy{EgressExtraDenyHosts: []string{"metadata.internal.invalid"}},
			widenNext:   adjudicator.Policy{},
		},
		{
			// Arming the strict allowlist TIGHTENS even though it is all
			// additions; disarming it WIDENS even though it is all removals.
			field:       "ResearchEgressAllowHosts",
			tightenOld:  adjudicator.Policy{},
			tightenNext: adjudicator.Policy{ResearchEgressAllowHosts: []string{"docs.example.invalid"}},
			widenOld:    adjudicator.Policy{ResearchEgressAllowHosts: []string{"docs.example.invalid"}},
			widenNext:   adjudicator.Policy{},
		},
		{
			field:       "EgressAllowHosts",
			tightenOld:  adjudicator.Policy{EgressAllowHosts: []string{"cdn.example.invalid"}},
			tightenNext: adjudicator.Policy{},
			widenOld:    adjudicator.Policy{},
			widenNext:   adjudicator.Policy{EgressAllowHosts: []string{"cdn.example.invalid"}},
		},
		{
			field:       "EgressBlockLists",
			tightenOld:  adjudicator.Policy{},
			tightenNext: adjudicator.Policy{EgressBlockLists: []string{"community-ads"}},
			widenOld:    adjudicator.Policy{EgressBlockLists: []string{"community-ads"}},
			widenNext:   adjudicator.Policy{},
		},
		{
			field:       "EgressRestrict",
			tightenOld:  adjudicator.Policy{},
			tightenNext: adjudicator.Policy{EgressRestrict: true},
			widenOld:    adjudicator.Policy{EgressRestrict: true},
			widenNext:   adjudicator.Policy{},
		},
		{
			field:       "AutoRepairSidestep",
			tightenOld:  adjudicator.Policy{AutoRepairSidestep: true},
			tightenNext: adjudicator.Policy{},
			widenOld:    adjudicator.Policy{},
			widenNext:   adjudicator.Policy{AutoRepairSidestep: true},
		},
		{
			field:       "TestLanes",
			tightenOld:  adjudicator.Policy{},
			tightenNext: adjudicator.Policy{TestLanes: []string{"test_a"}},
			widenOld:    adjudicator.Policy{TestLanes: []string{"test_a"}},
			widenNext:   adjudicator.Policy{},
		},
		{
			field:       "ExemptLanes",
			tightenOld:  adjudicator.Policy{ExemptLanes: []string{"exempt_a"}},
			tightenNext: adjudicator.Policy{},
			widenOld:    adjudicator.Policy{},
			widenNext:   adjudicator.Policy{ExemptLanes: []string{"exempt_a"}},
		},
		{
			field:       "DisableTestImmunity",
			tightenOld:  adjudicator.Policy{DisableTestImmunity: true},
			tightenNext: adjudicator.Policy{DisableTestImmunity: false},
			widenOld:    adjudicator.Policy{DisableTestImmunity: false},
			widenNext:   adjudicator.Policy{DisableTestImmunity: true},
		},
		{
			field:       "Lane",
			tightenOld:  adjudicator.Policy{},
			tightenNext: adjudicator.Policy{Lane: "strict"},
			widenOld:    adjudicator.Policy{Lane: "strict"},
			widenNext:   adjudicator.Policy{},
		},
	}
}

// TestDiffAmendmentCoversEveryPolicyField is the fail-open closure gate: every
// exported adjudicator.Policy field must have a direction case, and moving that
// field must produce a delta that is NOT AmendmentNone and that names the
// field. A knob added tomorrow with no rule fails here loudly, while
// residualAmendmentChanges keeps the RUNTIME answer fail-closed in the meantime.
func TestDiffAmendmentCoversEveryPolicyField(t *testing.T) {
	byField := map[string]amendmentDirectionCase{}
	for _, tc := range amendmentDirectionCases() {
		if _, dup := byField[tc.field]; dup {
			t.Fatalf("duplicate direction case for %s", tc.field)
		}
		byField[tc.field] = tc
	}

	pt := reflect.TypeOf(adjudicator.Policy{})
	for i := 0; i < pt.NumField(); i++ {
		f := pt.Field(i)
		if !f.IsExported() {
			continue
		}
		tc, ok := byField[f.Name]
		if !ok {
			t.Errorf("adjudicator.Policy field %s has no amendment-direction case; "+
				"add a rule in amendment_direction.go and a case here (the residual sweep "+
				"gates it as WIDEN in the meantime, so the floor is safe but blunt)", f.Name)
			continue
		}
		for _, m := range []struct {
			what      string
			old, next adjudicator.Policy
		}{
			{what: "tighten", old: tc.tightenOld, next: tc.tightenNext},
			{what: "widen", old: tc.widenOld, next: tc.widenNext},
		} {
			d := DiffAmendment(m.old, m.next)
			if d.Class() == AmendmentNone {
				t.Errorf("%s %s transition folds to AmendmentNone — both admission gates "+
					"admit that unconditionally, which is the fail-open hole #5414 closes", f.Name, m.what)
				continue
			}
			if !deltaNamesField(d, f.Name) {
				t.Errorf("%s %s transition produced %+v, which never names field %s", f.Name, m.what, d, f.Name)
			}
		}
	}

	// The reverse direction: every case must name a field that still exists.
	for name := range byField {
		if f, ok := pt.FieldByName(name); !ok || !f.IsExported() {
			t.Errorf("direction case %q names an adjudicator.Policy field that does not exist", name)
		}
	}
	// And analyzedAmendmentFields must not drift from the struct either.
	for name := range analyzedAmendmentFields {
		if f, ok := pt.FieldByName(name); !ok || !f.IsExported() {
			t.Errorf("analyzedAmendmentFields names %q, which is not an exported adjudicator.Policy field", name)
		}
	}
}

// TestAmendmentDirectionPerKnob is the sharpening #5414 asks for: a genuine
// TIGHTEN of each knob is admitted as a tighten (no gated toll) and a genuine
// WIDEN is gated. Before this, the twelve knobs the original DiffAmendment did
// not analyze had NO direction at all.
func TestAmendmentDirectionPerKnob(t *testing.T) {
	for _, tc := range amendmentDirectionCases() {
		t.Run(tc.field, func(t *testing.T) {
			tightened := DiffAmendment(tc.tightenOld, tc.tightenNext)
			if got := tightened.Class(); got != AmendmentTighten {
				t.Errorf("tighten of %s classified %q, want %q (delta=%+v)", tc.field, got, AmendmentTighten, tightened)
			}
			if len(tightened.Widen) != 0 || len(tightened.Frozen) != 0 {
				t.Errorf("tighten of %s leaked into a gated bucket: widen=%v frozen=%v", tc.field, tightened.Widen, tightened.Frozen)
			}
			d := DiffAmendment(tc.widenOld, tc.widenNext)
			if got := d.Class(); got != AmendmentWiden {
				t.Errorf("widen of %s classified %q, want %q (delta=%+v)", tc.field, got, AmendmentWiden, d)
			}
			if len(d.Tighten) != 0 {
				t.Errorf("widen of %s put changes in the tighten bucket: %v", tc.field, d.Tighten)
			}
		})
	}
}

// TestUnknownAmendmentDirectionFallsBackToWiden is the security pin. The two
// misclassifications are NOT symmetric: a widen labelled tighten silently
// loosens a floor that was supposed to be gated, while a tighten labelled widen
// only costs a confirm. So every case whose direction this package cannot prove
// must land in Widen. This test fails if any of them ever lands in Tighten.
func TestUnknownAmendmentDirectionFallsBackToWiden(t *testing.T) {
	t.Run("residual sweep of an unanalyzed field", func(t *testing.T) {
		// Simulate a Policy field that has no directional rule by handing the
		// sweep an `analyzed` set that omits one. The change is a genuine
		// TIGHTEN (an added deny host), and the sweep must STILL gate it,
		// because a sweep by construction knows nothing about the knob it caught.
		var d AmendmentDelta
		residualAmendmentChanges(&d,
			adjudicator.Policy{},
			adjudicator.Policy{EgressExtraDenyHosts: []string{"metadata.internal.invalid"}},
			map[string]bool{}, // nothing is analyzed
		)
		if len(d.Tighten) != 0 {
			t.Fatalf("residual sweep classified an unknown-direction change as TIGHTEN: %v", d.Tighten)
		}
		if got := d.Class(); got != AmendmentWiden {
			t.Fatalf("residual sweep Class() = %q, want %q (delta=%+v)", got, AmendmentWiden, d)
		}
		if !deltaNamesField(d, "EgressExtraDenyHosts") {
			t.Fatalf("residual sweep did not name the field it swept: %+v", d)
		}
	})

	t.Run("residual sweep emits nothing while every field has a rule", func(t *testing.T) {
		// The backstop must stay silent in production today, otherwise every
		// hand-written rule would be double-counted.
		var d AmendmentDelta
		residualAmendmentChanges(&d,
			adjudicator.Policy{},
			adjudicator.Policy{
				Allow:                map[string]bool{"t": true},
				EgressExtraDenyHosts: []string{"metadata.internal.invalid"},
				LintWrites:           true,
			},
			analyzedAmendmentFields,
		)
		if !d.Empty() {
			t.Fatalf("residual sweep double-counted analyzed fields: %+v", d)
		}
	})

	t.Run("edited RungProfile", func(t *testing.T) {
		// Two DIFFERENT non-nil profiles. The zero profile elides nothing;
		// DefaultRungProfile elides the read-class convenience rungs.
		elidesNothing := adjudicator.Policy{Profile: &adjudicator.RungProfile{}}
		elidesSome := adjudicator.Policy{Profile: adjudicator.DefaultRungProfile()}
		if reflect.DeepEqual(*elidesNothing.Profile, *elidesSome.Profile) {
			t.Fatalf("probe profiles are equal; DefaultRungProfile no longer elides anything")
		}

		// Equal profiles stay a no-op — the rule must not manufacture a delta.
		if d := DiffAmendment(elidesNothing, adjudicator.Policy{Profile: &adjudicator.RungProfile{}}); !d.Empty() {
			t.Fatalf("two equal profiles must be a no-op, got %+v", d)
		}

		// THE PIN, asserted FIRST so a flipped fallback is what the failure
		// names. Eliding FEWER rungs genuinely tightens, but the elision
		// bitmask is an unexported RungProfile field, so this package cannot
		// PROVE that. It must still gate. If this ever reports tighten, the
		// fallback has been inverted into the dangerous direction.
		d := DiffAmendment(elidesSome, elidesNothing)
		if len(d.Tighten) != 0 {
			t.Fatalf("an unprovable Profile edit was classified TIGHTEN: %v", d.Tighten)
		}
		if got := d.Class(); got != AmendmentWiden {
			t.Fatalf("unprovable Profile edit classified %q, want %q (delta=%+v)", got, AmendmentWiden, d)
		}

		// The opposite edit — eliding MORE rungs — genuinely loosens, and the
		// same rule already gates it. Asserted second so the pin above owns the
		// failure message when the fallback is inverted.
		if got := DiffAmendment(elidesNothing, elidesSome).Class(); got != AmendmentWiden {
			t.Fatalf("eliding more rungs classified %q, want %q", got, AmendmentWiden)
		}
	})

	t.Run("unrecognized SecretPosture value", func(t *testing.T) {
		// A value outside the recognized set is not rankable. Moving AWAY from
		// it toward the strictest recognized verdict LOOKS like a tighten; it is
		// not provable, so it must gate.
		// Scan for an unrecognized value rather than hard-coding one, so this
		// pin can never quietly turn into a skip if the recognized set grows.
		unknown, found := adjudicator.SecretPosture(0), false
		for v := 255; v >= 0; v-- {
			if _, ok := secretStrictness(adjudicator.SecretPosture(v)); !ok {
				unknown, found = adjudicator.SecretPosture(v), true
				break
			}
		}
		if !found {
			t.Fatalf("every uint8 is a recognized SecretPosture; the unrankable branch is unreachable and its fallback is unpinned")
		}
		toStrict := DiffAmendment(
			adjudicator.Policy{SecretPosture: unknown},
			adjudicator.Policy{SecretPosture: adjudicator.SecretFailClosed},
		)
		if len(toStrict.Tighten) != 0 || toStrict.Class() != AmendmentWiden {
			t.Fatalf("unrecognized->fail_closed must gate, got %+v class=%q", toStrict, toStrict.Class())
		}
		fromKnown := DiffAmendment(
			adjudicator.Policy{SecretPosture: adjudicator.SecretFailClosed},
			adjudicator.Policy{SecretPosture: unknown},
		)
		if len(fromKnown.Tighten) != 0 || fromKnown.Class() != AmendmentWiden {
			t.Fatalf("fail_closed->unrecognized must gate, got %+v class=%q", fromKnown, fromKnown.Class())
		}
		// The audit row must not launder the unknown value into "quarantine".
		want := fmt.Sprintf("secret_verdict=fail_closed->unrecognized(%d)", uint8(unknown))
		if got := FormatAmendmentChanges(fromKnown.Widen); got != want {
			t.Fatalf("unrecognized verdict audit text = %q, want %q", got, want)
		}
	})
}

// TestResearchAllowlistArmingIsATightenAndDisarmingIsAWiden pins the one knob
// whose direction inverts with emptiness — the case a naive "added element
// widens an allowlist" rule would get exactly backwards in the dangerous
// direction.
func TestResearchAllowlistArmingIsATightenAndDisarmingIsAWiden(t *testing.T) {
	live := adjudicator.Policy{ResearchEgressAllowHosts: []string{"docs.example.invalid"}}

	arm := DiffAmendment(adjudicator.Policy{}, live)
	if got := arm.Class(); got != AmendmentTighten {
		t.Errorf("arming the strict allowlist classified %q, want %q (delta=%+v)", got, AmendmentTighten, arm)
	}
	if got := FormatAmendmentChanges(arm.Tighten); got != "research_allowlist_on=docs.example.invalid" {
		t.Errorf("arming audit text = %q", got)
	}

	disarm := DiffAmendment(live, adjudicator.Policy{})
	if got := disarm.Class(); got != AmendmentWiden {
		t.Errorf("disarming the strict allowlist classified %q, want %q (delta=%+v)", got, AmendmentWiden, disarm)
	}

	// While the allowlist is live on BOTH sides it behaves like an ordinary
	// allowlist: one more reachable host is a widen.
	wider := DiffAmendment(live, adjudicator.Policy{
		ResearchEgressAllowHosts: []string{"docs.example.invalid", "blog.example.invalid"},
	})
	if got := wider.Class(); got != AmendmentWiden {
		t.Errorf("adding a host to a live allowlist classified %q, want %q (delta=%+v)", got, AmendmentWiden, wider)
	}
	narrower := DiffAmendment(adjudicator.Policy{
		ResearchEgressAllowHosts: []string{"docs.example.invalid", "blog.example.invalid"},
	}, live)
	if got := narrower.Class(); got != AmendmentTighten {
		t.Errorf("removing a host from a live allowlist classified %q, want %q (delta=%+v)", got, AmendmentTighten, narrower)
	}
}

// TestAmendmentDirectionFormatsStableLabels pins the audit/refusal text of the
// new knobs, so a journal row keeps binding a change back to its knob.
func TestAmendmentDirectionFormatsStableLabels(t *testing.T) {
	old := adjudicator.Policy{
		RedactFields:         []string{"authorization"},
		EgressExtraDenyHosts: []string{"metadata.internal.invalid"},
		EgressBlockLists:     []string{"community-ads"},
		LintWrites:           true,
		EgressRestrict:       true,
		SecretPosture:        adjudicator.SecretFailClosed,
	}
	next := adjudicator.Policy{
		Complain:           map[string]bool{"run_shell": true},
		AdvisoryReasons:    map[abi.ReasonCode]bool{abi.ReasonDefaultDeny: true},
		EgressAllowHosts:   []string{"cdn.example.invalid"},
		AutoRepairSidestep: true,
		SecretPosture:      adjudicator.SecretAdmitAndLog,
	}
	d := DiffAmendment(old, next)
	if got := d.Class(); got != AmendmentWiden {
		t.Fatalf("Class() = %q, want %q (delta=%+v)", got, AmendmentWiden, d)
	}
	if len(d.Tighten) != 0 || len(d.Frozen) != 0 {
		t.Fatalf("pure widen leaked buckets: tighten=%v frozen=%v", d.Tighten, d.Frozen)
	}
	want := "removed_redact_fields=authorization; " +
		"removed_extra_deny_hosts=metadata.internal.invalid; " +
		"removed_block_lists=community-ads; " +
		"added_egress_allow_hosts=cdn.example.invalid; " +
		"lint_writes=on->off; " +
		"egress_restrict=on->off; " +
		"auto_repair_sidestep=off->on; " +
		"added_complain=run_shell; " +
		"added_advisory_reasons=DEFAULT_DENY; " +
		"secret_verdict=fail_closed->admit_and_log"
	if got := FormatAmendmentChanges(d.Widen); got != want {
		t.Fatalf("widen audit text =\n  %q\nwant\n  %q", got, want)
	}
}

// TestSnakeFieldName pins the sweep's label key so a swept field's audit row
// reads like the hand-written ones.
func TestSnakeFieldName(t *testing.T) {
	for in, want := range map[string]string{
		"EgressBlockHosts": "egress_block_hosts",
		"Posture":          "posture",
		"LintWrites":       "lint_writes",
	} {
		if got := snakeFieldName(in); got != want {
			t.Errorf("snakeFieldName(%q) = %q, want %q", in, got, want)
		}
	}
}

// deltaNamesField reports whether any bucket carries a change attributed to
// field — the binding a journal row needs.
func deltaNamesField(d AmendmentDelta, field string) bool {
	for _, bucket := range [][]AmendmentChange{d.Tighten, d.Widen, d.Frozen} {
		for _, c := range bucket {
			if c.Field == field {
				return true
			}
		}
	}
	return false
}
