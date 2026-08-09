package policy

// orgcompose_test.go grades ComposeOrgFloor on the property the whole org plane
// rests on: the assembled floor is never MORE permissive than the channel above
// asked for, and the un-enrolled box does not change behavior at all.

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

func allowNames(p adjudicator.Policy) map[string]bool { return p.Allow }

func hasAllow(p adjudicator.Policy, name string) bool { return p.Allow[name] }

// TestComposeOrgFloorNoChannelsIsTheCompiledFloor is the degenerate case, and it
// is worth pinning because it is the one every box hits before it is configured:
// with nothing to compose the floor must be the shipped floor exactly, not an
// empty policy the assembly happened to build.
func TestComposeOrgFloorNoChannelsIsTheCompiledFloor(t *testing.T) {
	compiled := allowPolicy("read_docs")
	got := ComposeOrgFloor(compiled, nil, nil)
	if !hasAllow(got.Floor, "read_docs") || len(allowNames(got.Floor)) != 1 {
		t.Fatalf("floor: got %+v, want exactly the compiled allowlist", allowNames(got.Floor))
	}
	if got.CentralAuthority() {
		t.Fatal("a box with no central manifest reported central authority")
	}
	if got.Clamped() {
		t.Fatalf("nothing was proposed but something was clamped: %+v", got)
	}
}

// TestComposeOrgFloorUnenrolledOperatorWideningStands is the compatibility
// invariant of epic #5315, stated as a test so a future tightening of the central
// rule cannot reach an un-enrolled box by accident. `fak guard allow` widening the
// floor IS the feature on a box with no org authority.
func TestComposeOrgFloorUnenrolledOperatorWideningStands(t *testing.T) {
	compiled := allowPolicy("read_docs")
	operator := allowPolicy("read_docs", "deploy_prod")
	got := ComposeOrgFloor(compiled, nil, &operator)
	if !hasAllow(got.Floor, "deploy_prod") {
		t.Fatalf("an un-enrolled box refused its own allow overlay: %+v", allowNames(got.Floor))
	}
	if len(got.OperatorClamped) != 0 {
		t.Fatalf("an un-enrolled box clamped its operator overlay: %+v", got.OperatorClamped)
	}
	if got.CentralAuthority() {
		t.Fatal("a nil central manifest produced central authority")
	}
}

// TestComposeOrgFloorCentralGrantApplies is the IT-enable-more path: the org
// widens a knob fleet-wide and the box picks it up without the operator touching
// anything.
func TestComposeOrgFloorCentralGrantApplies(t *testing.T) {
	compiled := allowPolicy("read_docs")
	central := allowPolicy("read_docs", "deploy_stage")
	got := ComposeOrgFloor(compiled, &central, nil)
	if !hasAllow(got.Floor, "deploy_stage") {
		t.Fatalf("the central grant did not land: %+v", allowNames(got.Floor))
	}
	if !got.CentralAuthority() {
		t.Fatal("a verified central manifest did not register as central authority")
	}
	if len(got.Fold.CentralWiden) != 1 || got.Fold.CentralWiden[0].New != "deploy_stage" {
		t.Fatalf("central widen bucket: got %+v, want the deploy_stage grant", got.Fold.CentralWiden)
	}
	if k := foldKnob(t, got.Fold, "Allow"); k.Channel != ChannelCentral {
		t.Fatalf("Allow provenance: got %q, want %q", k.Channel, ChannelCentral)
	}
}

// TestComposeOrgFloorOperatorMayTightenUnderCentral proves centralized control is
// a FLOOR and not a straitjacket. A team that wants to be stricter than the org
// requires keeps that choice — the lattice caps loosening, never caution.
func TestComposeOrgFloorOperatorMayTightenUnderCentral(t *testing.T) {
	compiled := allowPolicy("read_docs")
	central := allowPolicy("read_docs", "deploy_stage")
	operator := allowPolicy("read_docs")
	got := ComposeOrgFloor(compiled, &central, &operator)
	if hasAllow(got.Floor, "deploy_stage") {
		t.Fatalf("the operator's tighten below the central grant was overridden: %+v", allowNames(got.Floor))
	}
	if len(got.OperatorClamped) != 0 {
		t.Fatalf("a tighten was clamped: %+v", got.OperatorClamped)
	}
}

// TestComposeOrgFloorOperatorCannotWidenPastCentral is Q2 of the R3 note carried
// all the way to a real assembled policy: the question the truth table answers in
// the abstract has to come out the same way on an actual floor, or the lattice is
// a document rather than a control.
func TestComposeOrgFloorOperatorCannotWidenPastCentral(t *testing.T) {
	compiled := allowPolicy("read_docs", "deploy_prod")
	central := allowPolicy("read_docs")
	operator := allowPolicy("read_docs", "deploy_prod")
	got := ComposeOrgFloor(compiled, &central, &operator)
	if hasAllow(got.Floor, "deploy_prod") {
		t.Fatalf("the operator climbed back past the central grant: %+v", allowNames(got.Floor))
	}
	if len(got.OperatorClamped) != 1 || got.OperatorClamped[0].Field != "Allow" {
		t.Fatalf("operator clamp: got %+v, want one Allow rollback", got.OperatorClamped)
	}
	// The clamp must ROLL BACK, not merely report. If the fold still sees the
	// operator ahead of central then the floor kept the widening and the report is
	// describing a refusal that did not happen.
	if len(got.Fold.OperatorPastCentral) != 0 {
		t.Fatalf("the assembled floor still widens past central: %+v", got.Fold.OperatorPastCentral)
	}
}

// TestComposeOrgFloorClampIsPerFieldNotWholesale is the design decision under
// test. One over-reaching field in an operator overlay must not take the rest of
// that overlay down with it, or a single bad allow silently disarms a box's local
// denies — turning a permission mistake into a protection outage.
func TestComposeOrgFloorClampIsPerFieldNotWholesale(t *testing.T) {
	compiled := adjudicator.Policy{Allow: map[string]bool{"read_docs": true, "deploy_prod": true}}
	central := adjudicator.Policy{Allow: map[string]bool{"read_docs": true}}
	operator := adjudicator.Policy{
		// reaches past central...
		Allow: map[string]bool{"read_docs": true, "deploy_prod": true},
		// ...while also tightening a different knob, which must survive.
		EgressBlockHosts: []string{"evil.example.com"},
	}
	got := ComposeOrgFloor(compiled, &central, &operator)
	if hasAllow(got.Floor, "deploy_prod") {
		t.Fatalf("the over-reaching field was not clamped: %+v", allowNames(got.Floor))
	}
	if len(got.Floor.EgressBlockHosts) != 1 || got.Floor.EgressBlockHosts[0] != "evil.example.com" {
		t.Fatalf("a legitimate tighten was discarded with the clamped field: %+v", got.Floor.EgressBlockHosts)
	}
}

// TestComposeOrgFloorCentralIsCappedByCompiledFloor pins the top of the lattice.
// Central sits BELOW the compiled-in floor, so a manifest reaching at a knob no
// central channel may move is rolled back and recorded — the org plane is
// powerful, not sovereign.
func TestComposeOrgFloorCentralIsCappedByCompiledFloor(t *testing.T) {
	// A field with no PolicyKnobRegistry classification routes to Frozen through
	// DiffAmendment's fail-closed route(). Every field-backed knob IS classified
	// today, so this exercises the cap via the registry rather than by inventing
	// an unclassified field: SafeSinks-style movement is caught by the reflection
	// backstop and lands in Frozen only if unclassified. Use the real shape
	// instead — a central manifest is capped by whatever DiffAmendment refuses.
	compiled := allowPolicy("read_docs")
	central := allowPolicy("read_docs", "deploy_stage")
	got := ComposeOrgFloor(compiled, &central, nil)
	// Nothing is FROZEN-classified among field-backed knobs today, so a
	// well-formed central manifest is refused nothing. Assert that explicitly:
	// if a future FROZEN field-backed knob lands, this is the test that has to be
	// extended alongside it rather than a silent behavior change.
	if len(got.CentralRefused) != 0 {
		t.Fatalf("a central manifest touching only classified knobs was refused: %+v", got.CentralRefused)
	}
	frozen := 0
	for _, k := range PolicyKnobRegistry {
		if k.Field != "" && k.Class == AmendFrozen {
			frozen++
		}
	}
	if frozen != 0 {
		t.Fatalf("%d field-backed FROZEN knob(s) now exist; extend this test to prove central is capped by them", frozen)
	}
}

// TestComposeOrgFloorRestoreFieldsLeavesOtherKnobsAlone guards the reflection
// clamp itself. A rollback that reset the whole struct — or that missed the named
// field — would both pass a test that only looked at the clamped knob.
func TestComposeOrgFloorRestoreFieldsLeavesOtherKnobsAlone(t *testing.T) {
	base := adjudicator.Policy{
		Allow:            map[string]bool{"read_docs": true},
		EgressBlockHosts: []string{"a.example.com"},
		Posture:          adjudicator.PostureFailClosed,
	}
	next := adjudicator.Policy{
		Allow:            map[string]bool{"read_docs": true, "deploy_prod": true},
		EgressBlockHosts: []string{"a.example.com", "b.example.com"},
		Posture:          adjudicator.PostureAdmitAndLog,
	}
	got := restoreFields(next, base, map[string]bool{"Allow": true})
	if got.Allow["deploy_prod"] {
		t.Fatalf("the named field was not restored: %+v", got.Allow)
	}
	if len(got.EgressBlockHosts) != 2 {
		t.Fatalf("an unnamed field was restored: %+v", got.EgressBlockHosts)
	}
	if got.Posture != adjudicator.PostureAdmitAndLog {
		t.Fatalf("an unnamed field was restored: posture %v", got.Posture)
	}
	// The caller's inputs must be untouched — the clamp copies, so a caller that
	// still holds `next` (to report what was asked for) is not reading a value
	// the clamp rewrote underneath it.
	if !next.Allow["deploy_prod"] {
		t.Fatal("restoreFields mutated its input")
	}
}

func foldKnob(t *testing.T, f OrgFold, field string) OrgKnobProvenance {
	t.Helper()
	return knobProvenance(t, f, field)
}
