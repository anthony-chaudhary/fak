package fleetbus

import (
	"testing"
	"time"
)

func TestFoldOutstandingIsWhyTheRosterExists(t *testing.T) {
	d, _ := NewDirective("op", "steer", "go", Selector{All: true}, time.Minute, "", testNow)
	roster := []Instance{
		testInstance(t, "serve-1", "box-a", "serve", testNow),
		testInstance(t, "serve-2", "box-b", "serve", testNow),
		testInstance(t, "serve-3", "box-c", "serve", testNow),
	}
	acks := []Ack{
		{Schema: AckSchema, Directive: d.ID, Instance: "serve-1", Status: AckApplied, Witness: "steered", Affected: 4, AckedUTC: utc(testNow)},
		{Schema: AckSchema, Directive: d.ID, Instance: "serve-2", Status: AckRefused, Reason: ApplyRefused, Detail: "STEER_NO_OWNED_LOOP", AckedUTC: utc(testNow)},
	}

	rep := Fold(d, roster, acks, testNow)
	if rep.Targeted != 3 || rep.Applied != 1 || rep.Refused != 1 || rep.Outstanding != 1 {
		t.Fatalf("fold = %+v, want targeted=3 applied=1 refused=1 outstanding=1", rep)
	}
	if rep.Complete {
		t.Fatal("Complete with an instance that never answered — that is the phantom this package exists to prevent")
	}
	if rep.AffectedTotal != 4 {
		t.Fatalf("AffectedTotal = %d, want 4", rep.AffectedTotal)
	}
	if len(rep.Rows) != 3 || rep.Rows[2].Instance != "serve-3" || rep.Rows[2].Status != RowOutstanding {
		t.Fatalf("rows = %+v, want serve-3 outstanding, id-sorted", rep.Rows)
	}
	// The local token has to survive the trip: an operator debugging a refusal
	// needs STEER_NO_OWNED_LOOP, not "refused".
	if rep.Rows[1].Detail != "STEER_NO_OWNED_LOOP" {
		t.Fatalf("row detail = %q, want the local token verbatim", rep.Rows[1].Detail)
	}

	// The last answer closes the loop.
	acks = append(acks, Ack{Schema: AckSchema, Directive: d.ID, Instance: "serve-3", Status: AckApplied, Affected: 1, AckedUTC: utc(testNow)})
	rep = Fold(d, roster, acks, testNow)
	if rep.Outstanding != 0 || !rep.Complete {
		t.Fatalf("fold = %+v, want outstanding=0 and Complete", rep)
	}
	// Complete means "everyone answered", NOT "everyone said yes".
	if rep.Refused != 1 {
		t.Fatalf("Refused = %d; Complete must not paper over a refusal", rep.Refused)
	}
}

// TestFoldKeepsALapsedTargetInTheDenominator is the regression for the way this report
// learns to lie. serve-2 is addressed at publish, dies before it acks, and its presence
// record ages out of the roster. With the denominator re-derived from the roster alone
// its row would vanish — leaving Outstanding=0, Complete=true, Applied==Targeted, and a
// control point that exits 0 announcing a fleet-wide apply serve-2 never performed.
//
// The whole point of the package is that exit 0 means witnessed. A denominator that
// shrinks when a witness dies converts silence into consent.
func TestFoldKeepsALapsedTargetInTheDenominator(t *testing.T) {
	d, _ := NewDirective("op", "pause", "", Selector{All: true}, 5*time.Minute, "", testNow)
	published := []Instance{
		testInstance(t, "serve-1", "box-a", "serve", testNow),
		testInstance(t, "serve-2", "box-b", "serve", testNow),
	}
	d = d.WithTargets(published)
	if len(d.Targets) != 2 || d.Targets[0] != "serve-1" || d.Targets[1] != "serve-2" {
		t.Fatalf("Targets = %v, want both addressed ids, sorted", d.Targets)
	}

	// serve-1 applied and is still announcing. serve-2 died: no ack, and it has aged
	// out of the roster the fold reads.
	roster := []Instance{testInstance(t, "serve-1", "box-a", "serve", testNow)}
	acks := []Ack{{Schema: AckSchema, Directive: d.ID, Instance: "serve-1", Status: AckApplied, Affected: 2, AckedUTC: utc(testNow)}}

	rep := Fold(d, roster, acks, testNow)
	if rep.Targeted != 2 || rep.Applied != 1 || rep.Outstanding != 1 {
		t.Fatalf("fold = %+v, want targeted=2 applied=1 outstanding=1 — the dead target must not leave the denominator", rep)
	}
	if rep.Complete {
		t.Fatal("Complete with an addressed instance that died silent: this is the exact false witness the target set exists to prevent")
	}
	var found bool
	for _, row := range rep.Rows {
		if row.Instance == "serve-2" {
			found = true
			if row.Status != RowOutstanding {
				t.Errorf("serve-2 status = %q, want outstanding", row.Status)
			}
			if row.InRoster {
				t.Error("serve-2 is reported as still in the roster it aged out of")
			}
		}
	}
	if !found {
		t.Fatal("the report has no row for serve-2 at all — an operator cannot even see who went missing")
	}

	// A directive from before Targets existed still folds, on the roster alone.
	old := d
	old.Targets = nil
	if rep := Fold(old, roster, acks, testNow); rep.Targeted != 1 || !rep.Complete {
		t.Fatalf("legacy fold = %+v, want the roster-only denominator (targeted=1, complete)", rep)
	}
}

// TestFoldStillAdmitsAnInstanceThatJoinedAfterThePublish — the target set is a FLOOR,
// not a ceiling. A fleet that grows mid-fan-out has a real new addressee, and pinning
// the denominator at publish time would report it as never having existed.
func TestFoldStillAdmitsAnInstanceThatJoinedAfterThePublish(t *testing.T) {
	d, _ := NewDirective("op", "pause", "", Selector{All: true}, 5*time.Minute, "", testNow)
	d = d.WithTargets([]Instance{testInstance(t, "serve-1", "box-a", "serve", testNow)})

	roster := []Instance{
		testInstance(t, "serve-1", "box-a", "serve", testNow),
		testInstance(t, "serve-2", "box-b", "serve", testNow), // announced after the publish
	}
	rep := Fold(d, roster, nil, testNow)
	if rep.Targeted != 2 || rep.Outstanding != 2 {
		t.Fatalf("fold = %+v, want targeted=2 — a late joiner the selector addresses is a real target", rep)
	}
	for _, row := range rep.Rows {
		if !row.InRoster {
			t.Errorf("row %s: InRoster=false for an instance that is in the roster", row.Instance)
		}
		if row.Machine == "" {
			t.Errorf("row %s: the roster pass did not fill in the display fields", row.Instance)
		}
	}
}

func TestFoldCountsOnlyAddressedInstances(t *testing.T) {
	d, _ := NewDirective("op", "pause", "", Selector{Role: []string{"serve"}}, time.Minute, "", testNow)
	roster := []Instance{
		testInstance(t, "serve-1", "box-a", "serve", testNow),
		testInstance(t, "worker-1", "box-a", "worker", testNow),
	}
	rep := Fold(d, roster, nil, testNow)
	if rep.Targeted != 1 || rep.Rows[0].Instance != "serve-1" {
		t.Fatalf("fold = %+v, want only the addressed role in the denominator", rep)
	}
}

func TestFoldKeepsAnAckFromADepartedInstance(t *testing.T) {
	// An ack is evidence of what happened; evidence does not stop being true when
	// the witness goes offline. Dropping it would make a churning fleet report
	// phantom outstanding work forever.
	d, _ := NewDirective("op", "steer", "go", Selector{All: true}, time.Minute, "", testNow)
	roster := []Instance{testInstance(t, "serve-1", "box-a", "serve", testNow)}
	acks := []Ack{
		{Schema: AckSchema, Directive: d.ID, Instance: "serve-1", Status: AckApplied, Affected: 1, AckedUTC: utc(testNow)},
		{Schema: AckSchema, Directive: d.ID, Instance: "serve-9", Status: AckApplied, Affected: 2, AckedUTC: utc(testNow)},
	}
	rep := Fold(d, roster, acks, testNow)
	if rep.Targeted != 2 || rep.Applied != 2 || rep.AffectedTotal != 3 {
		t.Fatalf("fold = %+v, want the departed instance's ack counted", rep)
	}
	for _, row := range rep.Rows {
		if row.Instance == "serve-9" && row.InRoster {
			t.Error("serve-9 reported as in-roster; it answered and left")
		}
	}
	if !rep.Complete {
		t.Fatal("not Complete though every targeted instance answered")
	}
}

func TestFoldIgnoresAcksForOtherDirectivesAndGarbage(t *testing.T) {
	d, _ := NewDirective("op", "steer", "go", Selector{All: true}, time.Minute, "", testNow)
	roster := []Instance{testInstance(t, "serve-1", "box-a", "serve", testNow)}
	acks := []Ack{
		{Schema: AckSchema, Directive: "d-somethingelse", Instance: "serve-1", Status: AckApplied, AckedUTC: utc(testNow)},
		{Schema: AckSchema, Directive: d.ID, Instance: "serve-1", Status: "maybe", AckedUTC: utc(testNow)},
	}
	rep := Fold(d, roster, acks, testNow)
	if rep.Outstanding != 1 || rep.Applied != 0 {
		t.Fatalf("fold = %+v, want the foreign and malformed acks ignored", rep)
	}
}

func TestFoldReportsAPermanentlyUnaccountedDirective(t *testing.T) {
	// Once the TTL lapses, an outstanding instance is not slow — it will never
	// answer. The report has to say which of the two it is.
	d, _ := NewDirective("op", "steer", "go", Selector{All: true}, 10*time.Second, "", testNow)
	roster := []Instance{testInstance(t, "serve-1", "box-a", "serve", testNow.Add(time.Minute))}

	live := Fold(d, roster, nil, testNow.Add(5*time.Second))
	if live.DirectiveExpired {
		t.Error("reported expired inside the TTL")
	}
	dead := Fold(d, roster, nil, testNow.Add(time.Minute))
	if !dead.DirectiveExpired || dead.Outstanding != 1 || dead.Complete {
		t.Fatalf("fold = %+v, want expired with one permanently unaccounted instance", dead)
	}
}

func TestFoldOfAnUnaddressedDirectiveIsNeverComplete(t *testing.T) {
	d, _ := NewDirective("op", "steer", "go", Selector{Machine: []string{"nowhere"}}, time.Minute, "", testNow)
	rep := Fold(d, []Instance{testInstance(t, "serve-1", "box-a", "serve", testNow)}, nil, testNow)
	if rep.Targeted != 0 {
		t.Fatalf("Targeted = %d, want 0", rep.Targeted)
	}
	if rep.Complete {
		t.Fatal("a directive nobody was addressed to reported Complete — vacuous success is the phantom in its purest form")
	}
}

func TestPublishTargetsResolvesTheDenominatorAtTheEdge(t *testing.T) {
	roster := []Instance{
		testInstance(t, "serve-1", "box-a", "serve", testNow),
		testInstance(t, "worker-1", "box-b", "worker", testNow),
	}
	if got := PublishTargets(Selector{All: true}, roster); len(got) != 2 {
		t.Fatalf("PublishTargets(all) = %+v, want 2", got)
	}
	if got := PublishTargets(Selector{Role: []string{"serve"}}, roster); len(got) != 1 || got[0].ID != "serve-1" {
		t.Fatalf("PublishTargets(role=serve) = %+v, want [serve-1]", got)
	}
	if got := PublishTargets(Selector{Machine: []string{"box-z"}}, roster); len(got) != 0 {
		t.Fatalf("PublishTargets(machine=box-z) = %+v, want none — this is what makes %s checkable", got, NoTarget)
	}
}
