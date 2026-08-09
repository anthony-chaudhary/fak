package fleetbus

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// unsupported_test.go — the witness for #5953's capability DECLARATION half.
//
// The gap it closes is not "the roster is missing a field". It is that the roster
// could not distinguish an instance that CANNOT do something from one that simply
// never said — so an operator with sixteen live guards had no way to learn that none
// of them could steer except by steering at all sixteen and reading sixteen
// refusals. Every arm below is about keeping the three answers separate.

// TestInstanceUnsupportedNormalizes pins the same normalization WithServedModels
// gives models: a roster an operator diffs must not churn because a caller listed the
// same claim twice, in a different order, with a stray space.
func TestInstanceUnsupportedNormalizes(t *testing.T) {
	now := time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC)
	inst, r := NewInstance("guard-1", "box-a", "guard", 7, "", []Op{"pause"}, now)
	if r != nil {
		t.Fatalf("NewInstance: %v", r)
	}
	inst = inst.WithUnsupportedOps([]Op{" steer ", "steer", "", "redirect", "steer"})
	want := []Op{"redirect", "steer"}
	if len(inst.Unsupported) != len(want) {
		t.Fatalf("Unsupported = %v, want %v", inst.Unsupported, want)
	}
	for i, op := range want {
		if inst.Unsupported[i] != op {
			t.Fatalf("Unsupported = %v, want %v (trimmed, de-duped, sorted)", inst.Unsupported, want)
		}
	}
	if !inst.DeclaresUnsupported("steer") || inst.DeclaresUnsupported("pause") {
		t.Fatalf("DeclaresUnsupported disagrees with the record: %v", inst.Unsupported)
	}
	if !inst.DeclaresOp("pause") || inst.DeclaresOp("steer") {
		t.Fatalf("DeclaresOp disagrees with the record: %v", inst.Ops)
	}
}

// TestInstanceDeclaringNothingWritesNoKey is the additive-field contract: an announcer
// with nothing to declare must produce the SAME bytes a binary from before the field
// produced. A record that gained `"unsupported":[]` would make every pre-field peer
// look like it had answered the question, which is the exact ambiguity this field
// exists to remove.
func TestInstanceDeclaringNothingWritesNoKey(t *testing.T) {
	now := time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC)
	inst, r := NewInstance("serve-1", "box-a", "serve", 7, "", []Op{"pause"}, now)
	if r != nil {
		t.Fatalf("NewInstance: %v", r)
	}
	for _, arm := range []struct {
		name string
		inst Instance
	}{
		{"never built", inst},
		{"built with an empty list", inst.WithUnsupportedOps(nil)},
		{"built with only blanks", inst.WithUnsupportedOps([]Op{"", "   "})},
	} {
		b, err := json.Marshal(arm.inst)
		if err != nil {
			t.Fatalf("%s: marshal: %v", arm.name, err)
		}
		if strings.Contains(string(b), "unsupported") {
			t.Fatalf("%s: wrote an unsupported key into %s", arm.name, b)
		}
	}
}

// TestUnsupportedIsNotASelectionInput is the load-bearing negative. Declaring an op
// unsupported must NOT quietly remove the instance from a fan-out: if it did, a fleet
// where nobody can steer would answer FLEETBUS_NO_TARGET — "nobody was there" — when
// the truth is "everybody was there and none of them can". The instance stays
// addressed, stays in the denominator, and refuses at its own applier.
func TestUnsupportedIsNotASelectionInput(t *testing.T) {
	now := time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC)
	guard, r := NewInstance("guard-1", "box-a", "guard", 7, "", []Op{"pause"}, now)
	if r != nil {
		t.Fatalf("NewInstance: %v", r)
	}
	guard = guard.WithUnsupportedOps([]Op{"steer"})
	roster := []Instance{guard}

	for _, sel := range []Selector{{All: true}, {Role: []string{"guard"}}, {Instance: []string{"guard-1"}}} {
		if !sel.MatchesInstance(guard) {
			t.Fatalf("selector %s stopped matching an instance that declared steer unsupported", sel.String())
		}
		if got := PublishTargets(sel, roster); len(got) != 1 {
			t.Fatalf("PublishTargets(%s) = %d targets, want 1 — a declared-unsupported instance is still addressed", sel.String(), len(got))
		}
	}

	// ...and it stays in the fold's denominator as OUTSTANDING until it actually acks.
	d := Directive{
		Schema: DirectiveSchema, ID: "d-1", Issuer: "op", Op: "steer",
		Selector: Selector{All: true}, IssuedUTC: utc(now), TTLSec: 300,
	}
	d = d.WithTargets(roster)
	rep := Fold(d, roster, nil, now)
	if rep.Targeted != 1 || rep.Outstanding != 1 {
		t.Fatalf("Fold targeted=%d outstanding=%d, want 1/1", rep.Targeted, rep.Outstanding)
	}
}

// TestCapabilityFoldSeparatesSilentFromUnsupported is the "16 instances, 0 can steer"
// answer. The three counts must stay distinct: a silent instance is neither a yes nor
// a no, and collapsing it into either one states something no instance said.
func TestCapabilityFoldSeparatesSilentFromUnsupported(t *testing.T) {
	now := time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC)
	mk := func(id, role string, ops, unsupported []Op) Instance {
		inst, r := NewInstance(id, "box-a", role, 1, "", ops, now)
		if r != nil {
			t.Fatalf("NewInstance %s: %v", id, r)
		}
		return inst.WithUnsupportedOps(unsupported)
	}
	roster := []Instance{
		mk("guard-1", "guard", []Op{"pause", "seat-refresh"}, []Op{"steer"}),
		mk("guard-2", "guard", []Op{"pause", "seat-refresh"}, []Op{"steer"}),
		mk("serve-1", "serve", []Op{"pause", "steer"}, nil),
		// A silent instance: an old binary that lists neither. It must not be counted
		// as capable OR incapable of seat-refresh.
		mk("old-1", "serve", []Op{"pause"}, nil),
	}

	byOp := map[Op]OpSupport{}
	for _, row := range Capability(roster) {
		byOp[row.Op] = row
		if row.Declared+row.Unsupported+row.Silent != row.Total {
			t.Fatalf("%s: %d+%d+%d != total %d", row.Op, row.Declared, row.Unsupported, row.Silent, row.Total)
		}
		if row.Total != len(roster) {
			t.Fatalf("%s: total = %d, want the roster size %d", row.Op, row.Total, len(roster))
		}
	}
	if got := byOp["steer"]; got.Declared != 1 || got.Unsupported != 2 || got.Silent != 1 {
		t.Fatalf("steer = %+v, want declared 1 / unsupported 2 / silent 1", got)
	}
	if got := byOp["seat-refresh"]; got.Declared != 2 || got.Unsupported != 0 || got.Silent != 2 {
		t.Fatalf("seat-refresh = %+v, want declared 2 / silent 2", got)
	}
	if got := byOp["pause"]; got.Declared != 4 || got.Silent != 0 {
		t.Fatalf("pause = %+v, want declared 4 / silent 0", got)
	}

	// An op nobody mentioned has no row, and CapabilityFor says so as all-silent
	// against a real denominator rather than a zero value the caller has to interpret.
	if _, mentioned := byOp["teleport"]; mentioned {
		t.Fatal("Capability invented a row for an op no instance mentioned")
	}
	if got := CapabilityFor(roster, "teleport"); got.Total != 4 || got.Silent != 4 || got.Declared != 0 {
		t.Fatalf("CapabilityFor(teleport) = %+v, want 0 declared of 4 with all silent", got)
	}

	// The all-guard fleet from the issue: 16 live, 0 can steer.
	var guards []Instance
	for i := 0; i < 16; i++ {
		guards = append(guards, mk("guard-"+string(rune('a'+i)), "guard", []Op{"pause"}, []Op{"steer"}))
	}
	if got := CapabilityFor(guards, "steer"); got.Declared != 0 || got.Unsupported != 16 || got.Total != 16 {
		t.Fatalf("16-guard fleet steer = %+v, want 0 of 16 with all 16 declaring unsupported", got)
	}
}

// TestCapabilityCountsAContradictionAsUnsupported: an instance that names one op both
// ways is a bug in that instance, and this fold's whole job is to stop a fleet
// over-claiming — so the reading that cannot manufacture capacity wins. Validate must
// still accept the record: refusing it would drop a LIVE instance out of the roster
// over a display field, shrinking the denominator every other reading depends on.
func TestCapabilityCountsAContradictionAsUnsupported(t *testing.T) {
	now := time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC)
	inst, r := NewInstance("confused-1", "box-a", "guard", 1, "", []Op{"steer"}, now)
	if r != nil {
		t.Fatalf("NewInstance: %v", r)
	}
	inst = inst.WithUnsupportedOps([]Op{"steer"})
	if v := inst.Validate(); v != nil {
		t.Fatalf("Validate refused a contradictory-but-addressable record: %v", v)
	}
	got := CapabilityFor([]Instance{inst}, "steer")
	if got.Unsupported != 1 || got.Declared != 0 {
		t.Fatalf("contradiction = %+v, want it counted as unsupported", got)
	}
}

// TestLegacyRecordStaysSilentAndStaysInTheDenominator is the mixed-version arm, run
// through the REAL DirBus so it is the on-disk shape being tested, not a struct
// literal. A pre-field record must parse, stay in the roster, count as Silent (never
// as capable), and keep its place in the fold's denominator.
func TestLegacyRecordStaysSilentAndStaysInTheDenominator(t *testing.T) {
	bus, err := OpenDir(t.TempDir())
	if err != nil {
		t.Fatalf("OpenDir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(bus.Root, "instances", "legacy-1.json"),
		[]byte(legacyInstanceJSON), 0o644); err != nil {
		t.Fatalf("seed a pre-field record: %v", err)
	}
	now := time.Date(2026, 8, 5, 12, 0, 30, 0, time.UTC)
	guard, r := NewInstance("guard-1", "box-a", "guard", 2, "", []Op{"pause"}, now)
	if r != nil {
		t.Fatalf("NewInstance: %v", r)
	}
	if err := bus.Announce(guard.WithUnsupportedOps([]Op{"steer"})); err != nil {
		t.Fatalf("Announce: %v", err)
	}
	roster, err := bus.Instances(now, DefaultInstanceTTL)
	if err != nil {
		t.Fatalf("Instances: %v", err)
	}
	if len(roster) != 2 {
		t.Fatalf("roster = %d instances, want 2 (the pre-field record must not be dropped)", len(roster))
	}
	got := CapabilityFor(roster, "steer")
	// legacy-1 DECLARES steer in its ops; the guard declares it unsupported. Neither is
	// silent here, and the denominator is both of them.
	if got.Total != 2 || got.Declared != 1 || got.Unsupported != 1 || got.Silent != 0 {
		t.Fatalf("steer across a mixed-version roster = %+v, want 1 declared / 1 unsupported of 2", got)
	}
	// ...and for an op only the new binary knows about, the old record is SILENT — not
	// counted against the new op and not counted for it.
	if got := CapabilityFor(roster, "seat-refresh"); got.Total != 2 || got.Silent != 2 {
		t.Fatalf("seat-refresh across a mixed-version roster = %+v, want both silent of 2", got)
	}
}
