package fleetbus

import (
	"strings"
	"testing"
	"time"
)

var testNow = time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)

func testInstance(t *testing.T, id, machine, role string, now time.Time) Instance {
	t.Helper()
	inst, r := NewInstance(id, machine, role, 4242, "127.0.0.1:0", []Op{"steer"}, now)
	if r != nil {
		t.Fatalf("NewInstance(%q): %v", id, r)
	}
	return inst
}

func TestValidTokenRefusesPathEscapes(t *testing.T) {
	// Ids become path segments in the directory transport, so a token that could
	// name a file outside the bus root has to be refused by SHAPE, before any
	// transport sees it.
	for _, bad := range []string{"", ".", "..", "../evil", "a/b", `a\b`, "a b", "a:b", "a*b", strings.Repeat("x", 129)} {
		if ValidToken(bad) {
			t.Errorf("ValidToken(%q) = true, want false", bad)
		}
	}
	for _, ok := range []string{"a", "serve-1", "d-0123456789abcdef", "host.example_1", strings.Repeat("x", 128)} {
		if !ValidToken(ok) {
			t.Errorf("ValidToken(%q) = false, want true", ok)
		}
	}
}

func TestSelectorRefusesTheMatchAllYouDidNotState(t *testing.T) {
	cases := []struct {
		name string
		sel  Selector
		want bool // want refused
	}{
		{"empty selector is malformed, not match-all", Selector{}, true},
		{"session axes alone never widen to every instance", Selector{Lane: "cmd"}, true},
		{"all is the affirmative statement", Selector{All: true}, false},
		{"all plus a filter contradicts itself", Selector{All: true, Role: []string{"serve"}}, true},
		{"an instance filter addresses somebody", Selector{Instance: []string{"serve-1"}}, false},
		{"all plus session narrowing is coherent", Selector{All: true, Lane: "cmd"}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := tc.sel.Validate()
			if (r != nil) != tc.want {
				t.Fatalf("Validate() = %v, want refused=%v", r, tc.want)
			}
			if r != nil && r.Reason != Malformed {
				t.Fatalf("reason = %q, want %q", r.Reason, Malformed)
			}
		})
	}
}

func TestSelectorMatchesInstanceOnInstanceAxesOnly(t *testing.T) {
	inst := testInstance(t, "serve-1", "box-a", "serve", testNow)
	other := testInstance(t, "serve-2", "box-b", "worker", testNow)

	cases := []struct {
		name       string
		sel        Selector
		self, peer bool
	}{
		{"all addresses everyone", Selector{All: true}, true, true},
		{"by instance", Selector{Instance: []string{"serve-1"}}, true, false},
		{"by machine", Selector{Machine: []string{"box-b"}}, false, true},
		{"by role", Selector{Role: []string{"serve"}}, true, false},
		{"axes are AND'd", Selector{Machine: []string{"box-a"}, Role: []string{"worker"}}, false, false},
		{"members within an axis are OR'd", Selector{Instance: []string{"serve-1", "serve-2"}}, true, true},
		// The point of the two-level split: a session axis narrows WITHIN an
		// instance and must not change which instances are addressed.
		{"session axes do not filter instances", Selector{All: true, Lane: "cmd", Wave: "w7"}, true, true},
		{"the empty selector matches nobody", Selector{}, false, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.sel.MatchesInstance(inst); got != tc.self {
				t.Errorf("MatchesInstance(serve-1) = %v, want %v", got, tc.self)
			}
			if got := tc.sel.MatchesInstance(other); got != tc.peer {
				t.Errorf("MatchesInstance(serve-2) = %v, want %v", got, tc.peer)
			}
		})
	}
}

func TestDirectiveIDIsContentDerived(t *testing.T) {
	sel := Selector{All: true}
	base, r := NewDirective("op-a", "steer", "go", sel, time.Minute, "", testNow)
	if r != nil {
		t.Fatalf("NewDirective: %v", r)
	}
	same, _ := NewDirective("op-a", "steer", "go", sel, time.Minute, "", testNow)
	if same.ID != base.ID {
		t.Fatalf("identical directives got different ids (%s vs %s) — a retried publish would become a second command", base.ID, same.ID)
	}
	if !ValidToken(base.ID) {
		t.Fatalf("derived id %q is not a bus token", base.ID)
	}

	// Anything that changes what the directive MEANS must change the id, or the
	// apply claim would suppress a genuinely different command.
	diffs := map[string]Directive{}
	diffs["issuer"], _ = NewDirective("op-b", "steer", "go", sel, time.Minute, "", testNow)
	diffs["op"], _ = NewDirective("op-a", "pause", "go", sel, time.Minute, "", testNow)
	diffs["payload"], _ = NewDirective("op-a", "steer", "stop", sel, time.Minute, "", testNow)
	diffs["selector"], _ = NewDirective("op-a", "steer", "go", Selector{Role: []string{"serve"}}, time.Minute, "", testNow)
	diffs["ttl"], _ = NewDirective("op-a", "steer", "go", sel, 2*time.Minute, "", testNow)
	diffs["reason"], _ = NewDirective("op-a", "steer", "go", sel, time.Minute, "because", testNow)
	diffs["instant"], _ = NewDirective("op-a", "steer", "go", sel, time.Minute, "", testNow.Add(time.Second))
	for field, d := range diffs {
		if d.ID == base.ID {
			t.Errorf("changing %s did not change the directive id", field)
		}
	}
}

func TestDirectiveValidationRefusesUnusableShapes(t *testing.T) {
	good, _ := NewDirective("op", "steer", "go", Selector{All: true}, time.Minute, "", testNow)

	mutate := map[string]func(d *Directive){
		"wrong schema":      func(d *Directive) { d.Schema = "fak.fleetbus.directive/v2" },
		"unusable id":       func(d *Directive) { d.ID = "../escape" },
		"no issuer":         func(d *Directive) { d.Issuer = "  " },
		"no op":             func(d *Directive) { d.Op = "" },
		"negative ttl":      func(d *Directive) { d.TTLSec = -1 },
		"bad timestamp":     func(d *Directive) { d.IssuedUTC = "yesterday" },
		"nobody addressed":  func(d *Directive) { d.Selector = Selector{} },
		"contradictory sel": func(d *Directive) { d.Selector = Selector{All: true, Role: []string{"serve"}} },
	}
	for name, mut := range mutate {
		t.Run(name, func(t *testing.T) {
			d := good
			mut(&d)
			r := d.Validate()
			if r == nil {
				t.Fatalf("Validate() accepted %s", name)
			}
			if r.Reason != Malformed {
				t.Fatalf("reason = %q, want %q", r.Reason, Malformed)
			}
		})
	}
	if r := good.Validate(); r != nil {
		t.Fatalf("Validate() refused a good directive: %v", r)
	}
}

func TestDirectiveExpiry(t *testing.T) {
	d, _ := NewDirective("op", "steer", "go", Selector{All: true}, 10*time.Second, "", testNow)
	if d.IsExpired(testNow.Add(9 * time.Second)) {
		t.Error("expired inside its TTL")
	}
	if !d.IsExpired(testNow.Add(11 * time.Second)) {
		t.Error("did not expire past its TTL")
	}

	forever, _ := NewDirective("op", "steer", "go", Selector{All: true}, 0, "", testNow)
	if forever.TTLSec != 0 {
		t.Fatalf("TTLSec = %d, want 0 for an unbounded directive", forever.TTLSec)
	}
	if forever.IsExpired(testNow.Add(365 * 24 * time.Hour)) {
		t.Error("an unbounded directive expired")
	}

	// A sub-second TTL must round UP to one second. Rounding down to zero would
	// silently turn "expire almost immediately" into "never expires" — the one
	// direction the operator cannot have meant.
	brief, _ := NewDirective("op", "steer", "go", Selector{All: true}, 200*time.Millisecond, "", testNow)
	if brief.TTLSec != 1 {
		t.Fatalf("TTLSec = %d for a 200ms ttl, want 1", brief.TTLSec)
	}

	// A NEGATIVE ttl is the same trap from the other side: 0 is the documented "never
	// expires", so folding -5m onto it would turn an already-lapsed deadline — what a
	// wrapper computing deadline.Sub(now) produces when it runs late — into a
	// directive every instance that ever joins the bus still applies.
	if _, refusal := NewDirective("op", "steer", "go", Selector{All: true}, -5*time.Minute, "", testNow); refusal == nil {
		t.Fatal("a negative ttl was accepted; it silently means 'never expires'")
	} else if refusal.Reason != Malformed {
		t.Fatalf("negative ttl refused with %s, want %s", refusal.Reason, Malformed)
	}
}

// TestWithTargetsRecordsWhoWasAddressed — the fold's denominator floor. Ids are deduped
// and sorted so the record is stable, and the directive ID must NOT move: the id names
// the command, and who happened to be listening is a property of the publish.
func TestWithTargetsRecordsWhoWasAddressed(t *testing.T) {
	d, _ := NewDirective("op", "pause", "", Selector{All: true}, time.Minute, "", testNow)
	before := d.ID
	stamped := d.WithTargets([]Instance{
		testInstance(t, "serve-2", "box-b", "serve", testNow),
		testInstance(t, "serve-1", "box-a", "serve", testNow),
		testInstance(t, "serve-2", "box-b", "serve", testNow), // a duplicate announcement
	})
	if got := stamped.Targets; len(got) != 2 || got[0] != "serve-1" || got[1] != "serve-2" {
		t.Fatalf("Targets = %v, want [serve-1 serve-2] deduped and sorted", got)
	}
	if stamped.ID != before {
		t.Fatalf("ID moved from %q to %q; the target set is not part of what was ordered", before, stamped.ID)
	}
	if len(d.Targets) != 0 {
		t.Fatal("WithTargets mutated its receiver instead of returning a stamped copy")
	}
}

func TestInstanceFreshness(t *testing.T) {
	inst := testInstance(t, "serve-1", "box", "serve", testNow)
	if !inst.Fresh(testNow.Add(DefaultInstanceTTL-time.Second), 0) {
		t.Error("went stale inside the TTL")
	}
	if inst.Fresh(testNow.Add(DefaultInstanceTTL+time.Second), 0) {
		t.Error("stayed fresh past the TTL — a dead drainer would inflate the denominator forever")
	}
	// Clock skew across hosts must not make a live peer invisible.
	if !inst.Fresh(testNow.Add(-time.Hour), 0) {
		t.Error("a record from the future read as stale")
	}
}

func TestAckRefusesARefusalWithNoToken(t *testing.T) {
	good := Ack{
		Schema: AckSchema, Directive: "d-0123456789abcdef", Instance: "serve-1",
		Status: AckApplied, AckedUTC: utc(testNow),
	}
	if r := good.Validate(); r != nil {
		t.Fatalf("Validate() refused a good ack: %v", r)
	}

	noToken := good
	noToken.Status = AckRefused
	if r := noToken.Validate(); r == nil {
		t.Fatal("a refused ack with no closed reason was accepted — that is a refusal the control point cannot act on")
	}

	withToken := noToken
	withToken.Reason = ApplyRefused
	if r := withToken.Validate(); r != nil {
		t.Fatalf("Validate() refused a tokened refusal: %v", r)
	}

	unknownStatus := good
	unknownStatus.Status = "maybe"
	if r := unknownStatus.Validate(); r == nil {
		t.Fatal("an out-of-vocabulary ack status was accepted")
	}
}

func TestReasonsIsTheClosedVocabulary(t *testing.T) {
	want := []RefuseReason{
		"FLEETBUS_MALFORMED", "FLEETBUS_EXPIRED", "FLEETBUS_NO_TARGET",
		"FLEETBUS_UNKNOWN_OP", "FLEETBUS_APPLY_REFUSED",
	}
	got := Reasons()
	if len(got) != len(want) {
		t.Fatalf("Reasons() has %d tokens, want %d — every token needs a dos.toml [reasons.*] block", len(got), len(want))
	}
	for i, w := range want {
		if got[i] != w {
			t.Errorf("Reasons()[%d] = %q, want %q", i, got[i], w)
		}
	}
	for _, r := range got {
		if !strings.HasPrefix(string(r), "FLEETBUS_") {
			t.Errorf("token %q is not namespaced to this bus", r)
		}
	}
}
