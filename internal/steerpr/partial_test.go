package steerpr

// partial_test.go — the #5027 witness. The load-bearing case is
// TestUnknownExpectedNeverComplete: silently treating M = N on a non-derivable
// denominator is the specific failure the acceptance gate exists to catch,
// because it would render every in-flight intent as finished.

import (
	"encoding/json"
	"strings"
	"testing"
)

// fanoutBody is an issue body carrying a fanout marker key, matching what
// issuefanout.LiveBody stamps into every filed child.
func fanoutBody(leaf, slug string) string {
	return "<!-- fak-issuefanout-key: fanout-" + leaf + "-" + slug + " -->\n\n## Lane\n\n" + leaf
}

func unitOf(leaf string, n int) Unit {
	u := Unit{Leaf: leaf, Types: map[string]int{}}
	for i := 0; i < n; i++ {
		u.Commits = append(u.Commits, Commit{SHA: strings.Repeat("a", 39) + string(rune('0'+i%10)), Subject: "feat: x (fak " + leaf + ")"})
	}
	return u
}

// TestPartialTable is the table the issue's Witness section names: a derivable M
// from open fanout children, an explicit cohort wave, and a non-derivable case
// asserting expected: unknown.
func TestPartialTable(t *testing.T) {
	spineOnly := []IntentIssue{{Number: 5015, Body: "the spine"}}
	withChildren := []IntentIssue{
		{Number: 5015, Body: "the spine"},
		{Number: 5027, Body: fanoutBody("steerpr", "partial-bundle")},
		{Number: 5028, Body: fanoutBody("steerpr", "ack")},
		{Number: 5030, Body: fanoutBody("steerpr", "redirect")},
		{Number: 6000, Body: fanoutBody("gateway", "other-leaf")}, // different leaf: not a member
	}

	tests := []struct {
		name         string
		landed       int
		exp          Expectation
		ok           bool
		wantKnown    bool
		wantExpected int
		wantComplete bool
		wantForming  bool
		wantSource   string
	}{
		{
			name:   "derivable M from spine plus open fanout children",
			landed: 1,
			exp: func() Expectation {
				e, _ := DeriveExpected("steerpr", "#5015", withChildren)
				return e
			}(),
			ok:           true,
			wantKnown:    true,
			wantExpected: 4, // spine + 3 steerpr children; the gateway child is not a member
			wantComplete: false,
			wantForming:  true,
			wantSource:   SourceFanout,
		},
		{
			name:   "spine with no children is a real M of 1",
			landed: 1,
			exp: func() Expectation {
				e, _ := DeriveExpected("steerpr", "#5015", spineOnly)
				return e
			}(),
			ok:           true,
			wantKnown:    true,
			wantExpected: 1,
			wantComplete: true,
			wantForming:  false,
			wantSource:   SourceFanout,
		},
		{
			name:   "explicit cohort wave size",
			landed: 3,
			exp: func() Expectation {
				e, _ := CohortExpectation(12)
				return e
			}(),
			ok:           true,
			wantKnown:    true,
			wantExpected: 12,
			wantComplete: false,
			wantForming:  true,
			wantSource:   SourceCohort,
		},
		{
			name:         "non-derivable denominator is unknown",
			landed:       7,
			exp:          Expectation{},
			ok:           false,
			wantKnown:    false,
			wantComplete: false,
			wantForming:  false,
		},
		{
			name:         "landed exceeding M reports complete, never re-fabricates M",
			landed:       9,
			exp:          Expectation{Total: 4, Source: SourceFanout},
			ok:           true,
			wantKnown:    true,
			wantExpected: 4,
			wantComplete: true,
			wantForming:  false,
			wantSource:   SourceFanout,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := NewPartial(tc.landed, tc.exp, tc.ok)
			if got := p.Known(); got != tc.wantKnown {
				t.Fatalf("Known() = %v, want %v", got, tc.wantKnown)
			}
			if got := p.Complete; got != tc.wantComplete {
				t.Errorf("Complete = %v, want %v", got, tc.wantComplete)
			}
			if got := p.Forming(); got != tc.wantForming {
				t.Errorf("Forming() = %v, want %v", got, tc.wantForming)
			}
			if got := p.Landed; got != tc.landed {
				t.Errorf("Landed = %d, want %d", got, tc.landed)
			}
			if tc.wantKnown {
				if p.Expected == nil {
					t.Fatalf("Expected = nil, want %d", tc.wantExpected)
				}
				if *p.Expected != tc.wantExpected {
					t.Errorf("Expected = %d, want %d", *p.Expected, tc.wantExpected)
				}
				if p.Source != tc.wantSource {
					t.Errorf("Source = %q, want %q", p.Source, tc.wantSource)
				}
			} else if p.Expected != nil {
				t.Errorf("Expected = %d, want nil (unknown must stay unknown)", *p.Expected)
			}
		})
	}
}

// TestUnknownExpectedNeverComplete is the #5027 ACCEPTANCE GATE: an unknown
// denominator must never render complete, at ANY landed count. The failure this
// catches is a future implementation defaulting M = N, which would make every
// forming unit read as finished — inverting the signal's purpose.
func TestUnknownExpectedNeverComplete(t *testing.T) {
	for _, landed := range []int{0, 1, 3, 50, 5000} {
		p := NewPartial(landed, Expectation{}, false)
		if p.Complete {
			t.Errorf("landed=%d: unknown denominator rendered complete — M = N was silently fabricated", landed)
		}
		if p.Known() {
			t.Errorf("landed=%d: unknown denominator reported Known()", landed)
		}
		if p.Expected != nil {
			t.Errorf("landed=%d: unknown denominator carried Expected = %d", landed, *p.Expected)
		}
		if _, ok := p.Remaining(); ok {
			t.Errorf("landed=%d: unknown denominator reported a remaining count", landed)
		}
		// The rendered line must SAY unknown, not go quiet: silence reads as
		// completeness to an operator scanning the overlay.
		line := p.Annotate()
		if !strings.Contains(line, "unknown") {
			t.Errorf("landed=%d: Annotate() = %q, want it to name the unknown denominator", landed, line)
		}
		if strings.Contains(line, "complete:") {
			t.Errorf("landed=%d: Annotate() = %q, rendered as complete", landed, line)
		}
	}
}

// TestUnknownExpectedMarshalsExplicitlyNull pins the payload contract: expected
// is always PRESENT and null when unknown, never omitted. An omitted key would
// let a machine consumer default it to 0 and compute completeness itself.
func TestUnknownExpectedMarshalsExplicitlyNull(t *testing.T) {
	buf, err := json.Marshal(NewPartial(3, Expectation{}, false))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got := string(buf)
	if !strings.Contains(got, `"expected":null`) {
		t.Errorf("payload = %s, want an explicit \"expected\":null", got)
	}
	if !strings.Contains(got, `"complete":false`) {
		t.Errorf("payload = %s, want \"complete\":false", got)
	}
	if !strings.Contains(got, `"landed":3`) {
		t.Errorf("payload = %s, want \"landed\":3", got)
	}
}

// TestDeriveExpectedRefusesWithoutSpine pins the derivability requirement: a
// graph the caller could not read (gh absent, empty scan, wrong repo) must yield
// unknown, never M = 1. Deriving a denominator from an empty gather is exactly
// the fabricated-M failure mode in a different costume.
func TestDeriveExpectedRefusesWithoutSpine(t *testing.T) {
	tests := []struct {
		name   string
		leaf   string
		spine  string
		issues []IntentIssue
	}{
		{"empty gather", "steerpr", "#5015", nil},
		{"spine absent from gather", "steerpr", "#5015", []IntentIssue{{Number: 9999, Body: fanoutBody("steerpr", "x")}}},
		{"no bound spine", "steerpr", "", []IntentIssue{{Number: 5015, Body: "spine"}}},
		{"no leaf", "", "#5015", []IntentIssue{{Number: 5015, Body: "spine"}}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			exp, ok := DeriveExpected(tc.leaf, tc.spine, tc.issues)
			if ok {
				t.Fatalf("DeriveExpected = (%+v, true), want unknown", exp)
			}
			if p := NewPartial(2, exp, ok); p.Complete || p.Known() {
				t.Errorf("undervivable expectation produced known/complete partial %+v", p)
			}
		})
	}
}

// TestCohortExpectationRefusesNonPositive pins that a mis-set wave degrades to
// unknown rather than to M = 0, which would render every unit complete.
func TestCohortExpectationRefusesNonPositive(t *testing.T) {
	for _, wave := range []int{0, -1, -12} {
		exp, ok := CohortExpectation(wave)
		if ok {
			t.Errorf("CohortExpectation(%d) = (%+v, true), want unknown", wave, exp)
		}
		if p := NewPartial(4, exp, ok); p.Complete {
			t.Errorf("CohortExpectation(%d) produced a complete partial", wave)
		}
	}
}

// TestAttachPartialsMarksUnknownExplicitly pins the deliberate difference from
// AttachCurves: a unit whose lookup fails gets an EXPLICITLY unknown Partial, not
// a nil one. Dropping to nil would make "unknown" indistinguishable from "not
// asked", and the operator would read neither.
func TestAttachPartialsMarksUnknownExplicitly(t *testing.T) {
	units := []Unit{unitOf("steerpr", 1), unitOf("gateway", 3)}
	AttachPartials(units, func(u Unit) (Expectation, bool) {
		if u.Leaf == "steerpr" {
			return Expectation{Total: 4, Source: SourceFanout}, true
		}
		return Expectation{}, false
	})

	if units[0].Partial == nil || !units[0].Partial.Forming() {
		t.Fatalf("steerpr unit = %+v, want a forming partial", units[0].Partial)
	}
	if units[1].Partial == nil {
		t.Fatal("gateway unit dropped to a nil Partial — unknown must be explicit, not absent")
	}
	if units[1].Partial.Known() || units[1].Partial.Complete {
		t.Errorf("gateway unit = %+v, want an explicitly unknown, non-complete partial", units[1].Partial)
	}

	forming := PartialUnits(units)
	if len(forming) != 1 || forming[0].Leaf != "steerpr" {
		t.Errorf("PartialUnits = %+v, want just the steerpr unit", forming)
	}
	unknown := UnknownExpectedUnits(units)
	if len(unknown) != 1 || unknown[0].Leaf != "gateway" {
		t.Errorf("UnknownExpectedUnits = %+v, want just the gateway unit", unknown)
	}
}

// TestAnnotateDistinguishesFormingFromComplete pins the render half of the done
// condition: a forming unit and a finished one must not read the same.
func TestAnnotateDistinguishesFormingFromComplete(t *testing.T) {
	forming := NewPartial(3, Expectation{Total: 12, Source: SourceCohort}, true)
	complete := NewPartial(12, Expectation{Total: 12, Source: SourceCohort}, true)

	fl, cl := forming.Annotate(), complete.Annotate()
	if fl == cl {
		t.Fatalf("forming and complete render identically: %q", fl)
	}
	if !strings.Contains(fl, "3 of 12") || !strings.Contains(fl, "forming") {
		t.Errorf("forming line = %q, want it to read \"3 of 12\" and say forming", fl)
	}
	if !strings.Contains(fl, "9 outstanding") {
		t.Errorf("forming line = %q, want the outstanding count", fl)
	}
	if !strings.Contains(cl, "complete") || !strings.Contains(cl, "12 of 12") {
		t.Errorf("complete line = %q, want it to read complete at 12 of 12", cl)
	}
	if (*Partial)(nil).Annotate() != "" {
		t.Error("a nil Partial must render nothing, not a warning")
	}
}

// TestWithPartialCountsItsOwnMembers pins that the numerator comes from the
// unit's real member list, so N can never disagree with the commits printed
// directly beneath it.
func TestWithPartialCountsItsOwnMembers(t *testing.T) {
	u := unitOf("steerpr", 5).WithPartial(Expectation{Total: 9, Source: SourceFanout}, true)
	if u.Partial == nil || u.Partial.Landed != 5 {
		t.Fatalf("Partial = %+v, want Landed 5 from the unit's own members", u.Partial)
	}
	if !u.Partial.Forming() {
		t.Error("5 of 9 must read as forming")
	}
	if n, ok := u.Partial.Remaining(); !ok || n != 4 {
		t.Errorf("Remaining = (%d, %v), want (4, true)", n, ok)
	}
}
