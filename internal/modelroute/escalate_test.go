package modelroute

import "testing"

// fullAuthority is the most permissive bounds an operator can declare: the whole ladder, and
// more attempts than any test needs. Tests that assert a REFUSAL use it deliberately, so the
// refusal cannot be explained away as a missing grant.
var fullAuthority = EscalationBounds{Ceiling: ZoneVendor, MaxAttempts: 10}

func placedIn(z PlacementZone) Placement { return Placement{Model: "m", Zone: z} }

func TestAWitnessedSuccessStandsAndAnUnverifiedOneIsCheckedRatherThanReRunHigher(t *testing.T) {
	// The rule this file exists to get right. Escalating an unwitnessed success is the
	// intuitive design and it is a trap: on a fleet whose witness wiring is incomplete it
	// escalates EVERY local success, turning local-first into vendor-by-default.
	for _, v := range []Verification{VerifyWitness, VerifyJudge} {
		got := AfterAttempt(placedIn(ZoneDevice), AttemptResult{Succeeded: true, Verify: v}, fullAuthority, 0)
		if got.Action != ActionAccept || got.Reason != ReasonAttemptStands {
			t.Errorf("independently checked success (%q) -> %q/%q, want accept", v, got.Action, got.Reason)
		}
	}
	for _, v := range []Verification{VerifyNone, "", "self", "definitely-verified"} {
		got := AfterAttempt(placedIn(ZoneDevice), AttemptResult{Succeeded: true, Verify: v}, fullAuthority, 0)
		if got.Action != ActionVerifyFirst || got.Reason != ReasonSuccessUnverified {
			t.Errorf("self-reported success (%q) -> %q/%q, want verify-then-accept", v, got.Action, got.Reason)
		}
	}
	// And no rung, no authority and no prior history turns a success into a spend.
	for _, z := range Zones() {
		for _, prior := range []int{0, 1, 9} {
			got := AfterAttempt(placedIn(z), AttemptResult{Succeeded: true}, fullAuthority, prior)
			if got.Escalates() {
				t.Errorf("a success on %q (prior=%d) escalated to %q", z, prior, got.To)
			}
		}
	}
}

func TestOnlyAnUnderpoweredFailureEarnsARung(t *testing.T) {
	// Each failure kind gets its OWN answer, because each one has a different fix: a bigger
	// model, a retry, a triage, or instrumenting the caller that could not say why.
	for _, tc := range []struct {
		fail   FailureKind
		action EscalationAction
		reason string
	}{
		{FailUnderpowered, ActionEscalate, ReasonEarnedByUnderpower},
		{FailTransport, ActionRetrySameRung, ReasonTransportRetry},
		{FailWorkItem, ActionStop, ReasonWorkItemBroken},
		{FailRefused, ActionStop, ReasonRefusalNotRetried},
		{FailUnclassified, ActionStop, ReasonFailureUnclassified},
		{FailNone, ActionStop, ReasonFailureUnclassified},             // failed, but nothing named why
		{FailureKind("weird"), ActionStop, ReasonFailureUnclassified}, // a token this vocabulary does not know
	} {
		got := AfterAttempt(placedIn(ZoneDevice), AttemptResult{Fail: tc.fail}, fullAuthority, 0)
		if got.Action != tc.action || got.Reason != tc.reason {
			t.Errorf("%q -> %q/%q, want %q/%q", tc.fail, got.Action, got.Reason, tc.action, tc.reason)
		}
	}
}

func TestARefusalIsNeverRetriedUpwardWhateverTheOperatorGranted(t *testing.T) {
	// Re-issuing a refused call on another rung is a guard bypass by retry, not a retry.
	// There is deliberately no bounds value that enables it, so sweep them all.
	for _, z := range Zones() {
		for _, b := range []EscalationBounds{{}, fullAuthority, {Ceiling: ZoneVendor, MaxAttempts: 1000}} {
			for _, prior := range []int{0, 1} {
				got := AfterAttempt(placedIn(z), AttemptResult{Fail: FailRefused}, b, prior)
				if got.Action != ActionStop || got.Reason != ReasonRefusalNotRetried {
					t.Errorf("refusal on %q with %+v -> %q/%q", z, b, got.Action, got.Reason)
				}
			}
		}
	}
}

func TestEscalationRefusesUntilAnOperatorDeclaresBothACeilingAndABudget(t *testing.T) {
	underpowered := AttemptResult{Fail: FailUnderpowered}
	// The zero value grants nothing — an escalator with no declared ceiling holds an
	// unbounded spend authority nobody wrote down.
	if got := AfterAttempt(placedIn(ZoneDevice), underpowered, EscalationBounds{}, 0); got.Action != ActionStop || got.Reason != ReasonNoCeiling {
		t.Errorf("zero bounds -> %q/%q, want stop/%s", got.Action, got.Reason, ReasonNoCeiling)
	}
	if got := AfterAttempt(placedIn(ZoneDevice), underpowered, EscalationBounds{Ceiling: "everywhere"}, 0); got.Action != ActionStop || got.Reason != ReasonNoCeiling {
		t.Errorf("unrecognised ceiling -> %q/%q, want stop/%s", got.Action, got.Reason, ReasonNoCeiling)
	}
	// A ceiling without a budget is still no authority: zero is not read as unlimited.
	if got := AfterAttempt(placedIn(ZoneDevice), underpowered, EscalationBounds{Ceiling: ZoneVendor}, 0); got.Action != ActionStop || got.Reason != ReasonBudgetSpent {
		t.Errorf("ceiling with no budget -> %q/%q, want stop/%s", got.Action, got.Reason, ReasonBudgetSpent)
	}
	// A budget is spent by prior escalations, not reset by a fresh call.
	b := EscalationBounds{Ceiling: ZoneVendor, MaxAttempts: 1}
	if got := AfterAttempt(placedIn(ZoneDevice), underpowered, b, 0); !got.Escalates() {
		t.Errorf("first escalation refused: %q/%q", got.Action, got.Reason)
	}
	if got := AfterAttempt(placedIn(ZoneFleet), underpowered, b, 1); got.Action != ActionStop || got.Reason != ReasonBudgetSpent {
		t.Errorf("second escalation on a 1-attempt budget -> %q/%q", got.Action, got.Reason)
	}
	// Granted() must agree with what the decision actually authorises: bounds that grant
	// nothing can never produce a spend, on any rung.
	for _, ungranted := range []EscalationBounds{{}, {Ceiling: ZoneVendor}, {MaxAttempts: 5}, {Ceiling: "", MaxAttempts: -1}} {
		if ungranted.Granted() {
			t.Errorf("%+v reported itself as granting authority", ungranted)
		}
		for _, z := range Zones() {
			if got := AfterAttempt(placedIn(z), underpowered, ungranted, 0); got.Escalates() {
				t.Errorf("%+v escalated from %q anyway", ungranted, z)
			}
		}
	}
}

func TestACeilingOfDeviceIsHowWorkSaysItMayNotLeaveTheBox(t *testing.T) {
	// A residency-bound class expresses itself as a ceiling rather than as a separate check
	// bolted on afterwards — the escalator then has nowhere to go, by construction.
	onBoxOnly := EscalationBounds{Ceiling: ZoneDevice, MaxAttempts: 5}
	got := AfterAttempt(placedIn(ZoneDevice), AttemptResult{Fail: FailUnderpowered}, onBoxOnly, 0)
	if got.Action != ActionStop || got.Reason != ReasonAtCeiling {
		t.Errorf("on-box-only work -> %q/%q, want stop/%s", got.Action, got.Reason, ReasonAtCeiling)
	}
	if got.To != "" {
		t.Errorf("a refused escalation still named a target rung: %q", got.To)
	}
	// A fleet ceiling authorises the first rung and refuses the second.
	selfHostedOnly := EscalationBounds{Ceiling: ZoneFleet, MaxAttempts: 5}
	up := AfterAttempt(placedIn(ZoneDevice), AttemptResult{Fail: FailUnderpowered}, selfHostedOnly, 0)
	if !up.Escalates() || up.To != ZoneFleet {
		t.Errorf("device -> %q/%q/%q, want an escalation to fleet", up.Action, up.To, up.Reason)
	}
	if !up.To.SelfHosted() {
		t.Errorf("a self-hosted-only ceiling escalated off the org's own silicon: %q", up.To)
	}
	stop := AfterAttempt(placedIn(ZoneFleet), AttemptResult{Fail: FailUnderpowered}, selfHostedOnly, 1)
	if stop.Action != ActionStop || stop.Reason != ReasonAtCeiling {
		t.Errorf("fleet under a fleet ceiling -> %q/%q, want stop/%s", stop.Action, stop.Reason, ReasonAtCeiling)
	}
}

func TestTheLadderClimbsOneRungAtATimeAndEndsAtTheTop(t *testing.T) {
	underpowered := AttemptResult{Fail: FailUnderpowered}
	for _, tc := range []struct {
		from PlacementZone
		to   PlacementZone
	}{
		{ZoneDevice, ZoneFleet},
		{ZoneFleet, ZoneVendor},
	} {
		got := AfterAttempt(placedIn(tc.from), underpowered, fullAuthority, 0)
		if !got.Escalates() || got.To != tc.to || got.From != tc.from {
			t.Errorf("%q -> %q/%q (from %q), want an escalation to %q", tc.from, got.Action, got.To, got.From, tc.to)
		}
		if got.To.Rank() <= got.From.Rank() {
			t.Errorf("escalation from %q went to %q, which is not strictly above it", got.From, got.To)
		}
	}
	// The top rung is a structural end, reported as such rather than as a spent budget — an
	// operator must be able to tell "granting more would not have helped".
	top := AfterAttempt(placedIn(ZoneVendor), underpowered, fullAuthority, 0)
	if top.Action != ActionStop || top.Reason != ReasonAtTopRung {
		t.Errorf("vendor rung -> %q/%q, want stop/%s", top.Action, top.Reason, ReasonAtTopRung)
	}
}

func TestAnUnplacedAttemptIsNamedRatherThanReadAsTheTopRung(t *testing.T) {
	// PlacementZone.Rank() ranks an unknown zone ABOVE vendor (fail-closed, so an
	// unattributable placement is never mistaken for the cheap rung). Read naively that makes
	// an EMPTY placement look like "already at the top", which hides a missing wire behind a
	// legitimate-sounding answer. It gets its own reason instead.
	for _, z := range []PlacementZone{"", "somewhere", "DEVICE"} {
		got := AfterAttempt(placedIn(z), AttemptResult{Fail: FailUnderpowered}, fullAuthority, 0)
		if got.Action != ActionStop || got.Reason != ReasonUnplacedAttempt {
			t.Errorf("placement in %q -> %q/%q, want stop/%s", z, got.Action, got.Reason, ReasonUnplacedAttempt)
		}
	}
	// Including on the success path: nothing ran, so there is nothing to accept.
	if got := AfterAttempt(Placement{}, AttemptResult{Succeeded: true, Verify: VerifyWitness}, fullAuthority, 0); got.Action != ActionStop {
		t.Errorf("an unplaced success -> %q/%q, want stop", got.Action, got.Reason)
	}
}

func TestEveryInputProducesExactlyOneNamedActionAndOnlyEscalationsNameARung(t *testing.T) {
	// Totality sweep. The decision is the gate on an automatic spend, so there must be no
	// input it answers with a blank reason, an unknown action, or a target rung it did not
	// actually authorise.
	actions := map[EscalationAction]bool{
		ActionAccept: true, ActionVerifyFirst: true, ActionRetrySameRung: true,
		ActionEscalate: true, ActionStop: true,
	}
	fails := []FailureKind{FailNone, FailUnderpowered, FailTransport, FailRefused, FailWorkItem, FailUnclassified, "junk"}
	bounds := []EscalationBounds{{}, {Ceiling: ZoneDevice, MaxAttempts: 1}, {Ceiling: ZoneFleet, MaxAttempts: 2}, fullAuthority}
	seen := map[string]bool{}
	for _, z := range append(Zones(), "", "nowhere") {
		for _, ok := range []bool{true, false} {
			for _, v := range []Verification{VerifyNone, VerifyJudge, VerifyWitness, "?"} {
				for _, f := range fails {
					for _, b := range bounds {
						for _, prior := range []int{0, 2} {
							got := AfterAttempt(placedIn(z), AttemptResult{Succeeded: ok, Verify: v, Fail: f}, b, prior)
							if !actions[got.Action] {
								t.Fatalf("unknown action %q for zone=%q ok=%v fail=%q", got.Action, z, ok, f)
							}
							if got.Reason == "" {
								t.Fatalf("blank reason for zone=%q ok=%v verify=%q fail=%q bounds=%+v", z, ok, v, f, b)
							}
							if got.Escalates() {
								if got.To.Rank() <= got.From.Rank() || !got.To.Valid() {
									t.Fatalf("escalated %q -> %q, which is not a valid rung above it", got.From, got.To)
								}
								if got.To.Rank() > b.Ceiling.Rank() {
									t.Fatalf("escalated to %q past the declared ceiling %q", got.To, b.Ceiling)
								}
								if prior >= b.MaxAttempts {
									t.Fatalf("escalated on a spent budget (prior=%d, max=%d)", prior, b.MaxAttempts)
								}
							} else if got.To != "" {
								t.Fatalf("%q named target rung %q without authorising it", got.Action, got.To)
							}
							seen[string(got.Action)] = true
						}
					}
				}
			}
		}
	}
	// The sweep must actually exercise every action, or it proves less than it looks.
	for a := range actions {
		if !seen[string(a)] {
			t.Errorf("the sweep never produced action %q", a)
		}
	}
}
