package main

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// escRoot is a workspace where the ladder is on, every rung is graded and dialable, and the
// operator has granted a vendor ceiling with two rungs per item. Each test then removes or
// narrows exactly the one thing it is about, so a refusal in a test is never ambient.
func escRoot(t *testing.T) string {
	t.Helper()
	root := rungRoot(t)
	withRungRoster(t, root)
	var turns []modelroute.TurnOutcome
	for _, m := range []string{"qwen3.6-4b", "glm-5.2", "opus-5"} {
		turns = append(turns, rungTurns(m, 24, time.Minute, modelroute.VerifyWitness)...)
	}
	withRungJournal(t, root, turns)
	t.Setenv(dispatchRungCeilingEnv, string(modelroute.ZoneVendor))
	t.Setenv(dispatchRungBudgetEnv, "2")
	return root
}

// underpoweredSlot is a finished slot that RAN the work and failed its own tests — the only
// outcome in the sweep's vocabulary that earns a rung.
func underpoweredSlot(issue int, log string, zone modelroute.PlacementZone, model string) dispatchtick.WitnessRecord {
	return dispatchtick.WitnessRecord{
		Issue:     issue,
		Log:       log,
		SHA:       "deadbeef",
		Claim:     dispatchtick.ClaimWitnessed,
		Model:     model,
		Zone:      string(zone),
		TestClaim: dispatchtick.ClaimTestRed,
	}
}

func escLedgerRows(t *testing.T, root string) []modelroute.EscalationRecord {
	t.Helper()
	f, err := os.Open(dispatchEscalationLedgerPath(root))
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	recs, _, err := modelroute.ReadEscalations(f)
	if err != nil {
		t.Fatal(err)
	}
	return recs
}

func mustNotSpend(t *testing.T, root string) {
	t.Helper()
	if rows := escLedgerRows(t, root); len(rows) != 0 {
		t.Errorf("a refusal still wrote %d debit(s): %+v", len(rows), rows)
	}
}

// With the seam off nothing is reported at all, so an unconfigured fleet's tick payload is
// byte-identical to before this existed. Every OTHER silence here is a decision and names
// itself.
func TestAnUnconfiguredTickNeverEscalates(t *testing.T) {
	root := escRoot(t)
	setDispatchRungPlacement(t, false)
	before := seatDefaultFor("opus-5")
	after, esc, skip := applyRungEscalation(root, true, 5416, routineTierLabels,
		[]dispatchtick.WitnessRecord{underpoweredSlot(5416, "a.log", modelroute.ZoneDevice, "qwen3.6-4b")}, before)
	if skip != "" || esc != nil {
		t.Errorf("an off seam reported %q / %+v, which the caller would write into every payload", skip, esc)
	}
	if !reflect.DeepEqual(after, before) {
		t.Errorf("an off seam changed the policy: %+v -> %+v", before, after)
	}
	mustNotSpend(t, root)
}

// The design, end to end: a slot that tried on the device rung and failed its tests comes back
// on the fleet rung — still self-hosted, which is the objective — and the rung it bought is on
// disk before the pin exists.
func TestARungIsRecordedBeforeItIsUsed(t *testing.T) {
	root := escRoot(t)
	rec := underpoweredSlot(5416, "slot-a.log", modelroute.ZoneDevice, "qwen3.6-4b")
	after, esc, skip := applyRungEscalation(root, true, 5416, routineTierLabels,
		[]dispatchtick.WitnessRecord{rec}, seatDefaultFor("opus-5"))
	if skip != "" {
		t.Fatalf("skip = %q, want an escalation", skip)
	}
	if esc == nil {
		t.Fatal("escalated with nothing to report")
	}
	if after.Model != "glm-5.2" {
		t.Errorf("model = %q, want the fleet rung glm-5.2", after.Model)
	}
	if after.Source != modelSourceEscalated {
		t.Errorf("source = %q, want %q", after.Source, modelSourceEscalated)
	}
	if esc.From != modelroute.ZoneDevice || esc.To != modelroute.ZoneFleet {
		t.Errorf("from/to = %s/%s, want device/fleet", esc.From, esc.To)
	}
	if esc.Reason != modelroute.ReasonEarnedByUnderpower {
		t.Errorf("reason = %q, want %q", esc.Reason, modelroute.ReasonEarnedByUnderpower)
	}
	if esc.Replayed {
		t.Error("a first escalation reported itself as a replay")
	}

	rows := escLedgerRows(t, root)
	if len(rows) != 1 {
		t.Fatalf("ledger holds %d rows, want exactly the one rung bought", len(rows))
	}
	if rows[0].Item != "5416" {
		t.Errorf("item = %q, want the work item that spent it", rows[0].Item)
	}
	if rows[0].From != modelroute.ZoneDevice || rows[0].To != modelroute.ZoneFleet {
		t.Errorf("row from/to = %s/%s, want device/fleet", rows[0].From, rows[0].To)
	}
	if rows[0].ID != escalationDebitID(rec) {
		t.Errorf("row id = %q, want the attempt it escalated away from", rows[0].ID)
	}
	if rows[0].At.IsZero() {
		t.Error("a debit with no timestamp: the ledger is the audit trail for an automatic spend")
	}
}

// One debit per ATTEMPT, not per tick. The same finished slot turns up in consecutive sweeps
// until it is reaped, and re-deciding it each time would drain the item's budget by looking at
// it — which reads as "this item exhausted its ladder" when it made a single failed attempt.
func TestTheSameFinishedAttemptIsChargedOnce(t *testing.T) {
	root := escRoot(t)
	records := []dispatchtick.WitnessRecord{underpoweredSlot(5416, "slot-a.log", modelroute.ZoneDevice, "qwen3.6-4b")}

	first, esc1, skip := applyRungEscalation(root, true, 5416, routineTierLabels, records, seatDefaultFor())
	if skip != "" || esc1 == nil {
		t.Fatalf("first escalation refused: %q", skip)
	}
	second, esc2, skip := applyRungEscalation(root, true, 5416, routineTierLabels, records, seatDefaultFor())
	if skip != "" || esc2 == nil {
		t.Fatalf("second look refused: %q", skip)
	}
	if len(escLedgerRows(t, root)) != 1 {
		t.Errorf("the same attempt was charged %d times", len(escLedgerRows(t, root)))
	}
	if second.Model != first.Model {
		t.Errorf("the replay landed elsewhere: %q then %q", first.Model, second.Model)
	}
	if !esc2.Replayed {
		t.Error("a replay did not say so, so an operator would read two purchases in the payload")
	}
	if esc2.To != esc1.To {
		t.Errorf("replay rung = %s, want the rung already bought (%s)", esc2.To, esc1.To)
	}
}

// The budget is a real bound across attempts, counted from disk rather than from memory. Two
// DISTINCT attempts spend two rungs; a third finds the item at its ceiling of purchases.
func TestAnItemAtItsBudgetStopsClimbing(t *testing.T) {
	root := escRoot(t)
	t.Setenv(dispatchRungBudgetEnv, "1")

	if _, _, skip := applyRungEscalation(root, true, 5416, routineTierLabels,
		[]dispatchtick.WitnessRecord{underpoweredSlot(5416, "slot-a.log", modelroute.ZoneDevice, "qwen3.6-4b")}, seatDefaultFor()); skip != "" {
		t.Fatalf("the first rung was refused: %q", skip)
	}
	before := seatDefaultFor()
	after, esc, skip := applyRungEscalation(root, true, 5416, routineTierLabels,
		[]dispatchtick.WitnessRecord{underpoweredSlot(5416, "slot-b.log", modelroute.ZoneFleet, "glm-5.2")}, before)
	if skip != modelroute.ReasonBudgetSpent {
		t.Errorf("skip = %q, want %q", skip, modelroute.ReasonBudgetSpent)
	}
	if esc != nil || !reflect.DeepEqual(after, before) {
		t.Errorf("a spent budget still moved the worker: %+v / %+v", after, esc)
	}
	if len(escLedgerRows(t, root)) != 1 {
		t.Errorf("ledger holds %d rows, want only the one rung that was authorised", len(escLedgerRows(t, root)))
	}
}

// A budget spent by ANOTHER item does not stop this one. Spent() charges unattributable debits
// to everybody, and a per-item count that quietly became a global one would look identical from
// the payload while switching the whole ladder off after N escalations fleet-wide.
func TestOneItemsSpendDoesNotStopAnother(t *testing.T) {
	root := escRoot(t)
	t.Setenv(dispatchRungBudgetEnv, "1")

	if _, _, skip := applyRungEscalation(root, true, 5416, routineTierLabels,
		[]dispatchtick.WitnessRecord{underpoweredSlot(5416, "slot-a.log", modelroute.ZoneDevice, "qwen3.6-4b")}, seatDefaultFor()); skip != "" {
		t.Fatalf("the first item was refused: %q", skip)
	}
	after, esc, skip := applyRungEscalation(root, true, 9999, routineTierLabels,
		[]dispatchtick.WitnessRecord{underpoweredSlot(9999, "slot-b.log", modelroute.ZoneDevice, "qwen3.6-4b")}, seatDefaultFor())
	if skip != "" || esc == nil {
		t.Fatalf("a fresh item inherited another's spend: %q", skip)
	}
	if after.Model != "glm-5.2" {
		t.Errorf("model = %q, want glm-5.2", after.Model)
	}
}

// The mirror-image reading rule, end to end through the actuator: a torn row in the ledger is
// a debit nobody can attribute, and it is CHARGED. The evidence journal would drop the same
// line, because there a lost row only weakens a claim.
func TestATornLedgerRowStillCountsAgainstTheBudget(t *testing.T) {
	root := escRoot(t)
	t.Setenv(dispatchRungBudgetEnv, "1")
	if err := os.MkdirAll(filepath.Join(root, dispatchtick.RunsDirName), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dispatchEscalationLedgerPath(root), []byte("{\"item\":\"5416\",\"t\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, esc, skip := applyRungEscalation(root, true, 5416, routineTierLabels,
		[]dispatchtick.WitnessRecord{underpoweredSlot(5416, "slot-a.log", modelroute.ZoneDevice, "qwen3.6-4b")}, seatDefaultFor())
	if skip != modelroute.ReasonBudgetSpent || esc != nil {
		t.Errorf("skip = %q / esc = %+v, want the unreadable debit to have been charged", skip, esc)
	}
}

// Only an underpowered failure buys a rung. Everything else in the sweep's vocabulary gets a
// named refusal and spends nothing — a guard refusal above all, since re-aiming a refusal at a
// bigger model is a bypass rather than a retry.
func TestOnlyAnUnderpoweredAttemptBuysARung(t *testing.T) {
	cases := []struct {
		name string
		rec  dispatchtick.WitnessRecord
		want string
	}{
		{"a guard refusal", dispatchtick.WitnessRecord{Issue: 5416, Log: "a.log", Claim: dispatchtick.ClaimNoCommit,
			Reason: dispatchtick.NoCommitPolicyBlock, Zone: string(modelroute.ZoneDevice)}, modelroute.ReasonRefusalNotRetried},
		{"an off-trunk refusal", dispatchtick.WitnessRecord{Issue: 5416, Log: "a.log", Claim: dispatchtick.ClaimNoCommit,
			Reason: dispatchtick.NoCommitOffTrunk, Zone: string(modelroute.ZoneDevice)}, modelroute.ReasonRefusalNotRetried},
		{"a usage wall", dispatchtick.WitnessRecord{Issue: 5416, Log: "a.log", Claim: dispatchtick.ClaimNoCommit,
			Reason: dispatchtick.NoCommitUsageCap, Zone: string(modelroute.ZoneDevice)}, modelroute.ReasonTransportRetry},
		{"an unknown model", dispatchtick.WitnessRecord{Issue: 5416, Log: "a.log", Claim: dispatchtick.ClaimNoCommit,
			Reason: dispatchtick.NoCommitModelUnknown, Zone: string(modelroute.ZoneDevice)}, modelroute.ReasonTransportRetry},
		{"an unwitnessed claim with no test rung", dispatchtick.WitnessRecord{Issue: 5416, Log: "a.log",
			Claim: dispatchtick.ClaimUnwitnessed, Zone: string(modelroute.ZoneDevice)}, modelroute.ReasonFailureUnclassified},
		{"a witnessed success", dispatchtick.WitnessRecord{Issue: 5416, Log: "a.log", Claim: dispatchtick.ClaimWitnessed,
			TestClaim: dispatchtick.ClaimTestGreen, Zone: string(modelroute.ZoneDevice)}, modelroute.ReasonAttemptStands},
		{"a slot whose rung nobody recorded", dispatchtick.WitnessRecord{Issue: 5416, Log: "a.log",
			Claim: dispatchtick.ClaimWitnessed, TestClaim: dispatchtick.ClaimTestRed}, modelroute.ReasonUnplacedAttempt},
		{"a red test at the top rung", dispatchtick.WitnessRecord{Issue: 5416, Log: "a.log", Claim: dispatchtick.ClaimWitnessed,
			TestClaim: dispatchtick.ClaimTestRed, Zone: string(modelroute.ZoneVendor)}, modelroute.ReasonAtTopRung},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			root := escRoot(t)
			before := seatDefaultFor("opus-5")
			after, esc, skip := applyRungEscalation(root, true, 5416, routineTierLabels,
				[]dispatchtick.WitnessRecord{tc.rec}, before)
			if skip != tc.want {
				t.Errorf("skip = %q, want %q", skip, tc.want)
			}
			if esc != nil || !reflect.DeepEqual(after, before) {
				t.Errorf("a refusal still moved the worker: %+v / %+v", after, esc)
			}
			mustNotSpend(t, root)
		})
	}
}

// The escalator and the Layer-2 downgrade are disjoint by construction, and it matters: they
// pull in opposite directions on the same slot. A model-switchable wall is a TRANSPORT
// failure — nothing reached a model that could try the work — so it retries where it stood
// while the downgrade re-aims it, and only an outcome the downgrade ignores earns a rung.
func TestEscalationAndDowngradeNeverFireOnTheSameSlot(t *testing.T) {
	root := escRoot(t)
	for _, reason := range []string{dispatchtick.NoCommitUsageCap, dispatchtick.NoCommitModelUnknown, dispatchtick.NoCommitRateLimit} {
		if !dispatchtick.ModelSwitchableReason(reason) {
			t.Fatalf("precondition: %q is no longer a downgrade reason", reason)
		}
		rec := dispatchtick.WitnessRecord{Issue: 5416, Log: "a.log", Claim: dispatchtick.ClaimNoCommit,
			Reason: reason, Zone: string(modelroute.ZoneDevice)}
		if _, _, skip := applyRungEscalation(root, true, 5416, routineTierLabels,
			[]dispatchtick.WitnessRecord{rec}, seatDefaultFor()); skip != modelroute.ReasonTransportRetry {
			t.Errorf("%s: skip = %q, want %q", reason, skip, modelroute.ReasonTransportRetry)
		}
	}
	earned := underpoweredSlot(5416, "b.log", modelroute.ZoneDevice, "qwen3.6-4b")
	if dispatchtick.ModelSwitchableReason(earned.Reason) {
		t.Error("the rung-earning outcome is also a downgrade outcome; the two would fight over one slot")
	}
	if _, esc, skip := applyRungEscalation(root, true, 5416, routineTierLabels,
		[]dispatchtick.WitnessRecord{earned}, seatDefaultFor()); skip != "" || esc == nil {
		t.Errorf("the rung-earning outcome was refused: %q", skip)
	}
	// Exactly one debit across all four outcomes: the three downgrade reasons bought nothing.
	if rows := escLedgerRows(t, root); len(rows) != 1 {
		t.Errorf("ledger holds %d rows, want only the underpowered outcome's", len(rows))
	}
}

// Escalation is an automatic spend, so it needs an authority somebody wrote down. Each way of
// not having one is named separately, because the cures are different: nothing declared, a
// ceiling that is not a rung, a budget that is not a number.
func TestEscalationRefusesUntilTheAuthorityIsDeclared(t *testing.T) {
	cases := []struct {
		name            string
		ceiling, budget string
		unsetBudget     bool
		want            string
	}{
		{name: "nothing declared", unsetBudget: true, want: escSkipNoBudgetDecl},
		{name: "a budget with no ceiling", budget: "2", want: modelroute.ReasonNoCeiling},
		{name: "a ceiling that is not a rung", ceiling: "fleeet", budget: "2", want: escSkipBadCeiling},
		{name: "a budget that is not a number", ceiling: "vendor", budget: "lots", want: escSkipBadBudget},
		{name: "a budget of zero", ceiling: "vendor", budget: "0", want: modelroute.ReasonBudgetSpent},
		{name: "a negative budget", ceiling: "vendor", budget: "-1", want: modelroute.ReasonBudgetSpent},
		{name: "an on-box ceiling", ceiling: "device", budget: "2", want: modelroute.ReasonAtCeiling},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			root := escRoot(t)
			t.Setenv(dispatchRungCeilingEnv, tc.ceiling)
			if tc.unsetBudget {
				os.Unsetenv(dispatchRungBudgetEnv)
			} else {
				t.Setenv(dispatchRungBudgetEnv, tc.budget)
			}
			_, esc, skip := applyRungEscalation(root, true, 5416, routineTierLabels,
				[]dispatchtick.WitnessRecord{underpoweredSlot(5416, "a.log", modelroute.ZoneDevice, "qwen3.6-4b")}, seatDefaultFor())
			if skip != tc.want {
				t.Errorf("skip = %q, want %q", skip, tc.want)
			}
			if esc != nil {
				t.Errorf("an undeclared authority still escalated: %+v", esc)
			}
			mustNotSpend(t, root)
		})
	}
}

// The declared ceiling bounds the WALK, not just the first step. Without a bound from above,
// an empty rung between the earned one and the vendor would let Place keep climbing — which
// silently upgrades a fleet ceiling to a vendor bill, the one outcome this epic exists to
// avoid.
func TestTheCeilingBoundsTheClimbAndNotJustItsFirstStep(t *testing.T) {
	root := rungRoot(t)
	withRungRoster(t, root)
	// Only the two ENDS of the ladder are graded: the fleet rung is bound but unmeasured, so
	// the earned rung cannot serve and the next one up is the vendor.
	var turns []modelroute.TurnOutcome
	for _, m := range []string{"qwen3.6-4b", "opus-5"} {
		turns = append(turns, rungTurns(m, 24, time.Minute, modelroute.VerifyWitness)...)
	}
	withRungJournal(t, root, turns)
	t.Setenv(dispatchRungBudgetEnv, "2")
	rec := underpoweredSlot(5416, "a.log", modelroute.ZoneDevice, "qwen3.6-4b")

	t.Setenv(dispatchRungCeilingEnv, string(modelroute.ZoneFleet))
	after, esc, skip := applyRungEscalation(root, true, 5416, routineTierLabels,
		[]dispatchtick.WitnessRecord{rec}, seatDefaultFor())
	if esc != nil || after.Model == "opus-5" {
		t.Fatalf("a fleet ceiling reached the vendor rung: %+v / %+v", after, esc)
	}
	// The exact reason is the assertion, not just the refusal. Under the ceiling the pool
	// holds only the unservable fleet rung and Place declines it. A pool NOT bounded from
	// above would instead hand back a vendor placement and the outcome check further down
	// would catch it — same silence, different cause — so an unbounded search would pass a
	// test that only asked "did it refuse".
	if skip != rungSkipRefused {
		t.Errorf("skip = %q, want %q — the pool offered more than the ceiling authorised", skip, rungSkipRefused)
	}
	mustNotSpend(t, root)

	// The contrast that proves the ceiling did the work, not the evidence: raise it and the
	// same inputs climb past the empty fleet rung to the vendor.
	t.Setenv(dispatchRungCeilingEnv, string(modelroute.ZoneVendor))
	after, esc, skip = applyRungEscalation(root, true, 5416, routineTierLabels,
		[]dispatchtick.WitnessRecord{rec}, seatDefaultFor())
	if skip != "" || esc == nil {
		t.Fatalf("a vendor ceiling refused: %q", skip)
	}
	if after.Model != "opus-5" || esc.To != modelroute.ZoneVendor {
		t.Errorf("model/to = %q/%s, want opus-5 on the vendor rung", after.Model, esc.To)
	}
}

// A ceiling LOWERED after a rung was bought applies to the next launch, not only to the next
// purchase. The replay path is the one that could quietly keep honouring the old authority.
func TestALoweredCeilingAppliesToAnAlreadyBoughtRung(t *testing.T) {
	root := escRoot(t)
	records := []dispatchtick.WitnessRecord{underpoweredSlot(5416, "a.log", modelroute.ZoneFleet, "glm-5.2")}
	after, esc, skip := applyRungEscalation(root, true, 5416, routineTierLabels, records, seatDefaultFor())
	if skip != "" || after.Model != "opus-5" {
		t.Fatalf("precondition: skip = %q, model = %q, want the vendor rung", skip, after.Model)
	}
	if esc.To != modelroute.ZoneVendor {
		t.Fatalf("precondition: to = %s, want vendor", esc.To)
	}

	t.Setenv(dispatchRungCeilingEnv, string(modelroute.ZoneFleet))
	before := seatDefaultFor()
	after, esc, skip = applyRungEscalation(root, true, 5416, routineTierLabels, records, before)
	if skip != modelroute.ReasonAtCeiling {
		t.Errorf("skip = %q, want %q", skip, modelroute.ReasonAtCeiling)
	}
	if esc != nil || !reflect.DeepEqual(after, before) {
		t.Errorf("a lowered ceiling still replayed the old rung: %+v / %+v", after, esc)
	}
}

// A preview tick must not spend a budget: the attempt it would be charging for is never
// launched. Pinning without charging would be worse — a rung outside the budget — so a
// non-live tick reports itself and changes nothing.
func TestAPreviewTickNeverSpendsABudget(t *testing.T) {
	root := escRoot(t)
	before := seatDefaultFor("opus-5")
	after, esc, skip := applyRungEscalation(root, false, 5416, routineTierLabels,
		[]dispatchtick.WitnessRecord{underpoweredSlot(5416, "a.log", modelroute.ZoneDevice, "qwen3.6-4b")}, before)
	if skip != escSkipNotLive {
		t.Errorf("skip = %q, want %q", skip, escSkipNotLive)
	}
	if esc != nil || !reflect.DeepEqual(after, before) {
		t.Errorf("a preview tick moved a worker: %+v / %+v", after, esc)
	}
	mustNotSpend(t, root)
}

// Only the ladder's OWN choices may be raised. An operator pin, a lane pin and the bench gate
// are deliberate human intent and outrank an automatic spend — the same doctrine the
// preventive placement gate follows.
func TestOnlyTheLaddersOwnChoiceMayBeRaised(t *testing.T) {
	root := escRoot(t)
	records := []dispatchtick.WitnessRecord{underpoweredSlot(5416, "a.log", modelroute.ZoneDevice, "qwen3.6-4b")}

	for _, source := range []string{modelSourceExplicit, modelSourceLane, modelSourceAccount,
		modelSourceProfile, modelSourceTier, modelSourceWorkClass, modelSourceDowngrade, modelSourcePlacement} {
		before := workerModelPolicy{Model: "operator-choice", Source: source}
		after, esc, skip := applyRungEscalation(root, true, 5416, routineTierLabels, records, before)
		if skip != escSkipOutranked {
			t.Errorf("%s: skip = %q, want %q", source, skip, escSkipOutranked)
		}
		if esc != nil || !reflect.DeepEqual(after, before) {
			t.Errorf("%s: an outranked decision was overwritten: %+v", source, after)
		}
	}
	mustNotSpend(t, root)

	// The two the ladder owns do move.
	for _, source := range []string{modelSourceSeatDefault, modelSourceRung} {
		root := escRoot(t)
		before := workerModelPolicy{Source: source}
		if source == modelSourceRung {
			before.Model = "qwen3.6-4b"
		}
		if _, esc, skip := applyRungEscalation(root, true, 5416, routineTierLabels, records, before); skip != "" || esc == nil {
			t.Errorf("%s: the ladder refused to raise its own choice: %q", source, skip)
		}
	}
}

// A rung the operator has not declared dialable is not a rung to escalate onto. The
// launchability gate is shared with the placement half on purpose: a pin the backend cannot
// serve is a walled slot, and walling a slot the ladder just paid to escalate is the worst of
// both.
func TestAnUnreachableRungAboveIsNotEscalatedOnto(t *testing.T) {
	root := escRoot(t)
	t.Setenv(dispatchRungAccountsEnv, "laptop")
	before := seatDefaultFor()
	after, esc, skip := applyRungEscalation(root, true, 5416, routineTierLabels,
		[]dispatchtick.WitnessRecord{underpoweredSlot(5416, "a.log", modelroute.ZoneDevice, "qwen3.6-4b")}, before)
	if skip != escSkipNoRungAbove {
		t.Errorf("skip = %q, want %q", skip, escSkipNoRungAbove)
	}
	if esc != nil || !reflect.DeepEqual(after, before) {
		t.Errorf("an undialable rung was still bought: %+v / %+v", after, esc)
	}
	mustNotSpend(t, root)

	// And an undeclared reach refuses the same way the placement half does, rather than
	// treating "nothing declared" as "everything reachable".
	os.Unsetenv(dispatchRungAccountsEnv)
	if _, _, skip := applyRungEscalation(root, true, 5416, routineTierLabels,
		[]dispatchtick.WitnessRecord{underpoweredSlot(5416, "a.log", modelroute.ZoneDevice, "qwen3.6-4b")}, before); skip != rungSkipNoReachDecl {
		t.Errorf("skip = %q, want %q", skip, rungSkipNoReachDecl)
	}
}

// A slot with nothing to key a debit to is refused rather than charged. The alternative is
// charging the same unrecognisable attempt on every sweep, which drains a budget instead of
// bounding it — and looks in the payload exactly like an item that really did climb.
func TestAnUnidentifiableAttemptIsRefusedRatherThanRecharged(t *testing.T) {
	root := escRoot(t)
	rec := underpoweredSlot(5416, "", modelroute.ZoneDevice, "qwen3.6-4b")
	before := seatDefaultFor()
	after, esc, skip := applyRungEscalation(root, true, 5416, routineTierLabels,
		[]dispatchtick.WitnessRecord{rec}, before)
	if skip != escSkipUnidentified {
		t.Errorf("skip = %q, want %q", skip, escSkipUnidentified)
	}
	if esc != nil || !reflect.DeepEqual(after, before) {
		t.Errorf("an unidentifiable attempt still escalated: %+v / %+v", after, esc)
	}
	mustNotSpend(t, root)
}

// A ledger the reader cannot finish is not a budget. Past the read cap the actuator refuses
// rather than counting from the part it managed to read.
func TestALedgerTooLargeToCountStopsTheLadder(t *testing.T) {
	root := escRoot(t)
	if err := os.MkdirAll(filepath.Join(root, dispatchtick.RunsDirName), 0o755); err != nil {
		t.Fatal(err)
	}
	big := strings.Repeat("{\"item\":\"x\"}\n", (dispatchRungJournalCap/13)+64)
	if err := os.WriteFile(dispatchEscalationLedgerPath(root), []byte(big), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, esc, skip := applyRungEscalation(root, true, 5416, routineTierLabels,
		[]dispatchtick.WitnessRecord{underpoweredSlot(5416, "a.log", modelroute.ZoneDevice, "qwen3.6-4b")}, seatDefaultFor()); skip != escSkipLedgerBig || esc != nil {
		t.Errorf("skip = %q / esc = %+v, want %q", skip, esc, escSkipLedgerBig)
	}
}

// Nothing is bought without a graded rung to buy. An escalation onto an unmeasured candidate
// would be spending money on the strength of a capability nobody measured — the exact failure
// the placement half refuses, and worse here because this one costs.
// The vendor ceiling is what makes this the real hazard rather than a hypothetical: Place
// admits an UNMEASURED candidate at the ladder's top rung as its last-resort fallback, so
// without this refusal the one rung that costs outside money is exactly the one a fleet with
// no vendor evidence would climb onto automatically.
func TestAnUngradedRungAboveIsNotBought(t *testing.T) {
	root := rungRoot(t)
	withRungRoster(t, root)
	withRungJournal(t, root, rungTurns("qwen3.6-4b", 24, time.Minute, modelroute.VerifyWitness))
	t.Setenv(dispatchRungCeilingEnv, string(modelroute.ZoneVendor))
	t.Setenv(dispatchRungBudgetEnv, "2")

	before := seatDefaultFor()
	after, esc, skip := applyRungEscalation(root, true, 5416, routineTierLabels,
		[]dispatchtick.WitnessRecord{underpoweredSlot(5416, "a.log", modelroute.ZoneDevice, "qwen3.6-4b")}, before)
	if skip != escSkipUnmeasured {
		t.Errorf("skip = %q, want %q", skip, escSkipUnmeasured)
	}
	if esc != nil || !reflect.DeepEqual(after, before) {
		t.Errorf("an ungraded rung was bought: %+v / %+v", after, esc)
	}
	mustNotSpend(t, root)
}

// The escalated pin carries the tier's reasoning posture across the switch, the same rule the
// Layer-2 downgrade follows: a rung wall is model-scoped, not reasoning-scoped. And the model
// it lands on leaves the downgrade chain, so a later switch never re-offers the rung the item
// just paid to leave.
func TestTheEscalatedPinCarriesThePostureAndLeavesTheChain(t *testing.T) {
	root := escRoot(t)
	before := workerModelPolicy{Source: modelSourceSeatDefault, Chain: []string{"glm-5.2", "opus-5"}, Effort: "xhigh", Ultracode: true}
	after, esc, skip := applyRungEscalation(root, true, 5416, routineTierLabels,
		[]dispatchtick.WitnessRecord{underpoweredSlot(5416, "a.log", modelroute.ZoneDevice, "qwen3.6-4b")}, before)
	if skip != "" || esc == nil {
		t.Fatalf("skip = %q", skip)
	}
	if after.Effort != "xhigh" || !after.Ultracode {
		t.Errorf("posture = %q/%v, want it carried across the switch", after.Effort, after.Ultracode)
	}
	if got := strings.Join(after.Chain, ","); got != "opus-5" {
		t.Errorf("chain = %q, want the escalated model dropped from it", got)
	}
}

// The reasons are a closed vocabulary: distinct from each other, distinct from the placement
// half's, and distinct from AfterAttempt's — which are reported through the same key verbatim,
// so a collision would make "the rule declined" indistinguishable from "the actuator could
// not act".
func TestTheEscalationSkipVocabularyIsClosedAndDistinct(t *testing.T) {
	mine := []string{escSkipOutranked, escSkipNotLive, escSkipNoAttempt, escSkipUnidentified,
		escSkipBadCeiling, escSkipBadBudget, escSkipNoBudgetDecl, escSkipLedgerBig,
		escSkipLedgerBad, escSkipLedgerWrite, escSkipNoRungAbove, escSkipUnmeasured}
	theirs := []string{
		rungSkipOutranked, rungSkipNoWorkClass, rungSkipBadWindow, rungSkipNoRoster,
		rungSkipRosterEmpty, rungSkipNoJournal, rungSkipJournalBig, rungSkipJournalBad,
		rungSkipNoEvidence, rungSkipRefused, rungSkipUnmeasured, rungSkipNotApplied,
		rungSkipNoReachDecl, rungSkipUnreachable,
		modelroute.ReasonNoCeiling, modelroute.ReasonAtCeiling, modelroute.ReasonBudgetSpent,
		modelroute.ReasonAtTopRung, modelroute.ReasonEarnedByUnderpower, modelroute.ReasonRefusalNotRetried,
		modelroute.ReasonUnplacedAttempt, modelroute.ReasonTransportRetry, modelroute.ReasonWorkItemBroken,
		modelroute.ReasonFailureUnclassified, modelroute.ReasonAttemptStands, modelroute.ReasonSuccessUnverified,
	}
	seen := map[string]bool{}
	for _, r := range mine {
		if strings.TrimSpace(r) == "" {
			t.Error("an empty reason: a silent refusal is the one an operator cannot diagnose")
		}
		if seen[r] {
			t.Errorf("duplicate reason %q", r)
		}
		seen[r] = true
	}
	for _, r := range theirs {
		if seen[r] {
			t.Errorf("%q collides with a reason from another vocabulary reported through the same key", r)
		}
	}
	// And the source token is its own, so a payload never reads an automatic SPEND as the
	// cheapest-rung placement it overrode.
	for _, s := range []string{modelSourceRung, modelSourceSeatDefault, modelSourceDowngrade,
		modelSourcePlacement, modelSourceTier, modelSourceWorkClass, modelSourceExplicit} {
		if s == modelSourceEscalated {
			t.Errorf("modelSourceEscalated collides with %q", s)
		}
	}
}

// The wire, not just the rule: a live tick whose target failed underpowered on the device rung
// LAUNCHES on the fleet rung. Both halves of the ladder run here — the placement half starts
// the seat default on the cheapest graded rung, and this raises that decision — so the test
// also pins down that escalation outranks the placement it is correcting rather than being
// outranked by it.
func TestALiveTickLaunchesTheEscalatedRung(t *testing.T) {
	root := escRoot(t)
	payload := map[string]any{}
	opts := dispatchTickOptions{Backend: "claude", Live: true}
	records := []dispatchtick.WitnessRecord{underpoweredSlot(5416, "slot-a.log", modelroute.ZoneDevice, "qwen3.6-4b")}

	launch, _, _, err := prepareDispatchWorkerCommand(root, opts, dispatchLanePick{Lane: "docs"},
		dispatchtick.Account{}, 5416, 0, routineTierLabels, records, payload)
	if err != nil {
		t.Fatal(err)
	}
	if launch.Model != "glm-5.2" {
		t.Errorf("launched on %q, want the escalated fleet rung glm-5.2 (payload=%+v)", launch.Model, payload)
	}
	esc, ok := payload["rung_escalation"].(map[string]any)
	if !ok {
		t.Fatalf("the tick launched without reporting the spend: %+v", payload)
	}
	if esc["to"] != string(modelroute.ZoneFleet) || esc["model"] != "glm-5.2" {
		t.Errorf("reported %+v, want the fleet rung", esc)
	}
	if _, ok := payload["rung_escalation_skipped"]; ok {
		t.Error("the tick reported both an escalation and a refusal")
	}
	if len(escLedgerRows(t, root)) != 1 {
		t.Errorf("the launch has no recorded purchase behind it: %d rows", len(escLedgerRows(t, root)))
	}
}

// The same tick not live: the placement half still places (it spends nothing), the escalation
// half reports itself and buys nothing. A preview that wrote a debit would charge a budget for
// an attempt that never happened.
func TestAPreviewTickReportsTheRefusalAndLaunchesTheCheapRung(t *testing.T) {
	root := escRoot(t)
	payload := map[string]any{}
	records := []dispatchtick.WitnessRecord{underpoweredSlot(5416, "slot-a.log", modelroute.ZoneDevice, "qwen3.6-4b")}

	launch, _, _, err := prepareDispatchWorkerCommand(root, dispatchTickOptions{Backend: "claude"},
		dispatchLanePick{Lane: "docs"}, dispatchtick.Account{}, 5416, 0, routineTierLabels, records, payload)
	if err != nil {
		t.Fatal(err)
	}
	if payload["rung_escalation_skipped"] != escSkipNotLive {
		t.Errorf("skipped = %v, want %q", payload["rung_escalation_skipped"], escSkipNotLive)
	}
	if _, ok := payload["rung_escalation"]; ok {
		t.Error("a preview tick reported a purchase")
	}
	if launch.Model == "opus-5" {
		t.Errorf("a preview tick launched on the vendor rung: %q", launch.Model)
	}
	mustNotSpend(t, root)
}

// A default fleet tick — the seam off — carries neither key, so this whole file is invisible
// to every workspace that has not opted in.
func TestADefaultTickCarriesNoEscalationKey(t *testing.T) {
	root := escRoot(t)
	setDispatchRungPlacement(t, false)
	payload := map[string]any{}
	records := []dispatchtick.WitnessRecord{underpoweredSlot(5416, "slot-a.log", modelroute.ZoneDevice, "qwen3.6-4b")}
	if _, _, _, err := prepareDispatchWorkerCommand(root, dispatchTickOptions{Backend: "claude", Live: true},
		dispatchLanePick{Lane: "docs"}, dispatchtick.Account{}, 5416, 0, routineTierLabels, records, payload); err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"rung_escalation", "rung_escalation_skipped"} {
		if _, ok := payload[k]; ok {
			t.Errorf("an opted-out tick carries %q", k)
		}
	}
	mustNotSpend(t, root)
}

// An escalated pin is still gated on work SHAPE. The rung was picked from a corpus of past
// outcomes for a CLASS of work, which says nothing about whether that model can hold this
// issue's shape — and here the cost of getting it wrong is a budget already spent.
func TestAnEscalatedPinIsStillShapeGated(t *testing.T) {
	p := workerModelPolicy{Model: dispatchtick.WorkerModelFable, Source: modelSourceEscalated}
	gated, fired := applyPlacementGate(p, dispatchtick.ShapeChurning)
	if !fired {
		t.Fatalf("an escalated pin escaped the shape gate: %+v", gated)
	}
	if gated.Source != modelSourcePlacement || gated.Model != dispatchtick.ChurningSafeModel {
		t.Errorf("gate produced %+v, want a re-route onto a model that holds the shape", gated)
	}
}

// A rung the ledger will not record is a rung outside the budget, so it is not bought. This is
// the last refusal in the chain and the only one that fires AFTER a model has been chosen —
// everything is resolved, and the escalation is still abandoned because the purchase could not
// be written down.
func TestARungTheLedgerWillNotRecordIsNotBought(t *testing.T) {
	root := escRoot(t)
	// A ledger that READS cleanly and will not take an append: the budget says this item has
	// rungs left, the placement resolves a model, and the escalation is abandoned anyway.
	path := dispatchEscalationLedgerPath(root)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, nil, 0o444); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Chmod(path, 0o644) })
	before := seatDefaultFor("opus-5")
	after, esc, skip := applyRungEscalation(root, true, 5416, routineTierLabels,
		[]dispatchtick.WitnessRecord{underpoweredSlot(5416, "slot-a.log", modelroute.ZoneDevice, "qwen3.6-4b")}, before)
	if skip != escSkipLedgerWrite {
		t.Errorf("skip = %q, want %q", skip, escSkipLedgerWrite)
	}
	if esc != nil || !reflect.DeepEqual(after, before) {
		t.Errorf("an unrecordable rung was launched anyway: %+v / %+v", after, esc)
	}
}

// The refusal contract the tick relies on: every named refusal hands back the policy it was
// given, byte for byte. The tick adopts the escalated policy only on the no-skip branch, and
// this is the other half of that belt — a future edit that returned a half-built policy
// alongside a reason would be caught here rather than in whatever it launched.
func TestEveryEscalationRefusalReturnsThePolicyUnchanged(t *testing.T) {
	before := workerModelPolicy{Model: "seat-default", Source: modelSourceSeatDefault,
		Chain: []string{"a", "b"}, Effort: "xhigh", Ultracode: true}
	rec := underpoweredSlot(5416, "a.log", modelroute.ZoneDevice, "qwen3.6-4b")

	cases := []struct {
		name string
		mut  func(t *testing.T, root string)
		recs []dispatchtick.WitnessRecord
	}{
		{"no budget declared", func(t *testing.T, _ string) { os.Unsetenv(dispatchRungBudgetEnv) }, []dispatchtick.WitnessRecord{rec}},
		{"a bad ceiling", func(t *testing.T, _ string) { t.Setenv(dispatchRungCeilingEnv, "nope") }, []dispatchtick.WitnessRecord{rec}},
		{"a bad budget", func(t *testing.T, _ string) { t.Setenv(dispatchRungBudgetEnv, "many") }, []dispatchtick.WitnessRecord{rec}},
		{"no reach declared", func(t *testing.T, _ string) { os.Unsetenv(dispatchRungAccountsEnv) }, []dispatchtick.WitnessRecord{rec}},
		{"no work class", func(t *testing.T, _ string) {}, []dispatchtick.WitnessRecord{rec}},
		{"no finished attempt", func(t *testing.T, _ string) {}, nil},
		{"an unreadable ledger", func(t *testing.T, root string) {
			if err := os.MkdirAll(dispatchEscalationLedgerPath(root), 0o755); err != nil {
				t.Fatal(err)
			}
		}, []dispatchtick.WitnessRecord{rec}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			root := escRoot(t)
			tc.mut(t, root)
			labels := routineTierLabels
			if tc.name == "no work class" {
				labels = nil
			}
			after, esc, skip := applyRungEscalation(root, true, 5416, labels, tc.recs, before)
			if skip == "" {
				t.Fatalf("expected a refusal, got %+v", esc)
			}
			if !reflect.DeepEqual(after, before) {
				t.Errorf("refusal %q returned a changed policy: %+v", skip, after)
			}
		})
	}
}
