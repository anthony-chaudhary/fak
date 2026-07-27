package dispatchtick

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// fullAuthority is the most permissive escalation grant an operator can write: climb to the
// vendor rung, ten times. Every "does this escalate" assertion below runs under it, so a
// refusal to escalate is the CLASSIFICATION refusing, never a bound quietly doing it.
var fullAuthority = modelroute.EscalationBounds{Ceiling: modelroute.ZoneVendor, MaxAttempts: 10}

func TestAWitnessedGreenSlotSucceedsAndNamesNoFailure(t *testing.T) {
	got := AttemptResultFor(WitnessRecord{Claim: ClaimWitnessed, TestClaim: ClaimTestGreen})
	if !got.Succeeded {
		t.Fatalf("a diff-witnessed slot whose tests ran green graded as a failure: %+v", got)
	}
	if got.Verify != modelroute.VerifyWitness {
		t.Errorf("provenance = %q, want %q", got.Verify, modelroute.VerifyWitness)
	}
	if got.Fail != modelroute.FailNone {
		t.Errorf("a success named a failure kind %q", got.Fail)
	}
}

// The rung-earning mapping, stated as a table so the ONE input that buys a bigger model is
// visible next to everything that does not.
func TestOnlyATestThatRanAndFailedEarnsARung(t *testing.T) {
	for _, tc := range []struct {
		name string
		rec  WitnessRecord
		want modelroute.FailureKind
	}{
		{"tests ran and failed", WitnessRecord{Claim: ClaimWitnessed, TestClaim: ClaimTestRed}, modelroute.FailUnderpowered},
		{"unwitnessed diff, tests failed", WitnessRecord{Claim: ClaimUnwitnessed, TestClaim: ClaimTestRed}, modelroute.FailUnderpowered},
		{"unwitnessed diff, no test ran", WitnessRecord{Claim: ClaimUnwitnessed, TestClaim: ClaimTestUnrun}, modelroute.FailUnclassified},
		{"unwitnessed diff, tests passed", WitnessRecord{Claim: ClaimUnwitnessed, TestClaim: ClaimTestGreen}, modelroute.FailUnclassified},
		{"no claim at all", WitnessRecord{}, modelroute.FailUnclassified},
	} {
		got := AttemptResultFor(tc.rec)
		if got.Succeeded {
			t.Fatalf("%s: graded as a success: %+v", tc.name, got)
		}
		if got.Fail != tc.want {
			t.Errorf("%s: fail kind = %q, want %q", tc.name, got.Fail, tc.want)
		}
	}
}

// An unwitnessed commit is the most tempting thing to call underpowered, and the reason it
// is not is that `dos commit-audit` also abstains on a vague subject: the bucket holds "the
// model could not do the work" and "the model described the work badly" together.
func TestAnUnwitnessedCommitDoesNotBuyABiggerModel(t *testing.T) {
	rec := WitnessRecord{Claim: ClaimUnwitnessed, TestClaim: ClaimTestUnrun}
	v := modelroute.AfterAttempt(placedOn(modelroute.ZoneDevice), AttemptResultFor(rec), fullAuthority, 0)
	if v.Escalates() {
		t.Fatalf("an unwitnessed commit escalated to %q under full authority (%s)", v.To, v.Reason)
	}
}

// Ordering is the contract: the refusal is read before any field that could earn a rung, so
// a guard refusal cannot be laundered into an escalation by whatever else the record holds.
func TestAGuardRefusalStaysARefusalWhateverElseTheRecordCarries(t *testing.T) {
	for _, reason := range []string{NoCommitSelfModify, NoCommitPolicyBlock, NoCommitOffTrunk} {
		for _, test := range []string{"", ClaimTestUnrun, ClaimTestRed, ClaimTestGreen} {
			rec := WitnessRecord{Claim: ClaimNoCommit, Reason: reason, TestClaim: test}
			got := AttemptResultFor(rec)
			if got.Fail != modelroute.FailRefused {
				t.Errorf("reason %q with test claim %q -> %q, want %q", reason, test, got.Fail, modelroute.FailRefused)
			}
			v := modelroute.AfterAttempt(placedOn(modelroute.ZoneDevice), got, fullAuthority, 0)
			if v.Escalates() {
				t.Errorf("reason %q with test claim %q escalated to %q under full authority", reason, test, v.To)
			}
		}
	}
}

// A capacity wall is a scheduling fact, not a capability one. Filing it as underpowered
// would turn every weekly-bucket rollover into vendor spend, precisely when the fleet is
// busiest.
func TestAWallThatNoModelEverSawIsTransportNotCapability(t *testing.T) {
	for _, reason := range []string{NoCommitAuthWall, NoCommitUsageCap, NoCommitModelUnknown, NoCommitRateLimit, NoCommitBannerNoop} {
		got := AttemptResultFor(WitnessRecord{Claim: ClaimNoCommit, Reason: reason})
		if got.Fail != modelroute.FailTransport {
			t.Errorf("reason %q -> %q, want %q", reason, got.Fail, modelroute.FailTransport)
		}
		v := modelroute.AfterAttempt(placedOn(modelroute.ZoneDevice), got, fullAuthority, 0)
		if v.Action != modelroute.ActionRetrySameRung {
			t.Errorf("reason %q -> %q/%q, want a retry on the same rung", reason, v.Action, v.Reason)
		}
	}
}

// A reason string this build does not recognise must stop the ladder rather than fall
// through to the branch that spends. A newer producer adding a reason is a routine event;
// paying for it is not.
func TestAnUnrecognisedReasonStopsRatherThanEarningARung(t *testing.T) {
	for _, reason := range []string{NoCommitUnknown, "quota_reshuffle_2027", "CLAIM_TEST_RED"} {
		// Paired with a red test on purpose: the record carries the one signal that DOES
		// earn a rung, and the unrecognised reason must still take precedence over it.
		got := AttemptResultFor(WitnessRecord{Claim: ClaimNoCommit, Reason: reason, TestClaim: ClaimTestRed})
		if got.Fail != modelroute.FailUnclassified {
			t.Errorf("reason %q -> %q, want %q", reason, got.Fail, modelroute.FailUnclassified)
		}
	}
	// The other side of the same rule: a blank reason is an ABSENT one, not an unrecognised
	// one, so it must fall through to the evidence beside it rather than mask it.
	blank := AttemptResultFor(WitnessRecord{Claim: ClaimUnwitnessed, Reason: "   ", TestClaim: ClaimTestRed})
	if blank.Fail != modelroute.FailUnderpowered {
		t.Errorf("a blank reason beside a red test -> %q, want %q", blank.Fail, modelroute.FailUnderpowered)
	}
	unknown := AttemptResultFor(WitnessRecord{Claim: ClaimNoCommit, Reason: NoCommitUnknown})
	if unknown.Fail != modelroute.FailUnclassified {
		t.Errorf("the unknown reason -> %q, want %q", unknown.Fail, modelroute.FailUnclassified)
	}
}

// The anti-divergence witness. Capability grading and escalation must read the same slot the
// same way; two graders would let a fleet bank a model's result in one ledger while buying a
// bigger model over it in the other.
func TestTheEscalationInputAgreesWithTheCapabilityFoldOnEverySlot(t *testing.T) {
	var records []WitnessRecord
	for _, claim := range []string{ClaimWitnessed, ClaimUnwitnessed, ClaimNoCommit, ""} {
		for _, test := range []string{"", ClaimTestUnrun, ClaimTestRed, ClaimTestGreen} {
			records = append(records, WitnessRecord{Issue: 1, Log: claim + test, Model: "m", Claim: claim, TestClaim: test})
		}
	}
	rows, stats := TurnOutcomesFromWitness(records, WitnessEvidenceOptions{
		Class: func(WitnessRecord) modelroute.WorkClass { return modelroute.ClassRoutine },
	})
	if stats.Produced != len(records) {
		t.Fatalf("fold produced %d rows for %d records", stats.Produced, len(records))
	}
	for i, row := range rows {
		got := AttemptResultFor(records[i])
		if got.Succeeded != row.Success {
			t.Errorf("%+v: escalation input says succeeded=%v, capability fold says %v", records[i], got.Succeeded, row.Success)
		}
		if got.Verify != row.Verify {
			t.Errorf("%+v: provenance %q vs the fold's %q", records[i], got.Verify, row.Verify)
		}
	}
}

// Totality: every record produces a usable result, and success and failure are never both
// reported. An unclassifiable slot must be a NAMED failure — reporting no failure at all
// would read as a success and accept work nobody graded.
func TestEverySlotGradesToExactlyOneOfSuccessOrANamedFailure(t *testing.T) {
	reasons := []string{"", NoCommitSelfModify, NoCommitPolicyBlock, NoCommitOffTrunk, NoCommitAuthWall,
		NoCommitUsageCap, NoCommitModelUnknown, NoCommitRateLimit, NoCommitBannerNoop, NoCommitUnknown, "novel"}
	seen := map[modelroute.FailureKind]bool{}
	for _, claim := range []string{ClaimWitnessed, ClaimUnwitnessed, ClaimNoCommit, "", "CLAIM_FUTURE"} {
		for _, test := range []string{"", ClaimTestUnrun, ClaimTestRed, ClaimTestGreen, "CLAIM_TEST_FUTURE"} {
			for _, reason := range reasons {
				rec := WitnessRecord{Claim: claim, TestClaim: test, Reason: reason}
				got := AttemptResultFor(rec)
				switch {
				case got.Succeeded && got.Fail != modelroute.FailNone:
					t.Fatalf("%+v: reported a success AND failure kind %q", rec, got.Fail)
				case !got.Succeeded && got.Fail == modelroute.FailNone:
					t.Fatalf("%+v: reported a failure with no kind", rec)
				}
				if got.Succeeded && !got.Verify.Trusted() {
					t.Fatalf("%+v: a success carrying untrusted provenance %q", rec, got.Verify)
				}
				seen[got.Fail] = true
			}
		}
	}
	for _, want := range []modelroute.FailureKind{
		modelroute.FailNone, modelroute.FailRefused, modelroute.FailTransport,
		modelroute.FailUnderpowered, modelroute.FailUnclassified,
	} {
		if !seen[want] {
			t.Errorf("the sweep never produced fail kind %q", want)
		}
	}
	// FailWorkItem is expressible and deliberately never claimed: nothing in a witness
	// record separates work no rung can finish from a model that could not finish it.
	if seen[modelroute.FailWorkItem] {
		t.Errorf("classified a slot as %q, which no witness record can establish", modelroute.FailWorkItem)
	}
}

func placedOn(z modelroute.PlacementZone) modelroute.Placement {
	return modelroute.Placement{Model: "m", Zone: z}
}
