package bench

// placementreservation_test.go — failure-class proof for the #4801 placement gate.
//
// The failure class this guards is a gate that SILENTLY ADMITS: a placement read that
// treats free capacity as consent, or that lets one cleared bar stand in for the rest,
// would green-light an ~805-GiB transfer and a peer eviction that no operator approved.
// The mirror-image failure is just as bad and much easier to miss — a gate hard-wired to
// refuse looks identical to a correct refusal today, and would keep refusing after a real
// reservation lands. So these tests prove BOTH directions: today's envelope refuses with
// the right typed reason, AND a fully approved envelope admits.

import (
	"encoding/json"
	"strings"
	"testing"
)

// approvedEnvelope is a hypothetical reservation that clears every bar. It exists only to
// prove the gate can say yes; nothing in the tree may set these bits without independent
// evidence.
func approvedEnvelope() ReservationEnvelope {
	return ReservationEnvelope{
		Name:                "hypothetical-approved-set",
		UsableHBMBytes:      RequiredUsableHBMBytes,
		StagingStorageBytes: RequiredStagingStorageBytes,
		WindowHours:         RequiredWindowHours,
		CollectiveRanks:     RequiredCollectiveRanks,
		OperatorApproved:    true,
		CollectiveWitnessed: true,
	}
}

// TestWitnessedPlacementRefuses is the #4801 headline: the reservation as directly
// witnessed today does NOT admit the transfer, and the refusal leads with the missing
// operator approval rather than with arithmetic.
func TestWitnessedPlacementRefuses(t *testing.T) {
	got := WitnessedPlacement()

	if got.Admissible {
		t.Fatal("witnessed placement admitted the transfer; no operator-approved reservation exists")
	}
	if got.Verdict != ReservationNoOperatorApproval {
		t.Errorf("verdict = %q, want %q (approval is the first binding constraint)",
			got.Verdict, ReservationNoOperatorApproval)
	}
	if TransferAdmissible() {
		t.Error("TransferAdmissible() = true; #4788 must stay transfer-refused")
	}
}

// TestWitnessedPlacementNamesEveryMissingResource proves the refusal is a dry-run
// placement table, not a single early return: an operator sees every unmet bar at once,
// each with a shortfall and a next action.
func TestWitnessedPlacementNamesEveryMissingResource(t *testing.T) {
	got := WitnessedPlacement()

	want := []ReservationVerdict{
		ReservationNoOperatorApproval,
		ReservationInsufficientHBM,
		ReservationInsufficientStorage,
		ReservationWindowTooShort,
		ReservationCollectiveUnwitnessed,
	}
	if len(got.Missing) != len(want) {
		t.Fatalf("Missing has %d entries, want %d: %+v", len(got.Missing), len(want), got.Missing)
	}
	for i, w := range want {
		if got.Missing[i].Verdict != w {
			t.Errorf("Missing[%d].Verdict = %q, want %q", i, got.Missing[i].Verdict, w)
		}
	}
	for _, m := range got.Missing {
		if m.Shortfall <= 0 {
			t.Errorf("%s: Shortfall = %d, want > 0 (an unmet bar has a real gap)", m.Verdict, m.Shortfall)
		}
		if strings.TrimSpace(m.NextOperator) == "" {
			t.Errorf("%s: NextOperator is empty; DoD item 7 requires a next operator action", m.Verdict)
		}
		if strings.TrimSpace(m.Why) == "" {
			t.Errorf("%s: Why is empty", m.Verdict)
		}
	}
}

// TestWitnessedHBMShortfallIsExact pins the arithmetic the whole refusal rests on: the
// non-evicting reach (480 GiB) is short of the artifact-plus-headroom bar (900 GiB) by
// 420 GiB. If a peer frees ranks and this number moves, the test should be updated with
// the new read-back — not deleted.
func TestWitnessedHBMShortfallIsExact(t *testing.T) {
	got := WitnessedPlacement()

	var hbm *MissingResource
	for i := range got.Missing {
		if got.Missing[i].Verdict == ReservationInsufficientHBM {
			hbm = &got.Missing[i]
		}
	}
	if hbm == nil {
		t.Fatal("no INSUFFICIENT_NONEVICTING_HBM entry; the 480-GiB reach cannot meet the 900-GiB bar")
	}

	const wantShortfall = (900 - 480) * reservationGiB
	if hbm.Shortfall != wantShortfall {
		t.Errorf("HBM shortfall = %d bytes, want %d", hbm.Shortfall, wantShortfall)
	}
	// The artifact alone already exceeds the witnessed reach — the refusal does not
	// depend on the headroom margin being generous.
	if ArtifactBytes <= got.ActualHBM {
		t.Errorf("artifact (%d) <= witnessed usable HBM (%d); the refusal's premise no longer holds",
			ArtifactBytes, got.ActualHBM)
	}
	if RequiredHeadroomBytes <= 0 {
		t.Errorf("RequiredHeadroomBytes = %d, want > 0 (the bar must sit above the artifact)",
			RequiredHeadroomBytes)
	}
}

// TestApprovedEnvelopeAdmits is the anti-stub proof. Without it, a gate hard-wired to
// refuse would pass every test above.
func TestApprovedEnvelopeAdmits(t *testing.T) {
	got := AdmitPlacement(approvedEnvelope())

	if !got.Admissible {
		t.Fatalf("a fully approved, fully resourced reservation was refused %q: %+v",
			got.Verdict, got.Missing)
	}
	if got.Verdict != ReservationAdmitted {
		t.Errorf("verdict = %q, want %q", got.Verdict, ReservationAdmitted)
	}
	if len(got.Missing) != 0 {
		t.Errorf("Missing = %+v, want empty", got.Missing)
	}
}

// TestEachBarBindsIndependently proves no single cleared bar can stand in for the rest:
// dropping any one input from the admitting envelope must refuse with that bar's own
// typed verdict. This is the check that would catch an `||` that should be an `&&`.
func TestEachBarBindsIndependently(t *testing.T) {
	for _, tc := range []struct {
		name string
		mut  func(*ReservationEnvelope)
		want ReservationVerdict
	}{
		{"no operator approval", func(e *ReservationEnvelope) { e.OperatorApproved = false }, ReservationNoOperatorApproval},
		{"one byte short on HBM", func(e *ReservationEnvelope) { e.UsableHBMBytes-- }, ReservationInsufficientHBM},
		{"one byte short on storage", func(e *ReservationEnvelope) { e.StagingStorageBytes-- }, ReservationInsufficientStorage},
		{"one hour short on window", func(e *ReservationEnvelope) { e.WindowHours-- }, ReservationWindowTooShort},
		{"collective not witnessed", func(e *ReservationEnvelope) { e.CollectiveWitnessed = false }, ReservationCollectiveUnwitnessed},
		{"collective spans too few ranks", func(e *ReservationEnvelope) { e.CollectiveRanks-- }, ReservationCollectiveUnwitnessed},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e := approvedEnvelope()
			tc.mut(&e)
			got := AdmitPlacement(e)

			if got.Admissible {
				t.Fatalf("%s still admitted the transfer", tc.name)
			}
			if got.Verdict != tc.want {
				t.Errorf("verdict = %q, want %q", got.Verdict, tc.want)
			}
		})
	}
}

// TestCapacityAloneIsNotConsent is the safety fence stated as a test: an envelope with
// every resource bar cleared and a passing collective STILL refuses without approval.
// Free capacity is not permission to occupy it.
func TestCapacityAloneIsNotConsent(t *testing.T) {
	e := approvedEnvelope()
	e.OperatorApproved = false

	got := AdmitPlacement(e)

	if got.Admissible {
		t.Fatal("abundant free capacity was read as consent; a reservation must be approved, never inferred")
	}
	if got.Verdict != ReservationNoOperatorApproval {
		t.Errorf("verdict = %q, want %q", got.Verdict, ReservationNoOperatorApproval)
	}
	if len(got.Missing) != 1 {
		t.Errorf("Missing = %+v, want exactly the approval entry", got.Missing)
	}
}

// TestResultCarriesAbortAndRollback covers DoD item 6: the verdict names what aborts a
// placement and how to undo a partial one.
func TestResultCarriesAbortAndRollback(t *testing.T) {
	got := WitnessedPlacement()

	if strings.TrimSpace(got.AbortThreshold) == "" {
		t.Error("AbortThreshold is empty")
	}
	if strings.TrimSpace(got.RollbackCommand) == "" {
		t.Error("RollbackCommand is empty")
	}
	if got.Schema != PlacementReservationSchema {
		t.Errorf("Schema = %q, want %q", got.Schema, PlacementReservationSchema)
	}
}

// TestResultIsScrubbedAndSerializable proves the artifact #4788 consumes carries no
// private identifier — the public/private boundary holds in the machine-readable form,
// not just in the prose around it.
func TestResultIsScrubbedAndSerializable(t *testing.T) {
	b, err := json.Marshal(WitnessedPlacement())
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	blob := strings.ToLower(string(b))

	// Node-class labels and counts are public; hosts, channels, and lab domains are not.
	for _, needle := range []string{"msl", ".lab", "slack", "http", "@", "/home/", "token"} {
		if strings.Contains(blob, needle) {
			t.Errorf("serialized verdict contains private-looking needle %q: %s", needle, blob)
		}
	}
}
