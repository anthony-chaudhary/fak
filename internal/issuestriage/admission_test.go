package issuestriage

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func testProvenance() Provenance {
	return Provenance{QueryOrRule: "label:needs-triage", Source: "github:issue/5948", Terms: "repository terms", RetrievedAt: time.Unix(1, 0).UTC(), ToolVersion: "issuestriage-test"}
}

func TestTwoSignalAdmissionStoresAbstain(t *testing.T) {
	got := TwoSignalAdmission(SignalAdmit, SignalReject, LikelyDuplicate, testProvenance())
	if got.Outcome != Abstain || got.Reason != LikelyDuplicate {
		t.Fatalf("got %+v", got)
	}
	b, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"outcome":"ABSTAIN"`) {
		t.Fatalf("ABSTAIN not stored: %s", b)
	}
}

func TestTwoSignalAdmissionRequiresCorroboration(t *testing.T) {
	cases := []struct {
		a, b Signal
		want Outcome
	}{
		{SignalAdmit, SignalAdmit, Admit}, {SignalReject, SignalReject, Reject},
		{SignalAdmit, SignalMissing, Abstain}, {SignalAdmit, SignalReject, Abstain},
	}
	for _, tc := range cases {
		if got := TwoSignalAdmission(tc.a, tc.b, NeedsKind, testProvenance()).Outcome; got != tc.want {
			t.Errorf("%s/%s = %s, want %s", tc.a, tc.b, got, tc.want)
		}
	}
}

func TestAuditAdmissionsMeasuresLegacySingleSignal(t *testing.T) {
	p := testProvenance()
	items := map[int]Admission{
		9: TwoSignalAdmission(SignalAdmit, SignalMissing, NeedsArea, p),
		3: TwoSignalAdmission(SignalAdmit, SignalAdmit, NeedsArea, p),
		7: TwoSignalAdmission(SignalAdmit, SignalReject, LikelyDuplicate, p),
	}
	got := AuditAdmissions(items)
	if got.BeforeSingleSignalAdmitted != 1 || got.AfterAdmitted != 1 || got.AfterAbstained != 2 {
		t.Fatalf("bad measurement: %+v", got)
	}
	if len(got.SingleSignalIssueNumbers) != 1 || got.SingleSignalIssueNumbers[0] != 9 {
		t.Fatalf("bad audit query: %+v", got.SingleSignalIssueNumbers)
	}
}

// This package has no network client or base URL. Keeping provenance ingestion pure
// makes reaching an external host impossible in its tests; this assertion guards the
// record itself from smuggling a live URL into the package witness.
func TestPackageWitnessUsesNoExternalHost(t *testing.T) {
	if strings.Contains(testProvenance().Source, "://") {
		t.Fatal("test provenance must use a stable source identity, not a live host")
	}
}

func TestIngestActionDoesNotTreatGardenHitAsLabel(t *testing.T) {
	got := IngestAction(Action{Number: 5948, Kind: "review", Reason: "likely-dup"}, SignalMissing, LikelyDuplicate, testProvenance())
	if got.SignalA != SignalAdmit || got.Outcome != Abstain {
		t.Fatalf("single retrieval signal must abstain: %+v", got)
	}
}
