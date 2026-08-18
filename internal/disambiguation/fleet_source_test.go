package disambiguation

import (
	"errors"
	"testing"
)

func TestResolveDispatchIdentityNeverParsesNarration(t *testing.T) {
	input := DispatchIdentityInput{Narration: "worker=w issue=99 lane=cmd lease=x"}
	if _, err := ResolveDispatchIdentity(input); !errors.Is(err, ErrDispatchIdentityMissing) {
		t.Fatalf("error=%v", err)
	}
}

func TestResolveDispatchIdentityUsesOnlyStructuredFields(t *testing.T) {
	input := DispatchIdentityInput{WorkerID: "w1", Issue: "6317", Lane: "dispatch", LeaseID: "l1", Narration: "worker=evil issue=1 lane=root lease=other"}
	got, err := ResolveDispatchIdentity(input)
	if err != nil {
		t.Fatal(err)
	}
	if got.WorkerID != "w1" || got.Issue != "6317" || got.Lane != "dispatch" || got.LeaseID != "l1" {
		t.Fatalf("identity=%#v", got)
	}
}

func TestRunFleetSourceSelfTest(t *testing.T) {
	report, err := RunFleetSourceSelfTest()
	if err != nil {
		t.Fatal(err)
	}
	if report.Schema != FleetSourceSelfTestSchemaVersion || len(report.Resolutions) != 8 || !report.NarrationRejected || !report.StructuredAccepted {
		t.Fatalf("report=%#v", report)
	}
	owners := map[string]bool{}
	for _, resolution := range report.Resolutions {
		if resolution.SourcePath == "" {
			t.Errorf("missing source=%#v", resolution)
		}
		owners[resolution.OwnerLeaf] = true
	}
	if len(owners) < 7 {
		t.Fatalf("owners=%v", owners)
	}
}
