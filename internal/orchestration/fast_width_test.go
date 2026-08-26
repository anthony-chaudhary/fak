package orchestration

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func loadFastWidthFixture(t *testing.T, name string) WidthEvidence {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	var e WidthEvidence
	if err := json.Unmarshal(raw, &e); err != nil {
		t.Fatal(err)
	}
	return e
}

func widthRequest(e WidthEvidence) *WidthRequest {
	return &WidthRequest{ObjectiveMillis: 2000, AsOf: "2026-08-25T13:00:00Z", Key: e.Key, Evidence: &e}
}

func TestFastWidthFrozenFrontiers(t *testing.T) {
	tests := []struct {
		name string
		want int
		hold WidthHold
	}{
		{"fast-width-scout.json", 4, ""},
		{"fast-width-writer.json", 1, WidthHoldNonGain},
		{"fast-width-incomplete.json", 1, WidthHoldIncomplete},
		{"fast-width-stale.json", 1, WidthHoldStale},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := loadFastWidthFixture(t, tt.name)
			got := SelectFastWidth(widthRequest(e), 8, 65536)
			if got.Selected != tt.want || got.Hold != tt.hold {
				t.Fatalf("selection=%+v", got)
			}
			if got.EvidenceDigest != e.Digest {
				t.Fatalf("digest=%q want %q", got.EvidenceDigest, e.Digest)
			}
		})
	}
}

func TestFastWidthUnequalOutcomeHolds(t *testing.T) {
	e := loadFastWidthFixture(t, "fast-width-scout.json")
	for i := range e.Cells {
		e.Cells[i].AcceptedOutcomeEqual = false
	}
	e.Digest, _ = WidthEvidenceDigest(e)
	got := SelectFastWidth(widthRequest(e), 8, 65536)
	if got.Selected != 1 || got.Hold != WidthHoldUnequalOutcome {
		t.Fatalf("unequal=%+v", got)
	}
}

func TestFastWidthRejectsTamperedDigest(t *testing.T) {
	e := loadFastWidthFixture(t, "fast-width-scout.json")
	e.Cells[0].CriticalPathMillis++
	got := SelectFastWidth(widthRequest(e), 8, 65536)
	if got.Selected != 1 || got.Hold != WidthHoldIncomparable {
		t.Fatalf("tampered=%+v", got)
	}
}

func TestFastWidthMissingAndIncomparableEvidenceHold(t *testing.T) {
	if got := SelectFastWidth(nil, 8, 65536); got.Selected != 1 || got.Hold != WidthHoldMissing {
		t.Fatalf("missing=%+v", got)
	}
	e := loadFastWidthFixture(t, "fast-width-scout.json")
	req := widthRequest(e)
	req.Key.ModelProvider = "other"
	if got := SelectFastWidth(req, 8, 65536); got.Selected != 1 || got.Hold != WidthHoldIncomparable {
		t.Fatalf("incomparable=%+v", got)
	}
}

func TestFastWidthRespectsCapsAndChoosesSmallestQualifying(t *testing.T) {
	e := loadFastWidthFixture(t, "fast-width-scout.json")
	if got := SelectFastWidth(widthRequest(e), 2, 65536); got.Selected != 1 || got.Cap != 2 {
		t.Fatalf("cap=%+v", got)
	}
	req := widthRequest(e)
	req.ObjectiveMillis = 2700
	if got := SelectFastWidth(req, 8, 65536); got.Selected != 2 {
		t.Fatalf("smallest=%+v", got)
	}
}

func TestAdaptiveWidthOperatorPinOverridesEvidenceWithinCap(t *testing.T) {
	e := loadFastWidthFixture(t, "fast-width-writer.json")
	exact, cap := 4, 8
	got, err := Resolve(OrchestrationProfile{Name: ProfileFast, MaxWorkers: &cap, ExactWorkers: &exact}, TaskSpec{Schema: "fak-orchestration-task/1", ID: "pin", WorkClass: WorkGrind, Width: widthRequest(e)}, HarnessCapabilities{})
	if err != nil {
		t.Fatal(err)
	}
	if got.Resolved.Budget.MaxWorkers != 4 || got.Resolved.Width == nil || got.Resolved.Width.Reason != "explicit operator exact-width pin" {
		t.Fatalf("resolution=%+v", got.Resolved)
	}
}

func TestAdaptiveWidthRejectsPinBeyondHardCap(t *testing.T) {
	exact, cap := 8, 4
	if _, err := Resolve(OrchestrationProfile{Name: ProfileFast, MaxWorkers: &cap, ExactWorkers: &exact}, TaskSpec{Schema: "fak-orchestration-task/1", ID: "pin"}, HarnessCapabilities{}); err == nil {
		t.Fatal("expected hard-cap error")
	}
}
