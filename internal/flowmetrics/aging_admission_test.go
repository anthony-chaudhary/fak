package flowmetrics

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipreadiness"
)

func readyReceipt(now time.Time, work ...wipreadiness.Work) *wipreadiness.Receipt {
	return &wipreadiness.Receipt{ObservedAt: now.Add(-time.Minute), ExpiresAt: now.Add(time.Minute), Verdict: wipreadiness.VerdictCurrent, Work: work}
}

func TestAdmitAgingWIPBlocksOldestActionableFreshStart(t *testing.T) {
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)
	receipt := AdmitAgingWIP(AgingAdmissionRequest{
		Intent: "fresh", Now: now, Budget: 7 * 24 * time.Hour, Readiness: readyReceipt(now),
		Units: []AgingUnit{
			{Unit: "/private/repo/issue/42", AgeDays: 12, Classification: AgingActionable},
			{Unit: "#7", AgeDays: 20, Classification: AgingLiveOwned},
			{Unit: "#9", AgeDays: 18, Classification: AgingRetained},
		},
	})
	if receipt.Verdict != "REFUSE" || receipt.ReasonCode != AgingWIPReasonCode {
		t.Fatalf("receipt = %#v", receipt)
	}
	if receipt.BlockingUnit == nil || receipt.BlockingUnit.Unit != "unidentified-unit" || receipt.BlockingUnit.AgeDays != 12 || receipt.BlockingUnit.Classification != AgingActionable {
		t.Fatalf("blocking unit = %#v", receipt.BlockingUnit)
	}
	if got, want := strings.Join(receipt.SafeActions, ","), "landing,recovery,parking,owned-continuation,safety,witnessed-supersession"; got != want {
		t.Fatalf("safe actions = %q, want %q", got, want)
	}
	b, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), "/private/") {
		t.Fatalf("receipt leaked path: %s", b)
	}
}

func TestAdmitAgingWIPExemptionsAndReadiness(t *testing.T) {
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)
	old := []AgingUnit{{Unit: "#42", AgeDays: 30, Classification: AgingActionable}}
	for _, intent := range []string{"landing", "recovery", "parking", "continuation", "owned-continuation", "safety"} {
		t.Run(intent, func(t *testing.T) {
			got := AdmitAgingWIP(AgingAdmissionRequest{Intent: intent, Now: now, Budget: 7 * 24 * time.Hour, Units: old})
			if got.Verdict != "ADMIT" {
				t.Fatalf("%s = %#v", intent, got)
			}
		})
	}
	for _, tc := range []AgingAdmissionRequest{
		{Intent: "supersession", SupersessionReason: "#42 replaced by witnessed issue #99", Now: now, Budget: 7 * 24 * time.Hour, Units: old},
		{Intent: "fresh", OverrideReason: "incident commander approved emergency intake", Now: now, Budget: 7 * 24 * time.Hour, Units: old},
	} {
		if got := AdmitAgingWIP(tc); got.Verdict != "ADMIT" || got.WitnessedReason == "" {
			t.Fatalf("witnessed action = %#v", got)
		}
	}
	if got := AdmitAgingWIP(AgingAdmissionRequest{Intent: "supersession", Now: now, Budget: 7 * 24 * time.Hour, Units: old}); got.Verdict != "REFUSE" {
		t.Fatalf("unwitnessed supersession = %#v", got)
	}
	stale := readyReceipt(now)
	stale.ExpiresAt = now.Add(-time.Second)
	if got := AdmitAgingWIP(AgingAdmissionRequest{Intent: "fresh", Now: now, Budget: 7 * 24 * time.Hour, Readiness: stale, Units: old}); got.Verdict != "REFUSE" || !strings.Contains(got.Reason, "current ready") {
		t.Fatalf("stale readiness = %#v", got)
	}
	if got := AdmitAgingWIP(AgingAdmissionRequest{Intent: "recovery", Now: now, Budget: 7 * 24 * time.Hour, Readiness: stale, Units: old}); got.Verdict != "ADMIT" {
		t.Fatalf("recovery must not require readiness: %#v", got)
	}
}

func TestBuildAgingUnitsReusesInventoryAndReadinessOwnership(t *testing.T) {
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)
	opened := now.Add(-20 * 24 * time.Hour)
	started := now.Add(-15 * 24 * time.Hour)
	issues := []Issue{
		{Number: 1, Title: "owned", CreatedAt: opened},
		{Number: 2, Title: "retained", CreatedAt: opened, Labels: []string{"wip/retained"}},
		{Number: 3, Title: "actionable", CreatedAt: opened},
		{Number: 4, Title: "unstarted", CreatedAt: opened},
	}
	commits := []Commit{
		{SHA: "a", When: started, Issues: []int{1}},
		{SHA: "b", When: started.Add(time.Hour), Issues: []int{2}},
		{SHA: "c", When: started.Add(2 * time.Hour), Issues: []int{3}},
	}
	units := BuildAgingUnits(issues, commits, readyReceipt(now, wipreadiness.Work{ID: "issue:1", Dirty: true, Ownership: wipreadiness.OwnershipRemote}), now)
	if len(units) != 3 {
		t.Fatalf("units = %#v", units)
	}
	classes := map[string]string{}
	for _, unit := range units {
		classes[unit.Unit] = unit.Classification
	}
	if classes["#1"] != AgingLiveOwned || classes["#2"] != AgingRetained || classes["#3"] != AgingActionable {
		t.Fatalf("classes = %#v", classes)
	}
}
