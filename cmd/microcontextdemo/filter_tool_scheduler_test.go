package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func schedulerFixture(t *testing.T) []schedulerRecord {
	t.Helper()
	root := filepath.Join("..", "..", "experiments", "microcontext", "s8m-semantic-tool-fold-2026-08-10.json")
	rs, _, _, err := loadSchedulerRecords(root)
	if err != nil {
		t.Fatal(err)
	}
	return rs
}

func TestTaskSpecificSufficiencyPredicatesAreSound(t *testing.T) {
	rs := schedulerFixture(t)
	for _, task := range []string{"existence", "top-k", "exhaustive"} {
		done := map[int]bool{}
		for _, i := range orderRecords(task, "planner", rs) {
			done[i] = true
			if taskSufficient(task, rs, done) {
				break
			}
		}
		if taskQuality(task, rs, done) != 1 {
			t.Fatalf("%s predicate changed oracle answer", task)
		}
		if task == "exhaustive" && len(done) != len(rs) {
			t.Fatalf("exhaustive stopped at %d/%d", len(done), len(rs))
		}
	}
}

func TestTimeoutAndSufficiencyReleaseSlots(t *testing.T) {
	rs := schedulerFixture(t)
	r := simulateScheduler("top-k", "run-all", rs, 2, 30, 10)
	seenTimeout := false
	for _, x := range r.receipts {
		if x.Status == "timed_out" && x.Reason == "deadline_released_slot" {
			seenTimeout = true
		}
	}
	if !seenTimeout {
		t.Fatal("missing timeout slot-release receipt")
	}
	r = simulateScheduler("existence", "adaptive", rs, 4, 900, 35)
	seenCancel := false
	for _, x := range r.receipts {
		if x.Status == "cancelled" && x.Reason == "witnessed_sufficiency_slot_released" {
			seenCancel = true
		}
	}
	if !seenCancel {
		t.Fatal("missing sufficiency slot-release receipt")
	}
}

func TestEffectStageCannotEnterSchedulerCatalog(t *testing.T) {
	rs := schedulerFixture(t)
	for _, policy := range []string{"run-all", "fixed-cascade", "planner", "adaptive", "adaptive-selective-hedge", "adaptive-universal-hedge"} {
		for _, r := range rs {
			stages, _ := stagesFor(policy, r)
			for _, s := range stages {
				if s == "effect" || s == "write-tool" {
					t.Fatalf("%s admitted effect stage", policy)
				}
			}
		}
	}
}

func TestFilterToolSchedulerArtifactReplays(t *testing.T) {
	root := filepath.Join("..", "..", "experiments", "microcontext")
	a := filepath.Join(t.TempDir(), "a.json")
	b := filepath.Join(t.TempDir(), "b.json")
	fold := filepath.Join(root, "s8m-semantic-tool-fold-2026-08-10.json")
	if err := runFilterToolScheduler(fold, a, 3); err != nil {
		t.Fatal(err)
	}
	if err := runFilterToolScheduler(fold, b, 3); err != nil {
		t.Fatal(err)
	}
	x, _ := os.ReadFile(a)
	y, _ := os.ReadFile(b)
	if string(x) != string(y) {
		t.Fatal("scheduler artifact is not deterministic")
	}
	if err := verifyFilterToolScheduler(a); err != nil {
		t.Fatal(err)
	}
	var rep filterToolSchedulerReport
	if err := json.Unmarshal(x, &rep); err != nil {
		t.Fatal(err)
	}
	if len(rep.Results) != 18 || len(rep.Receipts) == 0 {
		t.Fatal("matrix or receipts incomplete")
	}
}
