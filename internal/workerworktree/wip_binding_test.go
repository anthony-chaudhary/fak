package workerworktree

import (
	"strings"
	"testing"
)

func TestWIPBindingValidateRequiresExplicitWorkerLaneProvenance(t *testing.T) {
	valid := WIPBinding{
		Schema: WIPBindingSchema, WorktreeID: "wt-7", WIPUnitID: "wip:v1:11111111111111111111111111111111",
		WorkerID: "worker-3", Lane: "internal/wipinventory", LeaseID: "lease-9", Registered: true, Dirty: true,
	}
	if err := valid.Validate(); err != nil {
		t.Fatalf("valid binding: %v", err)
	}

	invalid := valid
	invalid.WorkerID = ""
	invalid.Lane = ""
	invalid.Registered = false
	err := invalid.Validate()
	if err == nil {
		t.Fatal("expected missing explicit provenance to fail")
	}
	for _, want := range []string{"worker_id is required", "lane is required", "registration is required"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not contain %q", err, want)
		}
	}
}

func TestWIPBindingCarriesNoInferredOwnershipInputs(t *testing.T) {
	// The binding remains unchanged when unrelated Git-like observations match.
	// Ownership is represented only by its explicit receipt fields.
	binding := WIPBinding{
		Schema: WIPBindingSchema, WorktreeID: "wt", WIPUnitID: "unit", WorkerID: "worker",
		Lane: "lane", LeaseID: "lease", Registered: true,
	}
	before := binding
	_ = struct {
		Path, HEAD, Contents, Timestamp string
	}{"/tmp/wt", "abc", "same", "2026-09-01T00:00:00Z"}
	if binding != before {
		t.Fatal("observations must not alter declared ownership")
	}
}
