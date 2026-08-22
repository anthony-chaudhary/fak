package orchestration

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
)

func TestWorkflowPlanNodeAccessContract(t *testing.T) {
	plan := WorkflowPlan{
		Schema: SchemaVersion,
		TaskID: "task-access",
		Roles: []Role{
			{ID: "observer", Purpose: "inspect", TaskID: "task-access", Access: ChildAccess{Mode: ChildAccessObserve, ReadSet: []string{"docs/z", `docs\\a`, "docs/a"}}},
			{ID: "worker", Purpose: "edit", TaskID: "task-access", Access: ChildAccess{Mode: ChildAccessEffect, ReadSet: []string{"internal/b"}, WriteSet: []string{"internal/z", `internal\\a`, "internal/a"}}},
		},
	}

	if err := NormalizeWorkflowPlanAccess(&plan); err != nil {
		t.Fatalf("NormalizeWorkflowPlanAccess: %v", err)
	}
	wantReads := []string{"docs/a", "docs/z"}
	wantWrites := []string{"internal/a", "internal/z"}
	if !reflect.DeepEqual(plan.Roles[0].Access.ReadSet, wantReads) {
		t.Fatalf("observer read_set = %#v, want %#v", plan.Roles[0].Access.ReadSet, wantReads)
	}
	if !reflect.DeepEqual(plan.Roles[1].Access.WriteSet, wantWrites) {
		t.Fatalf("worker write_set = %#v, want %#v", plan.Roles[1].Access.WriteSet, wantWrites)
	}

	first, err := json.Marshal(plan)
	if err != nil {
		t.Fatal(err)
	}
	var roundTripped WorkflowPlan
	if err := json.Unmarshal(first, &roundTripped); err != nil {
		t.Fatal(err)
	}
	if err := NormalizeWorkflowPlanAccess(&roundTripped); err != nil {
		t.Fatalf("normalize round-trip: %v", err)
	}
	second, err := json.Marshal(roundTripped)
	if err != nil {
		t.Fatal(err)
	}
	if string(first) != string(second) {
		t.Fatalf("workflow plan is not byte-stable:\nfirst:  %s\nsecond: %s", first, second)
	}

	for _, tc := range []struct {
		name   string
		access ChildAccess
		want   error
	}{
		{name: "unknown mode", access: ChildAccess{Mode: "maybe", ReadSet: []string{"docs"}}, want: ErrUnknownChildAccessMode},
		{name: "observer write set", access: ChildAccess{Mode: ChildAccessObserve, WriteSet: []string{"internal"}}, want: ErrObserverWriteSet},
		{name: "effect unknown write scope", access: ChildAccess{Mode: ChildAccessEffect}, want: ErrEffectWriteSetRequired},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bad := WorkflowPlan{Roles: []Role{{ID: "node", Access: tc.access}}}
			err := NormalizeWorkflowPlanAccess(&bad)
			if !errors.Is(err, tc.want) {
				t.Fatalf("error = %v, want %v", err, tc.want)
			}
			if err == nil || !strings.Contains(err.Error(), `role "node"`) {
				t.Fatalf("error does not identify role: %v", err)
			}
		})
	}
}
