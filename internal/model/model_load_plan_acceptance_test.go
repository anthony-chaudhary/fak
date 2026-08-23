package model_test

import (
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelloadplan"
)

func TestQwen38LoadPlanAcceptance(t *testing.T) {
	req := modelloadplan.Request{
		Setup:       "shared",
		Goal:        "quality",
		LocalPolicy: "require",
		Memory:      "split",
		DeviceBytes: 12 << 30,
		HostBytes:   16 << 30,
		DiskBytes:   24 << 30,
	}
	plan, err := modelloadplan.Build(req)
	if err != nil {
		t.Fatal(err)
	}
	repeat, err := modelloadplan.Build(req)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(plan, repeat) {
		t.Fatal("identical goals and hardware produced different plans")
	}
	if plan.Model != modelloadplan.ModelID || plan.Selected == nil {
		t.Fatalf("model=%q selected=%#v", plan.Model, plan.Selected)
	}
	if plan.Selected.Kind != "local" || plan.Selected.HostBytes == 0 || plan.Selected.NextCommand == "" {
		t.Fatalf("selected candidate does not describe a runnable local offload: %#v", plan.Selected)
	}
	if len(plan.Provenance) == 0 {
		t.Fatal("plan omitted catalog provenance")
	}

	local, hosted, rejected := 0, 0, 0
	for _, candidate := range plan.Candidates {
		switch candidate.Kind {
		case "local":
			local++
			if !strings.Contains(candidate.URI, modelloadplan.GGUFRevision) {
				t.Fatalf("local candidate is not revision-pinned: %q", candidate.URI)
			}
		case "hosted":
			hosted++
			if !strings.Contains(candidate.URI, modelloadplan.OpenRouterID) {
				t.Fatalf("unexpected hosted identity: %q", candidate.URI)
			}
		}
		if !candidate.Fits {
			rejected++
			if len(candidate.Reasons) == 0 {
				t.Fatalf("rejected candidate %q has no explanation", candidate.ID)
			}
		}
	}
	if local != 6 || hosted != 1 || rejected == 0 {
		t.Fatalf("catalog local=%d hosted=%d rejected=%d", local, hosted, rejected)
	}
}
