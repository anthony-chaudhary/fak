package workaccount

import (
	"strings"
	"testing"
)

func TestRegistryIsCompleteAndValid(t *testing.T) {
	rows := Registry()
	if err := Validate(rows); err != nil {
		t.Fatal(err)
	}
	want := []string{"provider_prompt_cache", "response_vdso_memo", "inline_tool_serving", "context_compaction", "kv_prefix_reuse", "context_elision", "schema_tool_filtering", "cold_tool_defer", "model_routing", "safety_intervention"}
	seen := map[string]bool{}
	for _, row := range rows {
		seen[row.ID] = true
	}
	for _, id := range want {
		if !seen[id] {
			t.Errorf("missing declared mechanism %s", id)
		}
	}
}

func TestValidateRejectsNewMeasurableMechanismWithoutMapping(t *testing.T) {
	rows := append(Registry(), Mechanism{ID: "new_reuse", Label: "new reuse", Producer: "internal/example", Status: Accounted, Units: []string{"input_tokens"}})
	if err := Validate(rows); err == nil || !strings.Contains(err.Error(), "source_id") {
		t.Fatalf("error=%v", err)
	}
}

func TestValidateRequiresExplicitReasonForUnavailableOrExcluded(t *testing.T) {
	for _, status := range []Status{Unavailable, Excluded} {
		if err := Validate([]Mechanism{{ID: "x", Label: "x", Producer: "p", Status: status}}); err == nil || !strings.Contains(err.Error(), "requires reason") {
			t.Fatalf("status=%s error=%v", status, err)
		}
	}
}

func TestBuildReportIsDeterministic(t *testing.T) {
	r, err := BuildReport(Registry())
	if err != nil {
		t.Fatal(err)
	}
	if r.Schema != "fak.info.work-accounting-coverage/1" || len(r.Mechanisms) != 10 {
		t.Fatalf("report=%#v", r)
	}
	for i := 1; i < len(r.Mechanisms); i++ {
		if r.Mechanisms[i-1].ID > r.Mechanisms[i].ID {
			t.Fatalf("not sorted: %#v", r.Mechanisms)
		}
	}
	if r.Counts[Accounted] != 6 || r.Counts[Unavailable] != 2 || r.Counts[Overlapping] != 1 || r.Counts[Excluded] != 1 {
		t.Fatalf("counts=%#v", r.Counts)
	}
}
