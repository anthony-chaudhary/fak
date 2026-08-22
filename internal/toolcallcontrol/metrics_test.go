package toolcallcontrol

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestResultBudgetMetricsBreaksDownAndCapturesEveryRequiredSignal(t *testing.T) {
	metrics := new(ResultBudgetMetrics)
	base := ResultBudgetMetricEvent{
		Tool:             "github.search_issues",
		Contract:         "/per_page",
		Reason:           "requested_items_above_policy_maximum",
		Mode:             "enforce",
		Policy:           ResultBudgetArtifact{Name: "thimble/default", Version: "1.0.0", SHA256: "sha256:policy-a"},
		RequestedItems:   500,
		EffectiveItems:   10,
		ActualItems:      8,
		ResponseBytes:    8192,
		ToolLatency:      120 * time.Millisecond,
		Branch:           ResultBudgetBranchCost{Latency: 30 * time.Millisecond, InputTokens: 100, OutputTokens: 20, ModelRoundTrips: 1},
		OverrideApplied:  true,
		ContinuationUsed: true,
		ImmediateRefetch: true,
	}
	if err := metrics.Record(base); err != nil {
		t.Fatal(err)
	}
	second := base
	second.RequestedItems = 40
	second.EffectiveItems = 10
	second.ActualItems = 9
	second.ResponseBytes = 4096
	second.ToolLatency = 80 * time.Millisecond
	second.Branch = ResultBudgetBranchCost{}
	second.OverrideApplied = false
	second.ContinuationUsed = false
	second.ImmediateRefetch = false
	if err := metrics.Record(second); err != nil {
		t.Fatal(err)
	}

	variations := []ResultBudgetMetricEvent{
		withResultBudgetMetricAxis(base, func(event *ResultBudgetMetricEvent) { event.Tool = "github.search_pull_requests" }),
		withResultBudgetMetricAxis(base, func(event *ResultBudgetMetricEvent) { event.Contract = "/limit" }),
		withResultBudgetMetricAxis(base, func(event *ResultBudgetMetricEvent) { event.Reason = "structured_exemption" }),
		withResultBudgetMetricAxis(base, func(event *ResultBudgetMetricEvent) { event.Mode = "observe" }),
		withResultBudgetMetricAxis(base, func(event *ResultBudgetMetricEvent) { event.Policy.SHA256 = "sha256:policy-b" }),
	}
	for _, event := range variations {
		if err := metrics.Record(event); err != nil {
			t.Fatal(err)
		}
	}

	snapshot := metrics.Snapshot()
	if snapshot.Schema != ResultBudgetMetricsSchema {
		t.Fatalf("schema=%q want=%q", snapshot.Schema, ResultBudgetMetricsSchema)
	}
	if len(snapshot.Buckets) != 6 {
		t.Fatalf("buckets=%d want=6: %+v", len(snapshot.Buckets), snapshot.Buckets)
	}
	got, ok := resultBudgetMetricBucket(snapshot, base)
	if !ok {
		t.Fatalf("missing base bucket: %+v", snapshot.Buckets)
	}
	if got.Decisions != 2 || got.RequestedItems != 540 || got.EffectiveItems != 20 || got.ActualItems != 17 {
		t.Fatalf("counts=%+v", got)
	}
	if got.ResponseBytes != 12288 || got.ToolLatencyNanos != int64(200*time.Millisecond) {
		t.Fatalf("response usage=%+v", got)
	}
	if got.Branch.LatencyNanos != int64(30*time.Millisecond) || got.Branch.InputTokens != 100 || got.Branch.OutputTokens != 20 || got.Branch.ModelRoundTrips != 1 {
		t.Fatalf("branch cost=%+v", got.Branch)
	}
	if got.Overrides != 1 || got.Continuations != 1 || got.ImmediateRefetches != 1 {
		t.Fatalf("quality signals=%+v", got)
	}

	encoded, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"requested_items", "effective_items", "actual_items", "response_bytes", "tool_latency_ns", "branch_cost", "overrides", "continuations", "immediate_refetches"} {
		if !strings.Contains(string(encoded), `"`+field+`"`) {
			t.Fatalf("published snapshot missing %q: %s", field, encoded)
		}
	}
}

func TestResultBudgetMetricsRejectsAmbiguousOrNegativeEvents(t *testing.T) {
	valid := ResultBudgetMetricEvent{
		Tool:           "github.search_issues",
		Contract:       "/per_page",
		Reason:         "pass",
		Mode:           "observe",
		Policy:         ResultBudgetArtifact{Name: "thimble/default", Version: "1.0.0", SHA256: "sha256:policy-a"},
		RequestedItems: 10,
		EffectiveItems: 10,
		ActualItems:    8,
		ResponseBytes:  1024,
	}
	tests := []struct {
		name   string
		mutate func(*ResultBudgetMetricEvent)
	}{
		{name: "tool", mutate: func(event *ResultBudgetMetricEvent) { event.Tool = "" }},
		{name: "contract", mutate: func(event *ResultBudgetMetricEvent) { event.Contract = "" }},
		{name: "reason", mutate: func(event *ResultBudgetMetricEvent) { event.Reason = "" }},
		{name: "mode", mutate: func(event *ResultBudgetMetricEvent) { event.Mode = "" }},
		{name: "policy name", mutate: func(event *ResultBudgetMetricEvent) { event.Policy.Name = "" }},
		{name: "policy version", mutate: func(event *ResultBudgetMetricEvent) { event.Policy.Version = "" }},
		{name: "policy digest", mutate: func(event *ResultBudgetMetricEvent) { event.Policy.SHA256 = "" }},
		{name: "requested items", mutate: func(event *ResultBudgetMetricEvent) { event.RequestedItems = -1 }},
		{name: "effective items", mutate: func(event *ResultBudgetMetricEvent) { event.EffectiveItems = -1 }},
		{name: "actual items", mutate: func(event *ResultBudgetMetricEvent) { event.ActualItems = -1 }},
		{name: "response bytes", mutate: func(event *ResultBudgetMetricEvent) { event.ResponseBytes = -1 }},
		{name: "tool latency", mutate: func(event *ResultBudgetMetricEvent) { event.ToolLatency = -1 }},
		{name: "branch latency", mutate: func(event *ResultBudgetMetricEvent) { event.Branch.Latency = -1 }},
		{name: "branch input", mutate: func(event *ResultBudgetMetricEvent) { event.Branch.InputTokens = -1 }},
		{name: "branch output", mutate: func(event *ResultBudgetMetricEvent) { event.Branch.OutputTokens = -1 }},
		{name: "branch round trips", mutate: func(event *ResultBudgetMetricEvent) { event.Branch.ModelRoundTrips = -1 }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			event := valid
			tc.mutate(&event)
			if err := new(ResultBudgetMetrics).Record(event); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
}

func withResultBudgetMetricAxis(base ResultBudgetMetricEvent, mutate func(*ResultBudgetMetricEvent)) ResultBudgetMetricEvent {
	mutate(&base)
	return base
}

func resultBudgetMetricBucket(snapshot ResultBudgetMetricsSnapshot, event ResultBudgetMetricEvent) (ResultBudgetMetricBucket, bool) {
	for _, bucket := range snapshot.Buckets {
		if bucket.Tool == event.Tool && bucket.Contract == event.Contract && bucket.Reason == event.Reason && bucket.Mode == event.Mode && bucket.Policy == event.Policy {
			return bucket, true
		}
	}
	return ResultBudgetMetricBucket{}, false
}
