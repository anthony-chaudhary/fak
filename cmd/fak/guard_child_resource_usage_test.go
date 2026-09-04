package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

func TestGuardChildResourceUsageLedgerRecordsInvocationAndFolds(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "child-resource-usage.jsonl")
	t.Setenv("FAK_CHILD_RESOURCE_USAGE_PATH", path)

	stop := make(chan struct{})
	defer close(stop)
	policy := guardResourcePolicy{
		PollInterval: 50 * time.Millisecond,
		Metric:       procguard.MemoryMetricRSS,
		MaxTreeBytes: 500 << 20,
		Stop:         stop,
	}

	traceID := "trace-usage-ledger-test"
	agent := "codex"
	rootPID := 9876

	ch := startGuardChildResourceMonitor(rootPID, traceID, agent, policy)
	if ch == nil {
		t.Fatal("expected non-nil channel from startGuardChildResourceMonitor")
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read usage ledger: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 1 {
		t.Fatalf("expected 1 line in usage ledger, got %d:\n%s", len(lines), string(data))
	}

	var row guardResourceReceipt
	if err := json.Unmarshal([]byte(lines[0]), &row); err != nil {
		t.Fatalf("failed to unmarshal usage receipt: %v", err)
	}

	if row.Schema != "fak.guard.child-resource.v1" {
		t.Errorf("schema = %q, want %q", row.Schema, "fak.guard.child-resource.v1")
	}
	if row.TraceID != traceID {
		t.Errorf("traceID = %q, want %q", row.TraceID, traceID)
	}
	if row.Agent != agent {
		t.Errorf("agent = %q, want %q", row.Agent, agent)
	}
	if row.RootPID != rootPID {
		t.Errorf("rootPID = %d, want %d", row.RootPID, rootPID)
	}
	if row.MemoryMetric != string(procguard.MemoryMetricRSS) {
		t.Errorf("metric = %q, want %q", row.MemoryMetric, string(procguard.MemoryMetricRSS))
	}
	if row.Action != "containment_active" {
		t.Errorf("action = %q, want %q", row.Action, "containment_active")
	}
	if row.Reason != "CHILD_RESOURCE_CONTAINMENT_ACTIVE" {
		t.Errorf("reason = %q, want %q", row.Reason, "CHILD_RESOURCE_CONTAINMENT_ACTIVE")
	}
	if row.ThresholdBytes != 500<<20 {
		t.Errorf("threshold = %d, want %d", row.ThresholdBytes, 500<<20)
	}
	if !row.DescendantsSurvive {
		t.Errorf("descendantsSurvive = false, want true")
	}

	// Verify fold with explicit path
	counts, err := foldChildResourceReceiptsByWeek(path)
	if err != nil {
		t.Fatalf("foldChildResourceReceiptsByWeek error: %v", err)
	}
	now := time.Now().UTC()
	year, week := now.ISOWeek()
	expectedWeekKey := fmt.Sprintf("%04d-W%02d", year, week)
	if counts[expectedWeekKey] != 1 {
		t.Errorf("counts[%s] = %d, want 1", expectedWeekKey, counts[expectedWeekKey])
	}

	// Verify fold with default path (empty string)
	defaultCounts, err := foldChildResourceReceiptsByWeek("")
	if err != nil {
		t.Fatalf("foldChildResourceReceiptsByWeek default error: %v", err)
	}
	if defaultCounts[expectedWeekKey] != 1 {
		t.Errorf("defaultCounts[%s] = %d, want 1", expectedWeekKey, defaultCounts[expectedWeekKey])
	}
}

func TestGuardChildResourceUsageSurfaceRendersJSONAndText(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "child-resource-usage.jsonl")
	t.Setenv("FAK_CHILD_RESOURCE_USAGE_PATH", path)

	rows := []guardResourceReceipt{
		{Schema: "fak.guard.child-resource.v1", At: "2026-08-10T12:00:00Z", TraceID: "t1", Action: "containment_active"},
		{Schema: "fak.guard.child-resource.v1", At: "2026-08-11T14:00:00Z", TraceID: "t2", Action: "containment_active"},
		{Schema: "fak.guard.child-resource.v1", At: "2026-08-25T09:00:00Z", TraceID: "t3", Action: "containment_active"},
	}
	for _, r := range rows {
		if err := appendGuardResourceReceipt(path, r); err != nil {
			t.Fatalf("append receipt: %v", err)
		}
	}

	out := captureStdout(t, func() { cmdUsage([]string{"--child-resource", "--json"}) })
	var got guardChildResourceUsageSummary
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("failed to decode JSON usage: %v\noutput:\n%s", err, out)
	}

	if got.Schema != "fak-guard-child-resource-usage-summary/1" {
		t.Fatalf("schema = %q, want fak-guard-child-resource-usage-summary/1", got.Schema)
	}
	if len(got.Weeks) != 2 {
		t.Fatalf("len(weeks) = %d, want 2: %+v", len(got.Weeks), got.Weeks)
	}
	if got.Weeks[0].Week != "2026-W33" || got.Weeks[0].Invocations != 2 {
		t.Errorf("weeks[0] = %+v, want 2026-W33 invocations=2", got.Weeks[0])
	}
	if got.Weeks[1].Week != "2026-W35" || got.Weeks[1].Invocations != 1 {
		t.Errorf("weeks[1] = %+v, want 2026-W35 invocations=1", got.Weeks[1])
	}

	// Plain text readout
	textOut := captureStdout(t, func() { cmdUsage([]string{"--child-resource"}) })
	if !strings.Contains(textOut, "2026-W33 invocations=2") {
		t.Errorf("text output missing 2026-W33 invocations=2: %s", textOut)
	}
	if !strings.Contains(textOut, "2026-W35 invocations=1") {
		t.Errorf("text output missing 2026-W35 invocations=1: %s", textOut)
	}
}

func TestGuardChildResourceDogfoodAdoption(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "child-resource-usage.jsonl")
	t.Setenv("FAK_CHILD_RESOURCE_USAGE_PATH", path)

	// Simulate configured default guard policy
	policy := guardResourcePolicyConfigured()
	stop := make(chan struct{})
	defer close(stop)
	policy.Stop = stop

	startGuardChildResourceMonitor(4321, "session-mac-adoption-1", "claude", policy)
	startGuardChildResourceMonitor(4322, "session-mac-adoption-2", "claude", policy)

	counts, err := foldChildResourceReceiptsByWeek(path)
	if err != nil {
		t.Fatalf("foldChildResourceReceiptsByWeek: %v", err)
	}
	year, week := time.Now().UTC().ISOWeek()
	currentWeek := fmt.Sprintf("%04d-W%02d", year, week)
	if counts[currentWeek] != 2 {
		t.Errorf("adoption count for %s = %d, want 2", currentWeek, counts[currentWeek])
	}

	// Verify the recorded entries have the expected policy metric and thresholds
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read usage file: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 lines in usage file, got %d", len(lines))
	}
	for i, line := range lines {
		var r guardResourceReceipt
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			t.Fatalf("decode line %d: %v", i, err)
		}
		if r.MemoryMetric != string(policy.Metric) {
			t.Errorf("line %d: metric = %q, want %q", i, r.MemoryMetric, string(policy.Metric))
		}
		if r.ThresholdBytes != policy.MaxTreeBytes {
			t.Errorf("line %d: threshold = %d, want %d", i, r.ThresholdBytes, policy.MaxTreeBytes)
		}
		if r.Action != "containment_active" {
			t.Errorf("line %d: action = %q, want containment_active", i, r.Action)
		}
	}
}
