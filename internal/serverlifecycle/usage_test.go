package serverlifecycle

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestLifecycleInvocationAppendsPrivacySafeUsage(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "private-model.gguf")
	executable := filepath.Join(dir, "private-server.exe")
	result, err := Init(context.Background(), InitOptions{
		InstanceDirectory: dir,
		ServerName:        "private-hostname",
		ModelPath:         model,
		ArtifactSHA256:    strings.Repeat("a", 64),
		AdapterExecutable: executable,
	})
	if err != nil {
		t.Fatalf("Init: %v", err)
	}
	if result.State != StateConfigured {
		t.Fatalf("state = %q, want %q", result.State, StateConfigured)
	}

	data, err := os.ReadFile(filepath.Join(dir, UsageLedgerFilename))
	if err != nil {
		t.Fatalf("read usage ledger: %v", err)
	}
	for _, secret := range []string{dir, model, executable, "private-hostname"} {
		if bytes.Contains(data, []byte(secret)) {
			t.Fatalf("usage row leaked private value %q: %s", secret, data)
		}
	}
	var row UsageRow
	if err := json.Unmarshal(bytes.TrimSpace(data), &row); err != nil {
		t.Fatalf("decode row: %v", err)
	}
	if row.Schema != UsageLedgerSchema || row.Operation != "init" || row.Outcome != "ok" || row.State != StateConfigured {
		t.Fatalf("unexpected usage row: %+v", row)
	}
	folds, err := FoldUsage(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("fold usage: %v", err)
	}
	t.Logf("ledger=%s", bytes.TrimSpace(data))
	t.Logf("weekly_fold=%+v", folds)
}

func TestUsageLedgerConcurrentAppendAndWeeklyFold(t *testing.T) {
	dir := t.TempDir()
	const invocations = 24
	var wg sync.WaitGroup
	for i := 0; i < invocations; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			result := Result{State: StateReady}
			var err error
			if i%3 == 0 {
				err = &RefusalError{Reason: ReasonInstanceLocked, Detail: "bounded detail"}
				result.Reason = ReasonInstanceLocked
			}
			if appendErr := recordInvocation(dir, "status", result, err); appendErr != nil {
				t.Errorf("record invocation: %v", appendErr)
			}
		}(i)
	}
	wg.Wait()

	file, err := os.Open(filepath.Join(dir, UsageLedgerFilename))
	if err != nil {
		t.Fatalf("open ledger: %v", err)
	}
	defer file.Close()
	folds, err := FoldUsage(file)
	if err != nil {
		t.Fatalf("FoldUsage: %v", err)
	}
	if len(folds) != 1 || folds[0].Total != invocations || folds[0].Operations["status"] != invocations {
		t.Fatalf("unexpected fold: %+v", folds)
	}
	if folds[0].Outcomes["refused"] != invocations/3 || folds[0].Outcomes["ok"] != invocations-invocations/3 {
		t.Fatalf("unexpected outcomes: %+v", folds[0].Outcomes)
	}
}

func TestFoldUsageCountsISOWeeksInOrder(t *testing.T) {
	rows := []UsageRow{
		{Schema: UsageLedgerSchema, ObservedAt: time.Date(2026, 1, 5, 0, 0, 0, 0, time.UTC), Operation: "up", Outcome: "ok"},
		{Schema: UsageLedgerSchema, ObservedAt: time.Date(2025, 12, 29, 0, 0, 0, 0, time.UTC), Operation: "init", Outcome: "error"},
	}
	var ledger bytes.Buffer
	for _, row := range rows {
		if err := json.NewEncoder(&ledger).Encode(row); err != nil {
			t.Fatal(err)
		}
	}
	folds, err := FoldUsage(&ledger)
	if err != nil {
		t.Fatal(err)
	}
	got := fmt.Sprint(folds[0].Week, ",", folds[1].Week)
	if got != "2026-W01,2026-W02" {
		t.Fatalf("weeks = %s", got)
	}
}
