package trajectory

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestAuditSnapshotUsageAppendPrivacyAndWeeklyFold(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "usage.jsonl")
	rows := []AuditSnapshotUsageRow{
		{ObservedAt: time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC), Operation: "capture", Outcome: "success"},
		{ObservedAt: time.Date(2026, 1, 2, 12, 0, 0, 0, time.UTC), Operation: "replay", Outcome: "refused", Reason: "SNAPSHOT_TAMPERED"},
		{ObservedAt: time.Date(2026, 1, 5, 12, 0, 0, 0, time.UTC), Operation: "replay", Outcome: "error", Reason: "OUTPUT_WRITE_FAILED"},
	}
	for _, row := range rows {
		if err := AppendAuditSnapshotUsage(ledger, row); err != nil {
			t.Fatal(err)
		}
	}
	info, err := os.Stat(ledger)
	if err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS != "windows" {
		if got := info.Mode().Perm(); got != 0o600 {
			t.Fatalf("ledger mode = %o, want 600", got)
		}
	}
	payload, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{filepath.Dir(ledger), "private-host", "transcript-id", "prompt content", strings.Repeat("a", 64)} {
		if bytes.Contains(payload, []byte(forbidden)) {
			t.Fatalf("ledger leaked %q: %s", forbidden, payload)
		}
	}
	weeks, err := ReadAuditSnapshotUsage(ledger)
	if err != nil {
		t.Fatal(err)
	}
	if len(weeks) != 2 || weeks[0].Week != "2026-W01" || weeks[0].Total != 2 || weeks[1].Week != "2026-W02" || weeks[1].Total != 1 {
		t.Fatalf("fold = %#v", weeks)
	}
	if weeks[0].Operations["capture"] != 1 || weeks[0].Operations["replay"] != 1 || weeks[0].Outcomes["refused"] != 1 {
		t.Fatalf("week one = %#v", weeks[0])
	}
}

func TestAuditSnapshotUsageConcurrentAppend(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "usage.jsonl")
	const count = 48
	var wg sync.WaitGroup
	errs := make(chan error, count)
	for i := 0; i < count; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			err := AppendAuditSnapshotUsage(ledger, AuditSnapshotUsageRow{
				ObservedAt: time.Date(2026, 8, 27, 12, 0, i, 0, time.UTC),
				Operation:  []string{"capture", "replay"}[i%2],
				Outcome:    "success",
			})
			if err != nil {
				errs <- err
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	weeks, err := ReadAuditSnapshotUsage(ledger)
	if err != nil {
		t.Fatal(err)
	}
	if len(weeks) != 1 || weeks[0].Total != count || weeks[0].Operations["capture"] != count/2 || weeks[0].Operations["replay"] != count/2 {
		t.Fatalf("fold = %#v", weeks)
	}
}

func TestAuditSnapshotUsageRejectsPrivateOrMalformedRows(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "usage.jsonl")
	err := AppendAuditSnapshotUsage(ledger, AuditSnapshotUsageRow{
		ObservedAt: time.Now().UTC(), Operation: "capture", Outcome: "refused", Reason: "/private/transcript",
	})
	if err == nil {
		t.Fatal("private-looking reason accepted")
	}
	malformed := fmt.Sprintf(`{"schema":%q,"observed_at":"2026-08-27T12:00:00Z","operation":"capture","outcome":"success","snapshot_path":"private"}`+"\n", AuditSnapshotUsageSchema)
	if _, err := FoldAuditSnapshotUsage(strings.NewReader(malformed)); err == nil {
		t.Fatal("unknown privacy-sensitive field accepted")
	}
}
