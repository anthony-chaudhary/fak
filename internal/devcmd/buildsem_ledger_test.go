package devcmd

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

// readBuildSemLedger parses every waits.jsonl row under dir.
func readBuildSemLedger(t *testing.T, dir string) []map[string]any {
	t.Helper()
	f, err := os.Open(filepath.Join(dir, buildSemLedgerName))
	if err != nil {
		t.Fatalf("open %s: %v", buildSemLedgerName, err)
	}
	defer f.Close()
	var rows []map[string]any
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var r map[string]any
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			t.Fatalf("malformed ledger line %q: %v", sc.Text(), err)
		}
		rows = append(rows, r)
	}
	return rows
}

func ledgerOutcomes(rows []map[string]any) []string {
	out := make([]string, 0, len(rows))
	for _, r := range rows {
		out = append(out, r["outcome"].(string))
	}
	return out
}

// fak-test:runtime fast est=200ms
// A public holder must leave acquired+released rows in the shared ledger so
// the private fold counts this pool's occupancy through its legacy path.
func TestBuildSemLedgerRows(t *testing.T) {
	dir := t.TempDir()
	sem := NewBuildSemaphore(dir, 2)

	release, err := sem.Acquire(context.Background(), 2*time.Second)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	release()

	rows := readBuildSemLedger(t, dir)
	if got, want := ledgerOutcomes(rows), []string{"acquired", "released"}; len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("outcomes = %v, want %v", got, want)
	}
	for i, r := range rows {
		if r["label"] != "fak buildsem" || r["tool"] != "fak buildsem" || r["pid"] == nil || r["slots"] == nil || r["at"] == nil {
			t.Fatalf("row %d = %v, want label/tool/pid/slots/at populated", i, r)
		}
	}
	held, ok := rows[1]["held_ms"].(float64)
	if !ok || held < 0 {
		t.Fatalf("released held_ms = %v, want a non-negative number", rows[1]["held_ms"])
	}
}

// fak-test:runtime fast est=300ms
// A refused acquisition leaves one timeout row carrying the wait actually spent.
func TestBuildSemLedgerTimeoutRow(t *testing.T) {
	dir := t.TempDir()
	sem := NewBuildSemaphore(dir, 2)

	held := make([]*os.File, 0, 2)
	for i := 0; i < 2; i++ {
		f, err := os.OpenFile(sem.slotPath(i), os.O_CREATE|os.O_RDWR, 0o666)
		if err != nil {
			t.Fatal(err)
		}
		if err := flock.TryLock(f); err != nil {
			t.Fatalf("hold slot %d: %v", i, err)
		}
		held = append(held, f)
	}
	defer func() {
		for _, f := range held {
			_ = flock.Unlock(f)
			_ = f.Close()
		}
	}()

	start := time.Now()
	_, err := sem.Acquire(context.Background(), 150*time.Millisecond)
	if !errors.Is(err, ErrBuildSlotTimeout) {
		t.Fatalf("err = %v, want ErrBuildSlotTimeout", err)
	}

	rows := readBuildSemLedger(t, dir)
	if len(rows) != 1 {
		t.Fatalf("rows = %v, want exactly one timeout row", rows)
	}
	last := rows[len(rows)-1]
	if last["outcome"] != "timeout" {
		t.Fatalf("outcome = %v, want timeout", last["outcome"])
	}
	waitMS, _ := last["wait_ms"].(float64)
	if waitMS < float64(time.Since(start).Milliseconds())/2 {
		t.Fatalf("wait_ms = %v, want roughly the full 150ms wait", last["wait_ms"])
	}
}

// fak-test:runtime fast est=100ms
// AcquireBuildSlotLabel records the caller-named label so per-tool folds can
// distinguish fak buildcheck from raw buildsem holders.
func TestAcquireBuildSlotLabelRecordsLabel(t *testing.T) {
	tmp := t.TempDir()
	SetDefaultBuildSemaphoreDir(tmp)
	SetDefaultBuildConcurrency(1)
	t.Cleanup(func() {
		SetDefaultBuildSemaphoreDir("")
		SetDefaultBuildConcurrency(0)
	})

	release, err := AcquireBuildSlotLabel(context.Background(), 100*time.Millisecond, "fak buildcheck")
	if err != nil {
		t.Fatalf("AcquireBuildSlotLabel: %v", err)
	}
	release()

	rows := readBuildSemLedger(t, tmp)
	if len(rows) != 2 {
		t.Fatalf("rows = %v, want acquired+released", rows)
	}
	for _, r := range rows {
		if r["label"] != "fak buildcheck" || r["tool"] != "fak buildcheck" {
			t.Fatalf("row = %v, want label and tool \"fak buildcheck\"", r)
		}
	}
}
