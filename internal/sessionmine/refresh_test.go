package sessionmine

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestRefreshIndexRunsImmediatelyThenReusesCheckpoint(t *testing.T) {
	root := t.TempDir()
	sessions := filepath.Join(root, "codex")
	os.MkdirAll(sessions, 0755)
	os.WriteFile(filepath.Join(sessions, "one.jsonl"), []byte(`{"type":"response_item","payload":{"type":"function_call","name":"view_image","arguments":"SECRET"}}`+"\n"), 0644)
	var receipts []RefreshReceipt
	err := RefreshIndex(context.Background(), RefreshOptions{Mine: Options{CodexRoot: sessions, MinSupport: 1}, IndexPath: filepath.Join(root, "index.json"), Interval: time.Millisecond, MaxRuns: 2}, func(r RefreshReceipt) error { receipts = append(receipts, r); return nil })
	if err != nil {
		t.Fatal(err)
	}
	if len(receipts) != 2 || receipts[0].ParsedFiles != 1 || receipts[1].ReusedFiles != 1 || receipts[1].ParsedFiles != 0 {
		t.Fatalf("receipts=%+v", receipts)
	}
}

func TestRefreshIndexStopsOnCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	runs := 0
	err := RefreshIndex(ctx, RefreshOptions{IndexPath: filepath.Join(t.TempDir(), "index.json"), Interval: time.Hour}, func(r RefreshReceipt) error { runs++; cancel(); return nil })
	if !errors.Is(err, context.Canceled) || runs != 1 {
		t.Fatalf("err=%v runs=%d", err, runs)
	}
}

func TestRefreshIndexValidatesBounds(t *testing.T) {
	if err := RefreshIndex(context.Background(), RefreshOptions{IndexPath: "x", MaxRuns: 2}, nil); err == nil {
		t.Fatal("expected interval error")
	}
}

func TestRefreshIndexWritesDurableReceiptAndReleasesLock(t *testing.T) {
	index := filepath.Join(t.TempDir(), "nested", "index.json")
	if err := RefreshIndex(context.Background(), RefreshOptions{IndexPath: index, MaxRuns: 1}, nil); err != nil {
		t.Fatal(err)
	}
	got := inspectRefreshReceipt(index)
	if got.State != "recorded" || got.Outcome != "ok" || got.CompletedAt == "" {
		t.Fatalf("%+v", got)
	}
	if _, err := os.Stat(refreshLockPath(index)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("lock remains: %v", err)
	}
}

func TestRefreshIndexRefusesLiveLockAndReapsStaleLock(t *testing.T) {
	index := filepath.Join(t.TempDir(), "index.json")
	live, _ := json.Marshal(refreshLock{PID: os.Getpid(), StartedAt: time.Now().Format(time.RFC3339)})
	if err := os.WriteFile(refreshLockPath(index), live, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := RefreshIndex(context.Background(), RefreshOptions{IndexPath: index, MaxRuns: 1}, nil); err == nil {
		t.Fatal("expected live contention")
	}
	stale, _ := json.Marshal(refreshLock{PID: 2147483647, StartedAt: time.Now().Format(time.RFC3339)})
	if err := os.WriteFile(refreshLockPath(index), stale, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := RefreshIndex(context.Background(), RefreshOptions{IndexPath: index, MaxRuns: 1}, nil); err != nil {
		t.Fatalf("stale lock was not reclaimed: %v", err)
	}
}

func TestRefreshOutcomeCountersAccumulateAcrossRuns(t *testing.T) {
	root := t.TempDir()
	src := filepath.Join(root, "codex")
	if err := os.MkdirAll(src, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(src, "s.jsonl"), []byte(`{"timestamp":"2026-08-19T00:00:00Z","type":"response_item","payload":{"type":"function_call","name":"read_file"}}`+"\n"), 0644); err != nil {
		t.Fatal(err)
	}
	idx := filepath.Join(root, "index.json")
	var got []RefreshReceipt
	err := RefreshIndex(context.Background(), RefreshOptions{Mine: Options{CodexRoot: src, MinSupport: 1, Limit: 1}, IndexPath: idx, Interval: time.Millisecond, MaxRuns: 2}, func(r RefreshReceipt) error { got = append(got, r); return nil })
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[1].Outcomes.OK != 2 || got[1].Outcomes.ParsedFiles != 1 || got[1].Outcomes.ReusedFiles != 1 {
		t.Fatalf("receipts=%+v", got)
	}
	h := InspectIndexHealthWithOptions(IndexHealthOptions{IndexPath: idx, CodexRoot: src, Now: time.Now()})
	if h.LastRefresh.Outcomes.OK != 2 {
		t.Fatalf("health=%+v", h.LastRefresh)
	}
}
