package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestRWBackoffHistoryReadsOnlySignedLaunches(t *testing.T) {
	p := filepath.Join(t.TempDir(), "l.jsonl")
	raw := "{\"ts\":\"2026-07-11T12:00:00Z\",\"session\":\"a\",\"signature\":\"x\",\"phase\":\"launched\"}\n{\"ts\":\"2026-07-11T12:01:00Z\",\"session\":\"b\",\"signature\":\"x\",\"phase\":\"deferred\"}\n"
	if err := os.WriteFile(p, []byte(raw), 0644); err != nil {
		t.Fatal(err)
	}
	got := rwBackoffHistory(p)
	if len(got) != 1 || got[0].Session != "a" || got[0].Signature != "x" || !got[0].At.Equal(time.Date(2026, 7, 11, 12, 0, 0, 0, time.UTC)) {
		t.Fatalf("got=%+v", got)
	}
}
