package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/usagelog"
)

func TestRecordUsageAndWeeklyFold(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	recordUsage(time.Now().Add(-25*time.Millisecond), path)
	rows, err := usagelog.ReadRows(path)
	if err != nil || len(rows) != 1 {
		t.Fatalf("rows=%d err=%v", len(rows), err)
	}
	if rows[0].Verb != "microcontextdemo" || rows[0].ArgsDigest != "" || rows[0].Args != nil {
		t.Fatalf("privacy-safe row=%+v", rows[0])
	}
	b, err := json.Marshal(usagelog.FoldRows(rows, usagelog.FoldOptions{SinceUnixNano: time.Now().Add(-7 * 24 * time.Hour).UnixNano()}))
	if err != nil || !json.Valid(b) {
		t.Fatalf("fold=%s err=%v", b, err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatal(err)
	}
}
