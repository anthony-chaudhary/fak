package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestLiveIssueSnapshotDrivesSanitizedWorkUnits(t *testing.T) {
	path := filepath.Join(t.TempDir(), "issues.json")
	input := `[{"number":6004,"title":"  fix:\naccept live work  ","body":"SECRET /private/path","labels":[{"name":"bug"}]},{"number":5833,"title":"dogfood readout","body":"token=DO_NOT_LOG"}]`
	if err := os.WriteFile(path, []byte(input), 0o600); err != nil {
		t.Fatal(err)
	}
	r, err := run(t.Context(), config{Workers: 2, Delay: time.Millisecond, LiveInput: path, Selfcheck: true})
	if err != nil {
		t.Fatal(err)
	}
	if r.LogicalShards != 2 || r.Completed != 2 || len(r.LiveInputRecords) != 2 {
		t.Fatalf("report did not consume records: %+v", r)
	}
	if got := r.LiveInputRecords[0]; got.Number != 6004 || got.Title != "fix: accept live work" {
		t.Fatalf("unexpected sanitized record: %+v", got)
	}
	encoded, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	for _, secret := range []string{"SECRET", "/private/path", "DO_NOT_LOG", "token="} {
		if strings.Contains(string(encoded), secret) {
			t.Fatalf("report leaked %q: %s", secret, encoded)
		}
	}
}

func TestLiveIssueSnapshotIsBounded(t *testing.T) {
	path := filepath.Join(t.TempDir(), "issues.json")
	records := make([]liveIssueRecord, maxLiveInputRecords+1)
	for i := range records {
		records[i] = liveIssueRecord{Number: i + 1, Title: "work"}
	}
	data, _ := json.Marshal(records)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadLiveWorkUnits(path); err == nil || !strings.Contains(err.Error(), "limit") {
		t.Fatalf("expected bound refusal, got %v", err)
	}
}
