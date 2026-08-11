package agentcheckpoint

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func validInput() Input {
	return Input{Actor: "worker-issue-123", Scope: "issue #123", State: StateProgress, StageCurrent: 2, StageTotal: 4, StageName: "implementation", Summary: "Added stale-lease classification", Evidence: []string{"tests/test_leases.py::test_stale"}, Next: "Run supervisor integration tests"}
}

func TestNewProgressComputesStagePercent(t *testing.T) {
	now := time.Date(2026, 8, 11, 12, 0, 0, 0, time.FixedZone("offset", -7*60*60))
	got, err := New(validInput(), now)
	if err != nil {
		t.Fatal(err)
	}
	if got.Stage == nil || got.Stage.Percent != 50 || got.Stage.Current != 2 || got.Stage.Total != 4 {
		t.Fatalf("stage = %#v", got.Stage)
	}
	if got.Timestamp.Location() != time.UTC {
		t.Fatalf("timestamp location = %v", got.Timestamp.Location())
	}
}

func TestNewRejectsIncompleteProgress(t *testing.T) {
	in := validInput()
	in.StageTotal = 0
	if _, err := New(in, time.Now()); err == nil || !strings.Contains(err.Error(), "required for progress") {
		t.Fatalf("err = %v", err)
	}
}

func TestNewEnforcesBlockedAndNextContracts(t *testing.T) {
	in := validInput()
	in.State = StateBlocked
	in.StageCurrent, in.StageTotal, in.StageName = 0, 0, ""
	in.Blockers = nil
	if _, err := New(in, time.Now()); err == nil || !strings.Contains(err.Error(), "blocker") {
		t.Fatalf("blocked err = %v", err)
	}
	in = validInput()
	in.Next = ""
	if _, err := New(in, time.Now()); err == nil || !strings.Contains(err.Error(), "next") {
		t.Fatalf("next err = %v", err)
	}
}

func TestAppendWritesOneDurableJSONLRecord(t *testing.T) {
	record, err := New(validInput(), time.Now())
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "nested", "agent-status.jsonl")
	if err := Append(path, record); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	if !scanner.Scan() {
		t.Fatal("missing row")
	}
	var got Record
	if err := json.Unmarshal(scanner.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Actor != record.Actor || got.Stage == nil || got.Stage.Percent != 50 {
		t.Fatalf("record = %#v", got)
	}
	if scanner.Scan() {
		t.Fatal("more than one row")
	}
}
