package toolcallcontrol

import (
	"strings"
	"testing"
)

func boolp(v bool) *bool { return &v }

func TestReplayIdenticalTraceAndNoSameTurnLeakage(t *testing.T) {
	trace := strings.Join([]string{
		`{"id":"a","turn":1,"tool":"read_file","args":{"path":"a"},"read_only":true,"state_epoch":"s","prompt_units":100,"needed":true,"result_id":"r1","succeeded":true}`,
		`{"id":"b","turn":1,"tool":"read_file","args":{"path":"a"},"read_only":true,"state_epoch":"s","prompt_units":100,"needed":true,"result_id":"r2","succeeded":true}`,
		`{"id":"c","turn":2,"tool":"read_file","args":{"path":"a"},"read_only":true,"state_epoch":"s","prompt_units":200,"needed":false,"result_id":"r3","succeeded":true}`,
	}, "\n")
	rows, err := DecodeReplay(strings.NewReader(trace))
	if err != nil {
		t.Fatal(err)
	}
	report := Replay(rows)
	if len(report.Arms) != 5 {
		t.Fatalf("arms=%d", len(report.Arms))
	}
	for _, arm := range report.Arms {
		if arm.Metrics.Proposed != 3 {
			t.Fatalf("%s saw %d rows", arm.Name, arm.Metrics.Proposed)
		}
	}
	exact, _ := report.Arm("exact-reuse")
	if exact.Decisions[1].Action != Allow {
		t.Fatalf("same-turn result leaked: %+v", exact.Decisions[1])
	}
	if exact.Metrics.UnneededAvoided != 1 || exact.Metrics.NeededSuppressed != 0 || exact.Metrics.ReplayUnitsSaved != 200 {
		t.Fatalf("metrics=%+v", exact.Metrics)
	}
}

func TestDecodeReplayRequiresIndependentLabel(t *testing.T) {
	_, err := DecodeReplay(strings.NewReader(`{"id":"a","tool":"read_file"}`))
	if err == nil || !strings.Contains(err.Error(), "needed label") {
		t.Fatalf("err=%v", err)
	}
}
