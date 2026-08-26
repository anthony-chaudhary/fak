package atif

import (
	"bytes"
	"strings"
	"testing"
)

func v17str(s string) *string { return &s }

func TestHarborV17NestedRoundTrip(t *testing.T) {
	t.Parallel()
	stamp := "2026-08-26T00:00:00Z"
	child := HarborTrajectory{
		SchemaVersion: HarborVersion,
		TrajectoryID:  v17str("child"),
		SessionID:     v17str("session"),
		Agent:         AgentV17{Name: "worker", Version: "test"},
		Steps:         []StepV17{},
	}
	step := StepV17{
		StepID:    1,
		Timestamp: &stamp,
		Source:    "agent",
		Message:   map[string]any{"role": "assistant", "content": "done"},
		ToolCalls: []ToolCallV17{{
			ToolCallID:   "call-1",
			FunctionName: "read",
			Arguments:    map[string]any{"path": "README.md"},
		}},
		Observation: &ObservationV17{Results: []ObservationResultV17{{
			SourceCallID: v17str("call-1"),
			Content:      "ok",
			SubagentTrajectoryRef: []SubagentRefV17{{
				TrajectoryID: v17str("child"),
			}},
		}}},
	}
	in := HarborTrajectory{
		SchemaVersion:        HarborVersion,
		TrajectoryID:         v17str("parent"),
		SessionID:            v17str("session"),
		Agent:                AgentV17{Name: "fak", Version: "test"},
		Steps:                []StepV17{step},
		SubagentTrajectories: []HarborTrajectory{child},
	}
	var buf bytes.Buffer
	if err := WriteHarbor(&buf, in); err != nil {
		t.Fatal(err)
	}
	got, err := ReadHarbor(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if got.TrajectoryID == nil || *got.TrajectoryID != "parent" || len(got.SubagentTrajectories) != 1 || got.SubagentTrajectories[0].TrajectoryID == nil || *got.SubagentTrajectories[0].TrajectoryID != "child" {
		t.Fatalf("relationship lost: %#v", got)
	}
}

func TestHarborV17RejectsUnknownVersion(t *testing.T) {
	t.Parallel()
	_, err := ReadHarbor(strings.NewReader(`{"schema_version":"ATIF-v9","trajectory_id":"x","agent":{"name":"fak"},"steps":[]}`))
	if err == nil || !strings.Contains(err.Error(), "unsupported schema_version") {
		t.Fatalf("unknown-version err=%v", err)
	}
}
