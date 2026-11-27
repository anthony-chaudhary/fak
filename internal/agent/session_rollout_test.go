package agent

import (
	"bytes"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

func TestSessionRolloutImportParser(t *testing.T) {
	t.Run("four_item_types_in_order", func(t *testing.T) {
		input := strings.Join([]string{
			`{"timestamp":"2026-08-21T00:01:00Z","type":"session_meta","payload":{"id":"codex-sess-1","timestamp":"2026-08-21T00:00:59Z","cwd":"/work/fak","originator":"codex-tui","cli_version":"0.148.0","model_provider":"openai","source":"cli"}}`,
			`{"timestamp":"2026-08-21T00:01:01Z","type":"response_item","payload":{"type":"message","id":"msg-1","role":"user","content":[{"type":"input_text","text":"inspect repo"}]}}`,
			`{"timestamp":"2026-08-21T00:01:02Z","type":"response_item","payload":{"type":"function_call","id":"fc-1","name":"exec_command","arguments":"{\"cmd\":\"rg fixture\"}","call_id":"call-1","status":"completed"}}`,
			`{"timestamp":"2026-08-21T00:01:03Z","type":"response_item","payload":{"type":"function_call_output","call_id":"call-1","output":"{\"exit_code\":0,\"output\":\"ok\"}"}}`,
			`{"timestamp":"2026-08-21T00:01:04Z","type":"compacted","payload":{"message":"compacted earlier turn","replacement_history":[{"type":"message","role":"developer","content":[{"type":"input_text","text":"pinned goal"}]}]}}`,
		}, "\n") + "\n"

		got, err := ParseCodexRollout([]byte(input))
		if err != nil {
			t.Fatalf("ParseCodexRollout unexpected error: %v", err)
		}
		if got.Meta.ID != "codex-sess-1" || got.Meta.CWD != "/work/fak" || got.Meta.Originator != "codex-tui" || got.Meta.CLIVersion != "0.148.0" || got.Meta.ModelProvider != "openai" {
			t.Fatalf("unexpected Meta: %+v", got.Meta)
		}
		if !got.MetaTimestamp.Equal(time.Date(2026, 8, 21, 0, 1, 0, 0, time.UTC)) {
			t.Fatalf("MetaTimestamp = %v, want 2026-08-21T00:01:00Z", got.MetaTimestamp)
		}
		if !got.Meta.Timestamp.Equal(time.Date(2026, 8, 21, 0, 0, 59, 0, time.UTC)) {
			t.Fatalf("Meta.Timestamp = %v, want 2026-08-21T00:00:59Z", got.Meta.Timestamp)
		}
		if got.Skipped != 0 || len(got.UnknownTypes) != 0 {
			t.Fatalf("expected 0 skipped, got skipped=%d unknown=%v", got.Skipped, got.UnknownTypes)
		}
		if len(got.Records) != 4 {
			t.Fatalf("len(Records) = %d, want 4", len(got.Records))
		}

		wantTypes := []RolloutItemType{
			RolloutItemMessage,
			RolloutItemFunctionCall,
			RolloutItemFunctionCallOutput,
			RolloutItemCompacted,
		}
		for i, want := range wantTypes {
			if got.Records[i].Type != want {
				t.Fatalf("Records[%d].Type = %q, want %q", i, got.Records[i].Type, want)
			}
		}
		if got.Records[0].Message == nil || got.Records[0].Message.Role != "user" || got.Records[0].Message.Text != "inspect repo" {
			t.Fatalf("Records[0].Message = %+v", got.Records[0].Message)
		}
		if got.Records[1].FunctionCall == nil || got.Records[1].FunctionCall.Name != "exec_command" || got.Records[1].FunctionCall.CallID != "call-1" || got.Records[1].FunctionCall.Arguments != `{"cmd":"rg fixture"}` {
			t.Fatalf("Records[1].FunctionCall = %+v", got.Records[1].FunctionCall)
		}
		if got.Records[2].FunctionCallOutput == nil || got.Records[2].FunctionCallOutput.CallID != "call-1" || got.Records[2].FunctionCallOutput.Output != `{"exit_code":0,"output":"ok"}` {
			t.Fatalf("Records[2].FunctionCallOutput = %+v", got.Records[2].FunctionCallOutput)
		}
		if got.Records[3].Compacted == nil || got.Records[3].Compacted.Message != "compacted earlier turn" || len(got.Records[3].Compacted.ReplacementHistory) != 1 {
			t.Fatalf("Records[3].Compacted = %+v", got.Records[3].Compacted)
		}
		if got.Records[3].Compacted.ReplacementHistory[0].Message == nil || got.Records[3].Compacted.ReplacementHistory[0].Message.Text != "pinned goal" {
			t.Fatalf("replacement history item = %+v", got.Records[3].Compacted.ReplacementHistory[0])
		}
	})

	t.Run("wrapped_compacted_and_flexible_variants", func(t *testing.T) {
		input := strings.Join([]string{
			`{"type":"session_meta","payload":{"session_id":"codex-alt"}}`,
			`{"type":"response_item","payload":{"type":"message","role":"assistant","content":"plain string"}}`,
			`{"type":"response_item","payload":{"type":"function_call","name":"search","arguments":{"q":"fak"},"call_id":"c1"}}`,
			`{"type":"response_item","payload":{"type":"function_call_output","call_id":"c1","output":[{"type":"input_text","text":"hit"}]}}`,
			`{"type":"response_item","payload":{"type":"compacted","message":"wrapped summary"}}`,
		}, "\n")

		got, err := ParseCodexRollout([]byte(input))
		if err != nil {
			t.Fatalf("ParseCodexRollout error: %v", err)
		}
		if got.Meta.ID != "codex-alt" || len(got.Records) != 4 {
			t.Fatalf("got meta=%+v len=%d", got.Meta, len(got.Records))
		}
		if !got.Records[0].Message.ContentIsString || got.Records[0].Message.Text != "plain string" {
			t.Fatalf("string message mismatch: %+v", got.Records[0].Message)
		}
		if got.Records[1].FunctionCall.Arguments != `{"q":"fak"}` {
			t.Fatalf("object arguments = %q", got.Records[1].FunctionCall.Arguments)
		}
		if got.Records[2].FunctionCallOutput.Output != "hit" || len(got.Records[2].FunctionCallOutput.OutputRaw) == 0 {
			t.Fatalf("array output mismatch: %+v", got.Records[2].FunctionCallOutput)
		}
		if !got.Records[3].Compacted.WrappedInResponse || got.Records[3].Compacted.Message != "wrapped summary" {
			t.Fatalf("wrapped compacted mismatch: %+v", got.Records[3].Compacted)
		}
	})

	t.Run("unknown_item_types_skipped_and_counted", func(t *testing.T) {
		input := strings.Join([]string{
			`{"timestamp":"2026-08-21T00:01:00Z","type":"session_meta","payload":{"id":"codex-unknowns"}}`,
			`{"timestamp":"2026-08-21T00:01:01Z","type":"turn_context","payload":{"model":"gpt-5.6-codex"}}`,
			`{"timestamp":"2026-08-21T00:01:02Z","type":"response_item","payload":{"type":"reasoning","encrypted_content":"opaque"}}`,
			`{"timestamp":"2026-08-21T00:01:03Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}}`,
			`{"timestamp":"2026-08-21T00:01:04Z","type":"event_msg","payload":{"type":"token_count"}}`,
			`{"timestamp":"2026-08-21T00:01:05Z","type":"future_envelope","payload":{"x":1}}`,
		}, "\n")

		got, err := ParseCodexRollout([]byte(input))
		if err != nil {
			t.Fatalf("expected unknown items to be non-fatal, got error: %v", err)
		}
		if len(got.Records) != 1 || got.Records[0].Type != RolloutItemMessage {
			t.Fatalf("got %d records, want 1 message: %+v", len(got.Records), got.Records)
		}
		if got.Skipped != 4 {
			t.Fatalf("Skipped = %d, want 4", got.Skipped)
		}
		wantUnknown := map[string]int{
			"turn_context":            1,
			"response_item/reasoning": 1,
			"event_msg":               1,
			"future_envelope":         1,
		}
		if !reflect.DeepEqual(got.UnknownTypes, wantUnknown) {
			t.Fatalf("UnknownTypes = %v, want %v", got.UnknownTypes, wantUnknown)
		}
	})

	t.Run("missing_session_meta_first_line_refused", func(t *testing.T) {
		cases := []struct {
			name  string
			input string
		}{
			{name: "empty", input: ""},
			{name: "whitespace_only", input: "   \n\t\n"},
			{name: "starts_with_response_item", input: `{"timestamp":"2026-08-21T00:01:00Z","type":"response_item","payload":{"type":"message","role":"user","content":"hi"}}` + "\n"},
			{name: "starts_with_compacted", input: `{"type":"compacted","payload":{"message":"sum"}}` + "\n"},
			{name: "session_meta_null_payload", input: `{"type":"session_meta","payload":null}` + "\n"},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				_, err := ParseCodexRollout([]byte(tc.input))
				if !errors.Is(err, ErrRolloutMissingSessionMeta) {
					t.Fatalf("err = %v, want errors.Is(_, ErrRolloutMissingSessionMeta)", err)
				}
			})
		}
	})

	t.Run("secondary_session_meta_preserves_primary_and_malformed_fails", func(t *testing.T) {
		input := strings.Join([]string{
			`{"type":"session_meta","payload":{"id":"primary-id"}}`,
			`{"type":"session_meta","payload":{"id":"child-embedded-id"}}`,
			`{"type":"response_item","payload":{"type":"message","role":"user","content":"ok"}}`,
		}, "\n")
		got, err := ParseCodexRollout([]byte(input))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got.Meta.ID != "primary-id" {
			t.Fatalf("Meta.ID = %q, want primary-id", got.Meta.ID)
		}
		if len(got.Records) != 1 {
			t.Fatalf("len(Records) = %d, want 1", len(got.Records))
		}

		if _, err := ParseCodexRollout([]byte(`{"type":"session_meta","payload":{"id":"p"}}` + "\n{BROKEN}\n")); !errors.Is(err, ErrRolloutMalformedLine) {
			t.Fatalf("malformed line err = %v, want ErrRolloutMalformedLine", err)
		}
	})
}

func TestSessionRolloutImportRoundTrip(t *testing.T) {
	t.Run("four_record_conversion_and_roundtrip", func(t *testing.T) {
		input := strings.Join([]string{
			`{"timestamp":"2026-08-21T00:01:00Z","type":"session_meta","payload":{"id":"codex-rt-1","timestamp":"2026-08-21T00:00:58Z","cwd":"/work/fak","originator":"codex-tui","cli_version":"0.148.0","model_provider":"openai","source":"cli"}}`,
			`{"timestamp":"2026-08-21T00:01:01Z","type":"response_item","payload":{"type":"message","id":"m1","role":"user","content":[{"type":"input_text","text":"list tests"}]}}`,
			`{"timestamp":"2026-08-21T00:01:02Z","type":"response_item","payload":{"type":"function_call","id":"fc1","name":"shell","arguments":"{\"cmd\":\"go test ./...\"}","call_id":"call-42"}}`,
			`{"timestamp":"2026-08-21T00:01:03Z","type":"response_item","payload":{"type":"function_call_output","call_id":"call-42","output":"ok"}}`,
			`{"timestamp":"2026-08-21T00:01:04Z","type":"compacted","payload":{"message":"turn compacted","replacement_history":[{"type":"message","role":"developer","content":[{"type":"input_text","text":"keep goal"}]}]}}`,
		}, "\n") + "\n"

		parsed, err := ParseCodexRollout([]byte(input))
		if err != nil {
			t.Fatalf("ParseCodexRollout error: %v", err)
		}

		traj, err := ConvertRolloutToTrajectory(parsed)
		if err != nil {
			t.Fatalf("ConvertRolloutToTrajectory error: %v", err)
		}
		if traj.SessionID != "codex-rt-1" || traj.CWD != "/work/fak" || traj.Provider != "openai" {
			t.Fatalf("unexpected trajectory header: %+v", traj)
		}
		if len(traj.Records) != 4 {
			t.Fatalf("len(traj.Records) = %d, want 4", len(traj.Records))
		}

		wantKinds := []SessionTrajectoryKind{
			SessionTrajectoryMessage,
			SessionTrajectoryToolAction,
			SessionTrajectoryToolObservation,
			SessionTrajectoryMarker,
		}
		for i, want := range wantKinds {
			if traj.Records[i].Seq != i+1 {
				t.Fatalf("Records[%d].Seq = %d, want %d", i, traj.Records[i].Seq, i+1)
			}
			if traj.Records[i].Kind != want {
				t.Fatalf("Records[%d].Kind = %q, want %q", i, traj.Records[i].Kind, want)
			}
		}
		if traj.Records[0].Role != "user" || traj.Records[0].Content != "list tests" {
			t.Fatalf("message record = %+v", traj.Records[0])
		}
		if traj.Records[1].ToolName != "shell" || traj.Records[1].CallID != "call-42" || traj.Records[1].Arguments != `{"cmd":"go test ./..."}` {
			t.Fatalf("tool_action record = %+v", traj.Records[1])
		}
		if traj.Records[2].CallID != "call-42" || traj.Records[2].Output != "ok" {
			t.Fatalf("tool_observation record = %+v", traj.Records[2])
		}
		if traj.Records[3].MarkerKind != "compacted" || traj.Records[3].Summary != "turn compacted" || traj.Records[3].ReplacementCount != 1 {
			t.Fatalf("marker record = %+v", traj.Records[3])
		}

		// Round-trip: SessionTrajectory -> CodexRollout -> JSONL -> CodexRollout -> SessionTrajectory.
		rebuiltRollout, err := ConvertTrajectoryToRollout(traj)
		if err != nil {
			t.Fatalf("ConvertTrajectoryToRollout error: %v", err)
		}
		encoded, err := EncodeCodexRollout(rebuiltRollout)
		if err != nil {
			t.Fatalf("EncodeCodexRollout error: %v", err)
		}
		reparsed, err := ParseCodexRollout(encoded)
		if err != nil {
			t.Fatalf("re-ParseCodexRollout error: %v\nencoded:\n%s", err, encoded)
		}
		if !reflect.DeepEqual(reparsed, parsed) {
			t.Fatalf("rollout round-trip mismatch:\ngot:  %+v\nwant: %+v", reparsed, parsed)
		}
		reconverted, err := ConvertRolloutToTrajectory(reparsed)
		if err != nil {
			t.Fatalf("re-ConvertRolloutToTrajectory error: %v", err)
		}
		if !reflect.DeepEqual(reconverted, traj) {
			t.Fatalf("trajectory round-trip mismatch:\ngot:  %+v\nwant: %+v", reconverted, traj)
		}
	})

	t.Run("empty_record_set_yields_empty_trajectory", func(t *testing.T) {
		for _, tc := range []struct {
			name    string
			rollout *CodexRollout
		}{
			{name: "nil_rollout", rollout: nil},
			{name: "zero_value_rollout", rollout: &CodexRollout{}},
			{name: "session_meta_only_jsonl", rollout: func() *CodexRollout {
				r, err := ParseCodexRollout([]byte(`{"type":"session_meta","payload":{"id":"empty-sess"}}` + "\n"))
				if err != nil {
					t.Fatalf("ParseCodexRollout meta-only error: %v", err)
				}
				return r
			}()},
		} {
			t.Run(tc.name, func(t *testing.T) {
				traj, err := ConvertRolloutToTrajectory(tc.rollout)
				if err != nil {
					t.Fatalf("ConvertRolloutToTrajectory(%s) error: %v", tc.name, err)
				}
				if traj == nil || traj.Records == nil || len(traj.Records) != 0 {
					t.Fatalf("expected non-nil empty trajectory, got %+v", traj)
				}
				events, err := traj.ToTrajectoryEvents()
				if err != nil {
					t.Fatalf("ToTrajectoryEvents error: %v", err)
				}
				if len(events) != 0 {
					t.Fatalf("expected 0 trajectory events, got %d", len(events))
				}
			})
		}
	})

	t.Run("trajectory_events_and_checkpoint_projection", func(t *testing.T) {
		input := strings.Join([]string{
			`{"timestamp":"2026-08-21T00:01:00Z","type":"session_meta","payload":{"id":"codex-proj-1","cwd":"/repo","model_provider":"openai"}}`,
			`{"timestamp":"2026-08-21T00:01:01Z","type":"response_item","payload":{"type":"message","id":"m1","role":"user","content":"check status"}}`,
			`{"timestamp":"2026-08-21T00:01:02Z","type":"response_item","payload":{"type":"function_call","name":"git_status","arguments":"{}","call_id":"call-9"}}`,
			`{"timestamp":"2026-08-21T00:01:03Z","type":"response_item","payload":{"type":"function_call_output","call_id":"call-9","output":"clean"}}`,
			`{"timestamp":"2026-08-21T00:01:04Z","type":"compacted","payload":{"message":"window compacted"}}`,
		}, "\n")

		parsed, err := ParseCodexRollout([]byte(input))
		if err != nil {
			t.Fatalf("ParseCodexRollout error: %v", err)
		}
		traj, err := ConvertRolloutToTrajectory(parsed)
		if err != nil {
			t.Fatalf("ConvertRolloutToTrajectory error: %v", err)
		}

		events, err := traj.ToTrajectoryEvents()
		if err != nil {
			t.Fatalf("ToTrajectoryEvents error: %v", err)
		}
		if len(events) != 4 {
			t.Fatalf("len(events) = %d, want 4", len(events))
		}
		if events[0].Kind != trajectory.EventMessage || events[0].Action != "completed" {
			t.Fatalf("events[0] = %+v", events[0])
		}
		if events[1].Kind != trajectory.EventTool || events[1].Action != "proposed" {
			t.Fatalf("events[1] = %+v", events[1])
		}
		if events[2].Kind != trajectory.EventTool || events[2].Action != "completed" {
			t.Fatalf("events[2] = %+v", events[2])
		}
		if events[3].Kind != trajectory.EventCheckpoint || events[3].Action != "compacted" {
			t.Fatalf("events[3] = %+v", events[3])
		}

		encoded, err := trajectory.EncodeEvents(events)
		if err != nil {
			t.Fatalf("EncodeEvents error: %v", err)
		}
		decoded, err := trajectory.DecodeEvents(encoded)
		if err != nil {
			t.Fatalf("DecodeEvents error: %v", err)
		}
		if len(decoded) != 4 {
			t.Fatalf("decoded events = %d, want 4", len(decoded))
		}
		turns, err := trajectory.CompactTurns(decoded)
		if err != nil || len(turns) != 1 || turns[0].Query != "check status" {
			t.Fatalf("CompactTurns = %+v, err=%v", turns, err)
		}

		cp := traj.ToSessionCheckpoint()
		if cp.Version != SessionCheckpointVersion || cp.SessionID != "codex-proj-1" || cp.CWD != "/repo" || cp.Provider != "openai" || cp.Status != "imported" || cp.Task != "check status" {
			t.Fatalf("unexpected SessionCheckpoint projection: %+v", cp)
		}
		if len(cp.Messages) != 3 {
			t.Fatalf("cp.Messages len = %d, want 3 (user + assistant tool call + tool result): %+v", len(cp.Messages), cp.Messages)
		}
		if cp.Messages[0].Role != RoleUser || cp.Messages[1].Role != RoleAssistant || len(cp.Messages[1].ToolCalls) != 1 || cp.Messages[1].ToolCalls[0].Function.Name != "git_status" || cp.Messages[2].Role != RoleTool || cp.Messages[2].Name != "git_status" || cp.Messages[2].Content != "clean" {
			t.Fatalf("unexpected projected messages: %+v", cp.Messages)
		}

		// Ensure JSON serialization of SessionTrajectory works cleanly.
		rawTraj, err := json.Marshal(traj)
		if err != nil || !bytes.Contains(rawTraj, []byte(`"tool_action"`)) || !bytes.Contains(rawTraj, []byte(`"tool_observation"`)) {
			t.Fatalf("json.Marshal(traj) failed or missing kinds: err=%v raw=%s", err, rawTraj)
		}
	})
}
