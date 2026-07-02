package toolprocgate

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/normgate"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

const syntheticChildCanary = "sk-abcdef0123456789abcdef0123"

func TestChildOutputAdmissionQuarantinesLeakBeforeContextLogTranscript(t *testing.T) {
	out := AdmitChildOutput(context.Background(), ChildOutput{
		AgentRunID: "agent-run-2359",
		CallID:     "child-stdout-1",
		Tool:       "BashOutput",
		Channel:    ChannelStdout,
		Bytes:      []byte("stdout chunk carried synthetic canary " + syntheticChildCanary),
	})

	if out.Verdict.Kind != abi.VerdictQuarantine || out.Verdict.Reason != abi.ReasonSecretExfil {
		t.Fatalf("verdict = %v/%s, want Quarantine/SECRET_EXFIL", out.Verdict.Kind, abi.ReasonName(out.Verdict.Reason))
	}
	if out.Meta["agent_run_id"] != "agent-run-2359" || out.Meta["source_channel"] != string(ChannelStdout) {
		t.Fatalf("metadata = %v, want agent_run_id + stdout channel", out.Meta)
	}
	if strings.Contains(string(out.Bytes), syntheticChildCanary) {
		t.Fatalf("admitted bytes still contain canary: %q", out.Bytes)
	}

	parentContext := "tool result:\n" + string(out.Bytes)
	normalLog := fmt.Sprintf("agent_run_id=%s source_channel=%s reason=%s len=%d digest=%s payload=%s",
		out.Meta["agent_run_id"], out.Meta["source_channel"], out.Meta["reason"], out.InputLen, out.InputSHA256, string(out.Bytes))
	transcriptSidecar, err := json.Marshal(map[string]any{
		"type": "user",
		"message": map[string]any{"content": []map[string]any{{
			"type": "tool_result", "content": string(out.Bytes),
		}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	for name, surface := range map[string]string{
		"parent context":     parentContext,
		"normal log":         normalLog,
		"transcript sidecar": string(transcriptSidecar),
	} {
		if strings.Contains(surface, syntheticChildCanary) {
			t.Fatalf("%s leaked canary after admission: %s", name, surface)
		}
	}
}

func TestChildOutputToolProcBenignPassThrough(t *testing.T) {
	body := "build completed without sensitive output\n"
	out := AdmitChildOutput(context.Background(), ChildOutput{
		AgentRunID: "agent-run-2359",
		CallID:     "child-stdout-clean",
		Tool:       "BashOutput",
		Channel:    ChannelStdout,
		Bytes:      []byte(body),
	})

	if out.Verdict.Kind != abi.VerdictDefer && out.Verdict.Kind != abi.VerdictAllow {
		t.Fatalf("benign verdict = %v/%s, want admitted Allow/Defer", out.Verdict.Kind, abi.ReasonName(out.Verdict.Reason))
	}
	if got := string(out.Bytes); got != body {
		t.Fatalf("benign output changed: got %q want %q", got, body)
	}
	if out.Meta["source_channel"] != string(ChannelStdout) || out.Meta["admit"] == "quarantined" {
		t.Fatalf("benign metadata = %v, want stdout admitted metadata", out.Meta)
	}
}

func TestChildOutputQuarantineDropsPostKillPayload(t *testing.T) {
	Reset()
	t.Cleanup(Reset)
	Kill("child-after-kill", toolproc.ReasonToolDeadlineExceededName)

	out := AdmitChildOutput(context.Background(), ChildOutput{
		AgentRunID: "agent-run-2359",
		CallID:     "child-after-kill",
		Tool:       "BashOutput",
		Channel:    ChannelStderr,
		Bytes:      []byte("late stderr payload " + syntheticChildCanary),
	})

	if out.Verdict.Kind != abi.VerdictQuarantine || out.Verdict.Reason != toolproc.ReasonToolResultAfterKill {
		t.Fatalf("post-kill verdict = %v/%s, want Quarantine/TOOL_RESULT_AFTER_KILL",
			out.Verdict.Kind, abi.ReasonName(out.Verdict.Reason))
	}
	if strings.Contains(string(out.Bytes), syntheticChildCanary) || strings.Contains(string(out.Bytes), "late stderr payload") {
		t.Fatalf("post-kill output returned dropped payload: %q", out.Bytes)
	}
	if out.Meta["kill_reason"] != toolproc.ReasonToolDeadlineExceededName ||
		out.Meta["source_channel"] != string(ChannelStderr) ||
		out.Meta["agent_run_id"] != "agent-run-2359" {
		t.Fatalf("post-kill metadata = %v, want kill reason + channel + agent run", out.Meta)
	}
}
