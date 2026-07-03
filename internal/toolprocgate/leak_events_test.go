package toolprocgate

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/normgate"
)

func TestLeakEventReportCountsAndRedactsCanary(t *testing.T) {
	canary := "CANARY_DO_NOT_SURFACE_2361"
	event := LeakEvent{
		Schema:          LeakEventSchema,
		Action:          LeakOutputQuarantined,
		AtMS:            1_700_000_000_000,
		AgentRunID:      "agent-child-2361",
		ParentRunID:     "agent-parent-2361",
		ToolCallID:      "toolu-leak-2361",
		TraceID:         "trace-leak-2361",
		PolicyDigest:    "sha256:policy2361",
		Backend:         "claude",
		Reason:          "TOOL_RESULT_AFTER_KILL",
		BoundedRef:      BoundedRef{Kind: "sha256", Digest: "sha256:" + sha256Hex(canary), Len: int64(len(canary))},
		SourceChannel:   "stdout",
		DescendantState: DescendantReaped,
	}
	line, err := json.Marshal(event)
	if err != nil {
		t.Fatal(err)
	}
	events, err := ParseLeakEvents(strings.NewReader(string(line) + "\n"))
	if err != nil {
		t.Fatal(err)
	}
	report := LeakReportFromEvents(events)
	if report.Rows != 1 || report.Denied != 1 {
		t.Fatalf("report counts rows=%d denied=%d, want 1/1", report.Rows, report.Denied)
	}
	if report.Counts.ByReason["TOOL_RESULT_AFTER_KILL"] != 1 || report.Counts.ByChannel["stdout"] != 1 {
		t.Fatalf("report counts = %+v", report.Counts)
	}
	var out strings.Builder
	RenderLeakReport(&out, report)
	rendered := out.String()
	for _, want := range []string{"agent-child-2361", "TOOL_RESULT_AFTER_KILL", "descendant=reaped"} {
		if !strings.Contains(rendered, want) {
			t.Fatalf("rendered report missing %q:\n%s", want, rendered)
		}
	}
	if strings.Contains(rendered, canary) {
		t.Fatalf("rendered report leaked canary:\n%s", rendered)
	}
	encoded, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), canary) {
		t.Fatalf("JSON report leaked canary:\n%s", encoded)
	}
}

func TestLeakEventParserRefusesPayloadFields(t *testing.T) {
	row := `{"schema":"fak.toolprocgate.leak-event.v1","action":"egress_denied","at_unix_ms":1,"agent_run_id":"a","parent_run_id":"p","tool_call_id":"t","trace_id":"tr","policy_digest":"sha256:p","backend":"claude","reason":"EGRESS_BLOCK","bounded_ref":{"kind":"sha256","digest":"sha256:x","len":7},"source_channel":"network","descendant_state":"running","payload":"CANARY_DO_NOT_SURFACE"}`
	if _, err := ParseLeakEvents(strings.NewReader(row + "\n")); err == nil {
		t.Fatal("raw payload fields must be refused, not silently folded into observability")
	}
}

func TestLeakEventEmissionFromEnforcementAdaptersRedactsCanary(t *testing.T) {
	Reset()
	t.Cleanup(Reset)
	canary := "sk-abcdef0123456789abcdef0123"

	broker := NewSpawnBroker()
	_, err := broker.admitAt(SpawnAttempt{
		AgentRunID:   "agent-child-2361",
		ParentRunID:  "agent-parent-2361",
		ToolCallID:   "toolu-spawn-2361",
		PolicyDigest: "sha256:policy2361",
		Argv:         []string{"agent", "--token=" + canary},
		Env:          []EnvVar{{Name: "API_TOKEN", Value: canary}},
		CWD:          t.TempDir(),
		Backend:      "guard",
		Envelope:     CapabilityEnvelope{Capabilities: []abi.Capability{"not.spawn"}},
	}, 1_700_000_000_000)
	var denied SpawnDeniedError
	if !errors.As(err, &denied) {
		t.Fatalf("spawn admit error = %T/%v, want SpawnDeniedError", err, err)
	}
	events := broker.LeakEvents()
	if len(events) != 1 || events[0].Action != LeakSpawnDenied || events[0].Reason != "MISSING_SPAWN_CAPABILITY" {
		t.Fatalf("spawn leak events = %+v, want one spawn_denied/MISSING_SPAWN_CAPABILITY", events)
	}

	out := AdmitChildOutput(context.Background(), ChildOutput{
		AgentRunID:      "agent-child-2361",
		ParentRunID:     "agent-parent-2361",
		ToolCallID:      "toolu-output-2361",
		TraceID:         "trace-output-2361",
		PolicyDigest:    "sha256:policy2361",
		Backend:         "guard",
		Channel:         ChannelStdout,
		DescendantState: DescendantRunning,
		AtMS:            1_700_000_000_100,
		Bytes:           []byte("stdout carried synthetic canary " + canary),
	})
	if out.LeakEvent == nil {
		t.Fatalf("quarantined child output did not emit a leak event: verdict=%v/%s", out.Verdict.Kind, abi.ReasonName(out.Verdict.Reason))
	}
	if out.LeakEvent.Action != LeakOutputQuarantined || out.LeakEvent.Reason != "SECRET_EXFIL" {
		t.Fatalf("output leak event = %+v, want output_quarantined/SECRET_EXFIL", *out.LeakEvent)
	}
	events = append(events, *out.LeakEvent)
	report := LeakReportFromEvents(events)
	if report.Rows != 2 || report.Denied != 2 {
		t.Fatalf("report rows/denied = %d/%d, want 2/2", report.Rows, report.Denied)
	}
	if report.Counts.ByDescendantState[string(DescendantRunning)] != 1 ||
		report.Counts.ByDescendantState[string(DescendantNone)] != 1 {
		t.Fatalf("descendant counts = %+v", report.Counts.ByDescendantState)
	}

	var rendered strings.Builder
	RenderLeakReport(&rendered, report)
	text := rendered.String()
	for _, want := range []string{
		"agent-child-2361",
		"agent-parent-2361",
		"toolu-output-2361",
		"SECRET_EXFIL",
		"descendant=running",
		"spawn_denied",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("report missing %q:\n%s", want, text)
		}
	}
	if strings.Contains(text, canary) {
		t.Fatalf("rendered report leaked canary:\n%s", text)
	}
	encoded, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), canary) {
		t.Fatalf("JSON report leaked canary:\n%s", encoded)
	}

	clean := AdmitChildOutput(context.Background(), ChildOutput{
		AgentRunID:   "agent-child-clean",
		ParentRunID:  "agent-parent-2361",
		ToolCallID:   "toolu-clean-2361",
		TraceID:      "trace-clean-2361",
		PolicyDigest: "sha256:policy2361",
		Backend:      "guard",
		Channel:      ChannelStdout,
		AtMS:         1_700_000_000_200,
		Bytes:        []byte("ordinary progress line\n"),
	})
	if clean.LeakEvent != nil {
		t.Fatalf("benign child output should not emit a leak event: %+v", *clean.LeakEvent)
	}
}

func sha256Hex(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}
