package toolprocgate

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"testing"
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

func sha256Hex(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}
