package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

func TestToolprocLeakEventReportRedactsCanary(t *testing.T) {
	canary := "CANARY_DO_NOT_SURFACE_TOOLPROC_2361"
	journal := filepath.Join(t.TempDir(), "leaks.jsonl")
	rows := []toolprocgate.LeakEvent{
		{
			Schema:          toolprocgate.LeakEventSchema,
			Action:          toolprocgate.LeakEgressDenied,
			AtMS:            1_700_000_000_000,
			AgentRunID:      "agent-child-cli",
			ParentRunID:     "agent-parent-cli",
			ToolCallID:      "toolu-egress",
			TraceID:         "trace-egress",
			PolicyDigest:    "sha256:policy-cli",
			Backend:         "claude",
			Reason:          "EGRESS_BLOCK",
			BoundedRef:      toolprocgate.BoundedRef{Kind: "sha256", Digest: "sha256:" + toolprocLeakHash(canary), Len: int64(len(canary))},
			SourceChannel:   "network",
			DescendantState: toolprocgate.DescendantRunning,
		},
		{
			Schema:          toolprocgate.LeakEventSchema,
			Action:          toolprocgate.LeakDescendantReaped,
			AtMS:            1_700_000_000_100,
			AgentRunID:      "agent-child-cli",
			ParentRunID:     "agent-parent-cli",
			ToolCallID:      "toolu-egress",
			TraceID:         "trace-egress",
			PolicyDigest:    "sha256:policy-cli",
			Backend:         "claude",
			Reason:          "TOOL_ORPHANED",
			BoundedRef:      toolprocgate.BoundedRef{Kind: "pidtree", Digest: "tree:bounded-17", Len: 0},
			SourceChannel:   "process_tree",
			DescendantState: toolprocgate.DescendantReaped,
		},
	}
	var raw strings.Builder
	for _, row := range rows {
		b, err := json.Marshal(row)
		if err != nil {
			t.Fatal(err)
		}
		raw.Write(b)
		raw.WriteByte('\n')
	}
	if err := os.WriteFile(journal, []byte(raw.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr strings.Builder
	if rc := runToolproc(&stdout, &stderr, []string{"leaks", "--events", journal}); rc != 0 {
		t.Fatalf("toolproc leaks rc=%d stderr=%s", rc, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"toolproc leak-events", "denied=2", "agent-child-cli", "EGRESS_BLOCK", "descendant=reaped"} {
		if !strings.Contains(out, want) {
			t.Fatalf("human report missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, canary) {
		t.Fatalf("human report leaked canary:\n%s", out)
	}

	stdout.Reset()
	stderr.Reset()
	if rc := runToolproc(&stdout, &stderr, []string{"leaks", "--events", journal, "--json"}); rc != 0 {
		t.Fatalf("toolproc leaks --json rc=%d stderr=%s", rc, stderr.String())
	}
	encoded := stdout.String()
	if !strings.Contains(encoded, `"agent_run_id": "agent-child-cli"`) || !strings.Contains(encoded, `"reason": "EGRESS_BLOCK"`) {
		t.Fatalf("JSON report missing identity/reason:\n%s", encoded)
	}
	if strings.Contains(encoded, canary) {
		t.Fatalf("JSON report leaked canary:\n%s", encoded)
	}
}

func toolprocLeakHash(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}
