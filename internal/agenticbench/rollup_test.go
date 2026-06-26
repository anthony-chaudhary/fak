package agenticbench

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestBuildKeepsEpicPendingForFixtureEvidence(t *testing.T) {
	root := fixtureRoot(t)
	report, err := Build(root, time.Date(2026, 6, 26, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if report.Status != "PENDING_EXTERNAL_HARNESS" || report.ResultClaimAllowed {
		t.Fatalf("status/claim = %s/%t, want pending/false", report.Status, report.ResultClaimAllowed)
	}
	if report.Summary.ChildrenParsed != report.Summary.ChildrenTotal {
		t.Fatalf("children parsed = %d/%d", report.Summary.ChildrenParsed, report.Summary.ChildrenTotal)
	}
	if report.Summary.ResultClaimArtifacts != 0 {
		t.Fatalf("result claim artifacts = %d, want 0", report.Summary.ResultClaimArtifacts)
	}
	if got, want := len(report.Summary.PendingChildren), 6; got != want {
		t.Fatalf("pending children = %v (len %d), want %d live lanes", report.Summary.PendingChildren, got, want)
	}
	if gateOK(report, "result_packet_graduated") {
		t.Fatal("result_packet_graduated must stay false for fixture-only evidence")
	}
	md := RenderMarkdown(report)
	for _, want := range []string{"PENDING_EXTERNAL_HARNESS", "#870", "result_packet_graduated"} {
		if !strings.Contains(md, want) {
			t.Fatalf("markdown missing %q:\n%s", want, md)
		}
	}
}

func TestBuildMarksMissingArtifactsAsFailed(t *testing.T) {
	root := fixtureRoot(t)
	if err := os.Remove(filepath.Join(root, "experiments", "agent-live", "swebench-opus-smoke-contract-20260626.json")); err != nil {
		t.Fatal(err)
	}
	report, err := Build(root, time.Unix(0, 0))
	if err != nil {
		t.Fatal(err)
	}
	if gateOK(report, "child_artifacts_parse") {
		t.Fatal("child_artifacts_parse must fail when a child artifact is missing")
	}
	if len(report.Summary.FailedChildren) == 0 || report.Summary.FailedChildren[0] != 871 {
		t.Fatalf("failed children = %v, want #871", report.Summary.FailedChildren)
	}
}

func fixtureRoot(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	write(t, root, "experiments/agent-live/agentdojo-fak-fullstack-20260625.json", `{
  "gate": "PASS",
  "asr_fullstack_succeeded": 0,
  "benign_completed": 2
}`)
	write(t, root, "experiments/vllm/glm52-agentic-battery/final-check.json", `{
  "summary": {
    "status": "PENDING_MEASUREMENT",
    "required_passed": 4,
    "required_artifacts": 11
  }
}`)
	write(t, root, "experiments/agent-live/swebench-opus-smoke-contract-20260626.json", `{
  "status": "READY_FOR_REMOTE_GRADING",
  "result_claim_allowed": false,
  "required_before_claim": ["raw predictions", "official reports"]
}`)
	write(t, root, "experiments/agent-live/deepswe-raw-fak-contract-20260626.json", `{
  "status": "READY_FOR_EXTERNAL_RUN",
  "result_claim_allowed": false,
  "required_before_claim": ["raw predictions", "fak predictions", "official reports"]
}`)
	write(t, root, "experiments/agent-live/toolsandbox-official-run-contract-20260626.json", `{
  "status": "READY_FOR_EXTERNAL_HARNESS",
  "result_claim_allowed": false,
  "required_before_claim": ["official task ids", "raw output", "fak output", "grader summary"]
}`)
	write(t, root, "experiments/agent-live/terminalbench-command-boundary-smoke-20260625.json", fixtureReport())
	write(t, root, "experiments/agent-live/browser-action-mediation-smoke-20260625.json", fixtureReport())
	write(t, root, "BENCHMARK-AUTHORITY.md", "# BENCHMARK AUTHORITY\n\n## AgentDojo Structural Safety Floor\n\n## ToolSandbox/tau3 Adapter Smoke\n\n### Promotion Gate\n")
	return root
}

func fixtureReport() string {
	return `{
  "evidence_class": "SIMULATED_LOCAL_FIXTURE",
  "result_claim_allowed": false,
  "promotion_requirements": ["official task ids", "grader output"]
}`
}

func write(t *testing.T, root, rel, body string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func gateOK(r *Report, name string) bool {
	for _, gate := range r.Acceptance {
		if gate.Name == name {
			return gate.OK
		}
	}
	return false
}
