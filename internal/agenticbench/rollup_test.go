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
	if report.Summary.ResultPacketsTotal != 0 || report.Summary.ResultPacketsPassed != 0 {
		t.Fatalf("result packets = %d/%d, want none", report.Summary.ResultPacketsPassed, report.Summary.ResultPacketsTotal)
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

func TestBuildRejectsIncompleteResultPacket(t *testing.T) {
	root := fixtureRoot(t)
	write(t, root, "experiments/agent-live/agentic-benchmark-result-packets/bad.json", `{
  "schema": "fak.agentic-benchmark-result-packet.v1",
  "issue": 871,
  "status": "PASS_RESULT",
  "result_claim_allowed": true,
  "benchmark_native": true,
  "same_task_ids": true,
  "same_model": true,
  "same_budget": true,
  "official_grader": {"available": false},
  "arms": [{"role": "raw", "name": "raw-opus"}],
  "metric_categories": {
    "task_success": true,
    "safe_success": true,
    "cost_or_token_budget": true,
    "latency": true,
    "policy_events": true,
    "evidence_completeness": true
  },
  "artifacts": ["experiments/agent-live/missing.json"]
}`)
	report, err := Build(root, time.Unix(0, 0))
	if err != nil {
		t.Fatal(err)
	}
	if report.Summary.ResultPacketsTotal != 1 || report.Summary.ResultPacketsPassed != 0 || report.Summary.ResultPacketsFailed != 1 {
		t.Fatalf("packet summary = %+v", report.Summary)
	}
	if report.Summary.ResultClaimArtifacts != 0 || gateOK(report, "result_packet_graduated") {
		t.Fatalf("incomplete packet must not graduate: summary=%+v", report.Summary)
	}
	packet := report.ResultPackets[0]
	for _, want := range []string{"official_grader.available", "arms must include a fak role", "artifact missing"} {
		if !strings.Contains(packet.Detail, want) {
			t.Fatalf("packet detail missing %q: %s", want, packet.Detail)
		}
	}
}

func TestBuildAcceptsWitnessedResultPacketButKeepsEpicPending(t *testing.T) {
	root := fixtureRoot(t)
	for _, rel := range []string{
		"experiments/agent-live/result-opus/raw-report.json",
		"experiments/agent-live/result-opus/fak-report.json",
		"experiments/agent-live/result-opus/compare.json",
	} {
		write(t, root, rel, `{"ok": true}`)
	}
	write(t, root, "experiments/agent-live/agentic-benchmark-result-packets/opus.json", `{
  "schema": "fak.agentic-benchmark-result-packet.v1",
  "issue": 871,
  "packet": "C",
  "status": "PASS_RESULT",
  "result_claim_allowed": true,
  "benchmark_native": true,
  "same_task_ids": true,
  "same_model": true,
  "same_budget": true,
  "official_grader": {"available": true},
  "arms": [
    {"role": "raw", "name": "raw-opus"},
    {"role": "fak", "name": "fak-opus"}
  ],
  "metric_categories": {
    "task_success": true,
    "safe_success": true,
    "cost_or_token_budget": true,
    "latency": true,
    "policy_events": true,
    "evidence_completeness": true
  },
  "artifacts": [
    "experiments/agent-live/result-opus/raw-report.json",
    "experiments/agent-live/result-opus/fak-report.json",
    "experiments/agent-live/result-opus/compare.json"
  ]
}`)
	report, err := Build(root, time.Unix(0, 0))
	if err != nil {
		t.Fatal(err)
	}
	if report.Summary.ResultPacketsPassed != 1 || report.Summary.ResultClaimArtifacts != 1 {
		t.Fatalf("packet did not graduate: summary=%+v packets=%+v", report.Summary, report.ResultPackets)
	}
	if !gateOK(report, "result_packet_graduated") {
		t.Fatal("result_packet_graduated should pass when a witnessed packet is present")
	}
	if report.Status != "PENDING_EXTERNAL_HARNESS" || report.ResultClaimAllowed {
		t.Fatalf("epic status/claim = %s/%t, want pending/false while sibling lanes remain pending", report.Status, report.ResultClaimAllowed)
	}
	md := RenderMarkdown(report)
	for _, want := range []string{"Result Packets", "opus.json", "PASS_RESULT"} {
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

func TestBuildExternalHarnessQueue(t *testing.T) {
	root := fixtureRoot(t)
	queue, err := BuildExternalHarnessQueue(root, time.Date(2026, 6, 26, 1, 2, 3, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if queue.Schema != ExternalHarnessQueueSchema || queue.ResultClaimAllowed {
		t.Fatalf("queue schema/claim = %s/%t", queue.Schema, queue.ResultClaimAllowed)
	}
	if got, want := queue.Summary.ItemsTotal, 6; got != want {
		t.Fatalf("items total = %d, want %d", got, want)
	}
	if got, want := queue.Summary.BlockedItems, 1; got != want {
		t.Fatalf("blocked items = %d, want %d", got, want)
	}
	glm := queueItem(queue, 870)
	if glm.ExternalState != "BLOCKED_ON_H200_VLLM_READY" {
		t.Fatalf("GLM external state = %q", glm.ExternalState)
	}
	if len(glm.Commands) != 2 {
		t.Fatalf("GLM commands = %+v, want preflight and serving witness", glm.Commands)
	}
	opus := queueItem(queue, 871)
	if len(opus.Arms) != 2 || opus.Arms[0].Command == "" || opus.Arms[1].Command == "" {
		t.Fatalf("Opus arms not queued: %+v", opus.Arms)
	}
	md := RenderExternalHarnessQueueMarkdown(queue)
	for _, want := range []string{"External Harness Queue", "BLOCKED_ON_H200_VLLM_READY", "raw-opus"} {
		if !strings.Contains(md, want) {
			t.Fatalf("queue markdown missing %q:\n%s", want, md)
		}
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
    "required_artifacts": 11,
    "required_failed": ["preflight"],
    "required_missing": ["serving_witness"],
    "honesty": "No GLM-5.2/vLLM benchmark number is quotable yet."
  },
  "steps": [
    {
      "id": "preflight",
      "kind": "live-readiness",
      "description": "Readiness gate",
      "required": true,
      "command": {"shell": "python tools/glm52_serve_preflight.py --require-ready"},
      "artifacts": ["experiments/vllm/glm52-vllm-preflight.json"],
      "artifact_status": "FAIL",
      "artifact_detail": "BLOCKED_ARCH"
    },
    {
      "id": "serving_witness",
      "kind": "live-serving",
      "description": "Serving witness",
      "required": true,
      "command": {"shell": "python tools/glm52_serving_witness.py"},
      "artifacts": ["experiments/glm52/full-size-serving-witness.json"],
      "artifact_status": "MISSING",
      "artifact_detail": "missing"
    }
  ]
}`)
	write(t, root, "experiments/agent-live/swebench-opus-smoke-contract-20260626.json", `{
  "status": "READY_FOR_REMOTE_GRADING",
  "result_claim_allowed": false,
  "arms": [
    {"name": "raw-opus", "harness": "raw", "command": "mini-extra swebench", "output_dir": "raw", "required_artifacts": ["predictions.json"]},
    {"name": "fak-opus", "harness": "fak", "command": "go run ./cmd/fak swebench run", "output_dir": "fak", "required_artifacts": ["predictions.json"]}
  ],
  "required_before_claim": ["raw predictions", "official reports"],
  "compare_metrics": ["solve_rate"]
}`)
	write(t, root, "experiments/agent-live/deepswe-raw-fak-contract-20260626.json", `{
  "status": "READY_FOR_EXTERNAL_RUN",
  "result_claim_allowed": false,
  "arms": [
    {"name": "raw-deepswe", "command": "raw"},
    {"name": "fak-deepswe", "command": "fak"}
  ],
  "required_before_claim": ["raw predictions", "fak predictions", "official reports"]
}`)
	write(t, root, "experiments/agent-live/toolsandbox-official-run-contract-20260626.json", `{
  "status": "READY_FOR_EXTERNAL_HARNESS",
  "result_claim_allowed": false,
  "arms": [
    {"name": "raw-toolsandbox", "command": "raw"},
    {"name": "fak-toolsandbox", "command": "fak"}
  ],
  "required_before_claim": ["official task ids", "raw output", "fak output", "grader summary"]
}`)
	write(t, root, "experiments/agent-live/terminalbench-official-run-contract-20260626.json", `{
  "status": "READY_FOR_EXTERNAL_HARNESS",
  "result_claim_allowed": false,
  "arms": [
    {"name": "raw-terminalbench", "command": "raw"},
    {"name": "fak-terminalbench", "command": "fak"}
  ],
  "required_before_claim": ["official task ids", "raw run dir", "fak run dir", "test summaries"]
}`)
	write(t, root, "experiments/agent-live/browseraction-official-run-contract-20260626.json", `{
  "status": "READY_FOR_EXTERNAL_HARNESS",
  "result_claim_allowed": false,
  "arms": [
    {"name": "raw-browseraction", "command": "raw"},
    {"name": "fak-browseraction", "command": "fak"}
  ],
  "required_before_claim": ["official task ids", "raw traces", "fak traces", "score reports"]
}`)
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

func queueItem(q *ExternalHarnessQueue, issue int) ExternalHarnessQueueItem {
	for _, item := range q.Items {
		if item.Issue == issue {
			return item
		}
	}
	return ExternalHarnessQueueItem{}
}
