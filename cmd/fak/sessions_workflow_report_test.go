package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestSessionsWorkflowDefaultReportCountsAuthoredEvidenceHonestly(t *testing.T) {
	home := t.TempDir()
	for _, witness := range []codexWorkflowDefaultWitness{
		{Schema: "fak.codex_workflow_default.v1", SessionID: "a", Classification: "consider-workflow", Decision: "inject"},
		{Schema: "fak.codex_workflow_default.v1", SessionID: "b", Classification: "likely-direct", Decision: "inject"},
	} {
		if err := writeCodexWorkflowDefaultWitness(home, witness); err != nil {
			t.Fatal(err)
		}
	}
	if err := writeCodexGuardWitness(home, "a"); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(home, "fak-workflow-defaults")
	if err := os.WriteFile(filepath.Join(dir, "broken.json"), []byte("{"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "ignore.txt"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	if code := runSessionsWorkflowDefaultReport(&stdout, &stderr, []string{"--codex-home", home, "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
	var got codexWorkflowDefaultReport
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.WitnessFiles != 3 || got.ValidDecisions != 2 || got.Malformed != 1 || got.GuardJoined != 1 {
		t.Fatalf("report=%+v", got)
	}
	if got.Classifications["consider-workflow"] != 1 || got.Classifications["likely-direct"] != 1 || got.Decisions["inject"] != 2 {
		t.Fatalf("counts=%+v %+v", got.Classifications, got.Decisions)
	}
	if got.ObservedUse != 0 || got.UnknownOutcome != 2 {
		t.Fatalf("observed=%d unknown=%d; injection must not imply use", got.ObservedUse, got.UnknownOutcome)
	}
}

func TestSessionsWorkflowDefaultReportHandlesNoWitnessDirectory(t *testing.T) {
	report, err := collectCodexWorkflowDefaultReport(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if report.WitnessFiles != 0 || report.ValidDecisions != 0 || report.UnknownOutcome != 0 {
		t.Fatalf("report=%+v", report)
	}
}

func TestSessionsWorkflowDefaultReportJoinsOrchestrationInvocation(t *testing.T) {
	home := t.TempDir()
	witness := codexWorkflowDefaultWitness{Schema: "fak.codex_workflow_default.v1", SessionID: "joined", Classification: "consider-workflow", Decision: "inject"}
	if err := writeCodexWorkflowDefaultWitness(home, witness); err != nil {
		t.Fatal(err)
	}
	if err := writeCodexOrchestrationInvocationReceipt(home, codexOrchestrationInvocationReceipt{
		Schema: "fak.codex_orchestration_invocation.v1", SessionID: "joined", Resolved: "ultracode", MaxWorkers: 4,
	}); err != nil {
		t.Fatal(err)
	}
	report, err := collectCodexWorkflowDefaultReport(home)
	if err != nil {
		t.Fatal(err)
	}
	if report.ObservedUse != 1 || report.UnknownOutcome != 0 || report.WorkerLaunches != 0 {
		t.Fatalf("report=%+v; invocation is observed use but not a worker launch", report)
	}
}
