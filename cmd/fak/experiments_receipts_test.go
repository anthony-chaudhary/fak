package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/experiments"
)

func TestExperimentsRecordLookupRoundTrip(t *testing.T) {
	dir := t.TempDir()
	receiptPath := filepath.Join(dir, "receipt.json")
	identityPath := filepath.Join(dir, "identity.json")
	ledgerPath := filepath.Join(dir, "receipts.jsonl")
	environment := "round-trip CPU-reference"
	artifactDigest := cmdTestDigest("round-trip artifact")
	receipt := map[string]any{
		"id": "round-trip", "hypothesis": "batch improves aggregate rate", "verdict": "lost",
		"baseline": "sequential", "candidate": "compatibility batch",
		"metric":   map[string]any{"name": "aggregate rate", "unit": "token_steps/s", "baseline_value": 5.858, "candidate_value": 3.159},
		"revision": "git:abc", "environment": environment,
		"environment_digest": experiments.DigestEnvironment(environment), "artifact_digest": artifactDigest,
		"scope": "CPU-reference fixture", "next_action": "run the native CUDA path",
	}
	writeExperimentJSON(t, receiptPath, receipt)
	identity := experiments.ReceiptIdentity{
		Hypothesis: "batch improves aggregate rate", Revision: "git:abc",
		Environment: environment, EnvironmentDigest: experiments.DigestEnvironment(environment), ArtifactDigest: artifactDigest,
	}
	writeExperimentJSON(t, identityPath, identity)

	var stdout, stderr bytes.Buffer
	if code := runExperiments(&stdout, &stderr, []string{"record", "--file", receiptPath, "--ledger", ledgerPath, "--json"}); code != 0 {
		t.Fatalf("record code=%d stderr=%s", code, stderr.String())
	}
	var recorded experiments.Receipt
	if err := json.Unmarshal(stdout.Bytes(), &recorded); err != nil {
		t.Fatalf("decode record response: %v\n%s", err, stdout.String())
	}
	if recorded.Schema != experiments.ReceiptSchema || recorded.RecordedAt == "" ||
		recorded.Environment != environment || recorded.EnvironmentDigest != experiments.DigestEnvironment(environment) ||
		recorded.ArtifactDigest != artifactDigest || recorded.EvidenceClass != experiments.EvidenceClassScreening {
		t.Fatalf("record response omitted generated fields or provenance: %#v", recorded)
	}
	if lines := strings.Count(strings.TrimSpace(string(mustReadExperimentFile(t, ledgerPath))), "\n") + 1; lines != 1 {
		t.Fatalf("record should append exactly one JSONL line, got %d", lines)
	}

	stdout.Reset()
	stderr.Reset()
	if code := runExperiments(&stdout, &stderr, []string{"lookup", "--file", identityPath, "--ledger", ledgerPath, "--json"}); code != 0 {
		t.Fatalf("lookup code=%d stderr=%s", code, stderr.String())
	}
	var result experiments.LookupResult
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("decode lookup: %v\n%s", err, stdout.String())
	}
	if result.Status != experiments.LookupExact || !result.MeasuredLoss || result.ClaimEligible || result.Receipt == nil {
		t.Fatalf("lookup result = %#v", result)
	}
	if result.Receipt.EvidenceClass != experiments.EvidenceClassScreening {
		t.Fatalf("evidence class = %q", result.Receipt.EvidenceClass)
	}
}

func TestExperimentsLookupDoesNotRenderInconclusiveAsLoss(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, "receipts.jsonl")
	receipt := experiments.Receipt{
		Schema: experiments.ReceiptSchema, ID: "probe", RecordedAt: "2026-08-27T12:00:00Z",
		EvidenceClass: experiments.EvidenceClassScreening,
		Hypothesis:    "fusion improves latency", Verdict: experiments.VerdictInconclusive,
		Revision: "git:def", Environment: "interrupted test host",
		EnvironmentDigest: experiments.DigestEnvironment("interrupted test host"), ArtifactDigest: cmdTestDigest("interrupted artifact"),
		Scope: "one interrupted probe", Reason: "device reset before comparison", NextAction: "rerun on sanctioned node",
	}
	if err := experiments.AppendReceipt(ledger, receipt); err != nil {
		t.Fatal(err)
	}
	queryPath := filepath.Join(dir, "identity.json")
	writeExperimentJSON(t, queryPath, experiments.ReceiptIdentity{
		Hypothesis: receipt.Hypothesis, Revision: receipt.Revision,
		Environment:       receipt.Environment,
		EnvironmentDigest: receipt.EnvironmentDigest, ArtifactDigest: receipt.ArtifactDigest,
	})
	var stdout, stderr bytes.Buffer
	if code := runExperiments(&stdout, &stderr, []string{"lookup", "--file", queryPath, "--ledger", ledger}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "not a measured loss") || !strings.Contains(stdout.String(), "claim eligible: false") {
		t.Fatalf("human output must preserve evidence boundary:\n%s", stdout.String())
	}
}

func TestExperimentsRecordHumanOutputIncludesIdentity(t *testing.T) {
	dir := t.TempDir()
	input := filepath.Join(dir, "receipt.json")
	environment := "human-output host"
	receipt := experiments.Receipt{
		ID: "human", Hypothesis: "probe is useful", Verdict: experiments.VerdictInconclusive,
		Revision: "git:human", Environment: environment, EnvironmentDigest: experiments.DigestEnvironment(environment),
		ArtifactDigest: cmdTestDigest("human artifact"), Scope: "one probe", Reason: "not enough samples", NextAction: "collect another sample",
	}
	writeExperimentJSON(t, input, receipt)
	var stdout, stderr bytes.Buffer
	if code := runExperiments(&stdout, &stderr, []string{"record", "--file", input, "--ledger", filepath.Join(dir, "receipts.jsonl")}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	for _, want := range []string{"Revision: git:human", "Environment: " + environment, "Environment digest: " + receipt.EnvironmentDigest, "Artifact digest: " + receipt.ArtifactDigest} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("human record output missing %q:\n%s", want, stdout.String())
		}
	}
}

func cmdTestDigest(value string) string {
	sum := sha256.Sum256([]byte(value))
	return "sha256:" + hex.EncodeToString(sum[:])
}

func writeExperimentJSON(t *testing.T, path string, value any) {
	t.Helper()
	b, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

func mustReadExperimentFile(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
