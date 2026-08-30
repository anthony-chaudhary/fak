package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

func TestAttributionNightlyWorkflowEntrypointPopulatedWitness(t *testing.T) {
	root := filepath.Join("..", "..")
	workflow := mustReadTrajectoryNightly(t, filepath.Join(root, ".github", "workflows", "trajectory-attribution-nightly.yml"))
	for _, want := range []string{"schedule:", "- cron:", "populated workflow-entrypoint", "bash .github/scripts/trajectory-attribution-nightly.sh", "FAK_TRAJECTORY_CLAUDE_ROOT:", "FAK_TRAJECTORY_CODEX_ROOT:", "root_present=false"} {
		if !strings.Contains(string(workflow), want) {
			t.Fatalf("workflow missing %q:\n%s", want, workflow)
		}
	}
	temp := t.TempDir()
	receiptPath := filepath.Join(temp, "receipt.json")
	historyPath := filepath.Join(temp, "history.jsonl")
	at := time.Date(2026, 8, 21, 22, 0, 0, 0, time.UTC)
	claudeRoot, codexRoot := stageTrajectoryNightlyFixtures(t, root, temp, at.Add(-time.Minute))
	command := exec.Command("bash", filepath.Join(".github", "scripts", "trajectory-attribution-nightly.sh"))
	command.Dir = root
	command.Env = append(os.Environ(),
		"FAK_TRAJECTORY_BUDGET="+mustAbsTrajectoryNightly(t, filepath.Join(root, "configs", "trajectory-attribution-nightly.v1.json")),
		"FAK_TRAJECTORY_HISTORY="+mustAbsTrajectoryNightly(t, historyPath),
		"FAK_TRAJECTORY_RECEIPT="+mustAbsTrajectoryNightly(t, receiptPath),
		"FAK_TRAJECTORY_CORPUS=fleet",
		"FAK_TRAJECTORY_CLAUDE_ROOT="+mustAbsTrajectoryNightly(t, claudeRoot),
		"FAK_TRAJECTORY_CODEX_ROOT="+mustAbsTrajectoryNightly(t, codexRoot),
		"FAK_TRAJECTORY_AT="+at.Format(time.RFC3339Nano),
	)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("workflow entrypoint: %v\n%s", err, output)
	}
	var receipt trajectory.AttributionReceipt
	if err := json.Unmarshal(mustReadTrajectoryNightly(t, receiptPath), &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Status != trajectory.AttributionStatusPass || len(receipt.Coverage) != 2 || receipt.Coverage[0].FilesScanned+receipt.Coverage[1].FilesScanned != 2 {
		t.Fatalf("receipt=%+v output=%s", receipt, output)
	}
	if rows := bytes.Count(bytes.TrimSpace(mustReadTrajectoryNightly(t, historyPath)), []byte{'\n'}) + 1; rows != 1 {
		t.Fatalf("history rows=%d, want 1", rows)
	}
}

func TestAttributionNightlyRejectsSharedHistoryAndReceipt(t *testing.T) {
	shared := filepath.Join(t.TempDir(), "shared.jsonl")
	var stdout, stderr bytes.Buffer
	rc := runTrajectory(&stdout, &stderr, []string{
		"nightly", "--budget", filepath.Join("..", "..", "configs", "trajectory-attribution-nightly.v1.json"),
		"--history", shared, "--receipt", shared,
	})
	if rc != 2 || !strings.Contains(stderr.String(), "must name different files") {
		t.Fatalf("rc=%d stderr=%q", rc, stderr.String())
	}
	if _, err := os.Stat(shared); !os.IsNotExist(err) {
		t.Fatalf("shared output unexpectedly exists: %v", err)
	}
}

func TestAttributionNightlyHistoryFailurePublishesTypedFailure(t *testing.T) {
	originalAppend := appendPreparedTrajectoryNightlyReceipt
	appendPreparedTrajectoryNightlyReceipt = func(string, *trajectory.AttributionReceipt) (func() error, error) {
		return nil, errors.New("injected history failure")
	}
	t.Cleanup(func() { appendPreparedTrajectoryNightlyReceipt = originalAppend })
	temp := t.TempDir()
	receiptPath := filepath.Join(temp, "receipt.json")
	var stdout, stderr bytes.Buffer
	rc := runTrajectory(&stdout, &stderr, []string{
		"nightly", "--budget", filepath.Join("..", "..", "configs", "trajectory-attribution-nightly.v1.json"),
		"--history", filepath.Join(temp, "history.jsonl"), "--receipt", receiptPath,
		"--claude-root", filepath.Join(temp, "missing-claude"), "--codex-root", filepath.Join(temp, "missing-codex"),
	})
	if rc != 1 {
		t.Fatalf("rc=%d stderr=%q", rc, stderr.String())
	}
	var receipt trajectory.AttributionReceipt
	if err := json.Unmarshal(mustReadTrajectoryNightly(t, receiptPath), &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Status != trajectory.AttributionStatusPublicationFailed || receipt.PublicationError != "history_append_failed" || receipt.CollectionError != "" {
		t.Fatalf("receipt=%+v", receipt)
	}
	if _, err := os.Stat(filepath.Join(temp, "history.jsonl")); !os.IsNotExist(err) {
		t.Fatalf("history unexpectedly exists: %v", err)
	}
}

func TestAttributionNightlyReceiptFailureDoesNotAppendHistory(t *testing.T) {
	temp := t.TempDir()
	receiptDir := filepath.Join(temp, "receipt-dir")
	if err := os.Mkdir(receiptDir, 0o700); err != nil {
		t.Fatal(err)
	}
	historyPath := filepath.Join(temp, "history.jsonl")
	var stdout, stderr bytes.Buffer
	rc := runTrajectory(&stdout, &stderr, []string{
		"nightly", "--budget", filepath.Join("..", "..", "configs", "trajectory-attribution-nightly.v1.json"),
		"--history", historyPath, "--receipt", receiptDir,
		"--claude-root", filepath.Join(temp, "missing-claude"), "--codex-root", filepath.Join(temp, "missing-codex"),
	})
	if rc != 1 || !strings.Contains(stderr.String(), "receipt_publish_failed") {
		t.Fatalf("rc=%d stderr=%q", rc, stderr.String())
	}
	if _, err := os.Stat(historyPath); !os.IsNotExist(err) {
		t.Fatalf("history unexpectedly exists: %v", err)
	}
}

func TestAttributionNightlyInjectedBudgetFailureWitness(t *testing.T) {
	root := filepath.Join("..", "..")
	budget, err := trajectory.ReadAttributionBudget(filepath.Join(root, "configs", "trajectory-attribution-nightly.v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	budget.MaxDuplicateEvents = 0
	temp := t.TempDir()
	budgetPath := filepath.Join(temp, "injected-budget.json")
	encoded, err := json.Marshal(budget)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(budgetPath, encoded, 0o600); err != nil {
		t.Fatal(err)
	}
	receiptPath := filepath.Join(temp, "receipt.json")
	at := time.Date(2026, 8, 21, 22, 0, 0, 0, time.UTC)
	claudeRoot, _ := stageTrajectoryNightlyFixtures(t, root, temp, at.Add(-time.Minute))
	var stdout, stderr bytes.Buffer
	rc := runTrajectory(&stdout, &stderr, []string{
		"nightly", "--budget", budgetPath, "--history", filepath.Join(temp, "history.jsonl"),
		"--receipt", receiptPath, "--corpus", "local",
		"--claude-root", claudeRoot,
		"--codex-root", filepath.Join(temp, "missing-codex"), "--at", at.Format(time.RFC3339Nano),
	})
	if rc != 3 {
		t.Fatalf("rc=%d, want budget failure 3; stderr=%s", rc, stderr.String())
	}
	var receipt trajectory.AttributionReceipt
	if err := json.Unmarshal(mustReadTrajectoryNightly(t, receiptPath), &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Status != trajectory.AttributionStatusBudgetFailed || len(receipt.Breaches) != 1 || receipt.Breaches[0].Metric != "duplicate_events" {
		t.Fatalf("receipt=%+v", receipt)
	}
	foundCoordinate := false
	for _, sample := range receipt.Samples {
		if sample.Metric == "duplicate_events" && sample.SourcePath != "" {
			foundCoordinate = true
		}
	}
	if !foundCoordinate {
		t.Fatalf("budget regression lacks a bounded content-free coordinate: %+v", receipt.Samples)
	}
	if strings.Contains(string(mustReadTrajectoryNightly(t, receiptPath)), "audit the fixture") {
		t.Fatal("receipt leaked transcript content")
	}
}

func stageTrajectoryNightlyFixtures(t *testing.T, repoRoot, tempRoot string, modTime time.Time) (string, string) {
	t.Helper()
	sourceRoot := filepath.Join(repoRoot, "internal", "trajectory", "testdata", "audit")
	claudeRoot := filepath.Join(tempRoot, "corpus", "claude", "projects")
	codexRoot := filepath.Join(tempRoot, "corpus", "codex", "sessions")
	fixtures := []struct {
		source string
		target string
	}{
		{source: filepath.Join(sourceRoot, "claude", "projects", "fak", "claude-session.jsonl"), target: filepath.Join(claudeRoot, "fak", "claude-session.jsonl")},
		{source: filepath.Join(sourceRoot, "codex", "sessions", "2026", "08", "21", "codex-session.jsonl"), target: filepath.Join(codexRoot, "2026", "08", "21", "codex-session.jsonl")},
	}
	for _, fixture := range fixtures {
		contents := mustReadTrajectoryNightly(t, fixture.source)
		if err := os.MkdirAll(filepath.Dir(fixture.target), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(fixture.target, contents, 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(fixture.target, modTime, modTime); err != nil {
			t.Fatal(err)
		}
	}
	return claudeRoot, codexRoot
}

func mustReadTrajectoryNightly(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func mustAbsTrajectoryNightly(t *testing.T, path string) string {
	t.Helper()
	abs, err := filepath.Abs(path)
	if err != nil {
		t.Fatal(err)
	}
	return abs
}
