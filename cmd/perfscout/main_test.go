package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/perfscout"
)

func TestCLIFixtureMarkdown(t *testing.T) {
	fixture := []perfscout.GitHubRawRepo{
		{
			FullName:        "author/qwen-nitro",
			Description:     "Qwen3.8-Flash-Next on DGX Spark GB10 with vLLM: measured 52 tok/s decode, MTP k=2",
			URL:             "https://github.com/author/qwen-nitro",
			StargazersCount: 5,
			UpdatedAt:       "2026-09-03T10:00:00Z",
			PushedAt:        "2026-09-03T10:00:00Z",
			CreatedAt:       "2026-08-15T00:00:00Z",
			Language:        "Python",
		},
		{
			FullName:        "author/glm-turbo",
			Description:     "GLM-5.3-Flash at 64 tok/s on DGX Spark: EXL3 2.05bpw + DFlash2",
			URL:             "https://github.com/author/glm-turbo",
			StargazersCount: 3,
			UpdatedAt:       "2026-09-03T09:00:00Z",
			PushedAt:        "2026-09-03T09:00:00Z",
			CreatedAt:       "2026-08-20T00:00:00Z",
			Language:        "Python",
		},
	}

	tmpDir := t.TempDir()
	fixFile := filepath.Join(tmpDir, "fix.json")
	fixBytes, err := json.Marshal(fixture)
	if err != nil {
		t.Fatalf("marshal fixture: %v", err)
	}
	if err := os.WriteFile(fixFile, fixBytes, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	outFile := filepath.Join(tmpDir, "report.md")
	var stdout, stderr bytes.Buffer
	code := run(&stdout, &stderr, []string{
		"-fixture", fixFile,
		"-out", outFile,
		"-cohorts", "2",
	})
	if code != 0 {
		t.Fatalf("expected 0, got %d. stderr: %s", code, stderr.String())
	}

	reportBytes, err := os.ReadFile(outFile)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	reportStr := string(reportBytes)
	if !strings.Contains(reportStr, "author/qwen-nitro") || !strings.Contains(reportStr, "author/glm-turbo") {
		t.Errorf("expected report to contain both repos, got:\n%s", reportStr)
	}
}

func TestCLIJSONOutput(t *testing.T) {
	fixture := []perfscout.GitHubRawRepo{
		{
			FullName:        "author/qwen-nitro",
			Description:     "Qwen3.8-Flash-Next on DGX Spark GB10 with vLLM: measured 52 tok/s decode",
			URL:             "https://github.com/author/qwen-nitro",
			StargazersCount: 5,
			UpdatedAt:       "2026-09-03T10:00:00Z",
			PushedAt:        "2026-09-03T10:00:00Z",
			CreatedAt:       "2026-08-15T00:00:00Z",
			Language:        "Python",
		},
	}

	tmpDir := t.TempDir()
	fixFile := filepath.Join(tmpDir, "fix.json")
	fixBytes, _ := json.Marshal(fixture)
	_ = os.WriteFile(fixFile, fixBytes, 0o644)

	var stdout, stderr bytes.Buffer
	code := run(&stdout, &stderr, []string{
		"-fixture", fixFile,
		"-json",
	})
	if code != 0 {
		t.Fatalf("expected 0, got %d. stderr: %s", code, stderr.String())
	}

	var report perfscout.InventoryReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("failed to decode JSON output: %v", err)
	}
	if report.RetainedCount != 1 {
		t.Errorf("expected 1 retained, got %d", report.RetainedCount)
	}
}
