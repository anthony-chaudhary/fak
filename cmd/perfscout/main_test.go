package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

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
	code := runAt(&stdout, &stderr, []string{
		"-fixture", fixFile,
		"-out", outFile,
		"-cohorts", "2",
	}, time.Date(2026, time.September, 3, 12, 0, 0, 0, time.UTC))
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
	code := runAt(&stdout, &stderr, []string{
		"-fixture", fixFile,
		"-json",
	}, time.Date(2026, time.September, 3, 12, 0, 0, 0, time.UTC))
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

func TestCLICrossArchitectureMatrix(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := run(&stdout, &stderr, []string{
		"-cross",
		"-source", "mlx",
	})
	if code != 0 {
		t.Fatalf("expected 0, got %d. stderr: %s", code, stderr.String())
	}
	outStr := stdout.String()
	if !strings.Contains(outStr, "XINNOV-01") || !strings.Contains(outStr, "slotstream") {
		t.Errorf("expected output to contain XINNOV-01 and slotstream, got:\n%s", outStr)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestCLIFixtureFreshnessBoundary(t *testing.T) {
	updated := time.Date(2026, time.September, 3, 12, 0, 0, 0, time.UTC)
	fixture := []perfscout.GitHubRawRepo{{
		FullName:        "author/qwen-freshness",
		Description:     "Qwen3.8-Flash-Next on DGX Spark GB10 with vLLM: measured 52 tok/s decode",
		URL:             "https://github.com/author/qwen-freshness",
		StargazersCount: 5,
		UpdatedAt:       updated.Format(time.RFC3339),
		PushedAt:        updated.Format(time.RFC3339),
		CreatedAt:       "2026-08-15T00:00:00Z",
		Language:        "Python",
	}}
	data, err := json.Marshal(fixture)
	if err != nil {
		t.Fatal(err)
	}
	fixturePath := filepath.Join(t.TempDir(), "freshness.json")
	if err := os.WriteFile(fixturePath, data, 0o644); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		days int
		want int
	}{
		{name: "30 days included", days: 30, want: 1},
		{name: "31 days expired", days: 31, want: 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			now := updated.Add(time.Duration(tc.days) * 24 * time.Hour)
			var stdout, stderr bytes.Buffer
			if code := runAt(&stdout, &stderr, []string{"-fixture", fixturePath, "-json"}, now); code != 0 {
				t.Fatalf("exit = %d, stderr: %s", code, stderr.String())
			}
			var report perfscout.InventoryReport
			if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
				t.Fatal(err)
			}
			if report.RetainedCount != tc.want || len(report.Repositories) != tc.want {
				t.Fatalf("%d-day fixture retained %d repos (%d rows), want %d", tc.days, report.RetainedCount, len(report.Repositories), tc.want)
			}
			if !report.GeneratedAt.Equal(now) {
				t.Fatalf("generated at %s, want injected clock %s", report.GeneratedAt, now)
			}
		})
	}
}
