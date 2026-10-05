package devcmd

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

func TestRunPerfScoutFixture(t *testing.T) {
	fixture := []perfscout.GitHubRawRepo{
		{
			FullName:        "indie/qwen38-fast",
			Description:     "Qwen3.8-Flash-Next at 262K context on 4x RTX 3090 with vLLM: 118 tok/s single-stream",
			URL:             "https://github.com/indie/qwen38-fast",
			StargazersCount: 2,
			UpdatedAt:       "2026-09-03T10:00:00Z",
			PushedAt:        "2026-09-03T10:00:00Z",
			CreatedAt:       "2026-08-15T00:00:00Z",
			Language:        "Python",
		},
	}

	tmpDir := t.TempDir()
	fixFile := filepath.Join(tmpDir, "fix.json")
	data, _ := json.Marshal(fixture)
	_ = os.WriteFile(fixFile, data, 0o644)

	var stdout, stderr bytes.Buffer
	code := runPerfScoutAt(&stdout, &stderr, []string{"-fixture", fixFile, "-json"}, time.Date(2026, time.September, 3, 12, 0, 0, 0, time.UTC))
	if code != 0 {
		t.Fatalf("expected 0, got %d. stderr: %s", code, stderr.String())
	}

	if !strings.Contains(stdout.String(), "indie/qwen38-fast") {
		t.Errorf("expected output to contain indie/qwen38-fast, got: %s", stdout.String())
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestRunPerfScoutFixtureFreshnessBoundary(t *testing.T) {
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
			if code := runPerfScoutAt(&stdout, &stderr, []string{"-fixture", fixturePath, "-json"}, now); code != 0 {
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
