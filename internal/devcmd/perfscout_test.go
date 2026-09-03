package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

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
	code := RunPerfScout(&stdout, &stderr, []string{"-fixture", fixFile, "-json"})
	if code != 0 {
		t.Fatalf("expected 0, got %d. stderr: %s", code, stderr.String())
	}

	if !strings.Contains(stdout.String(), "indie/qwen38-fast") {
		t.Errorf("expected output to contain indie/qwen38-fast, got: %s", stdout.String())
	}
}
