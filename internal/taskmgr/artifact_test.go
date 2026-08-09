package taskmgr

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestArtifactEvidenceBoundsRedactsAndReturnsPathRef(t *testing.T) {
	if os.Getenv("FAK_TASKMGR_ARTIFACT_HELPER") == "1" {
		_, _ = os.Stdout.WriteString("api_key=super-secret\n" + strings.Repeat("x", 512))
		os.Exit(7)
	}
	t.Setenv("FAK_TASKMGR_ARTIFACT_HELPER", "1")

	path := filepath.Join(t.TempDir(), "evidence", "failure.txt")
	result, err := CaptureCommandArtifact(context.Background(), ArtifactCommand{
		Argv:         []string{os.Args[0], "-test.run=TestArtifactEvidenceBoundsRedactsAndReturnsPathRef"},
		ArtifactPath: path,
		MaxBytes:     80,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.ExitCode != 7 || !result.Truncated {
		t.Fatalf("result = %+v, want exit 7 and truncated", result)
	}
	if result.Evidence.Kind != "path" || result.Evidence.Ref != filepath.Clean(path) {
		t.Fatalf("evidence = %+v", result.Evidence)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	text := string(got)
	if strings.Contains(text, "super-secret") {
		t.Fatalf("artifact leaked secret: %q", text)
	}
	if !strings.Contains(text, "api_key=[REDACTED]") || !strings.Contains(text, "truncated at 80 bytes") {
		t.Fatalf("artifact missing redaction/truncation witness: %q", text)
	}
}

func TestArtifactEvidenceRejectsMissingDurablePath(t *testing.T) {
	_, err := CaptureCommandArtifact(context.Background(), ArtifactCommand{Argv: []string{"ignored"}})
	if err == nil || !strings.Contains(err.Error(), "artifact path is required") {
		t.Fatalf("error = %v", err)
	}
}
