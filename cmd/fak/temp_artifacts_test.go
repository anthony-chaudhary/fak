package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/tempartifact"
)

func TestTempArtifactsRequiresExplicitPositiveMinAge(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runTempArtifacts(&stdout, &stderr, []string{"--json"}); code != 2 {
		t.Fatalf("exit = %d, stderr = %q", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "--min-age is required") {
		t.Fatalf("stderr = %q", stderr.String())
	}
}

func TestTempArtifactsDefaultIsReadOnlyStableJSON(t *testing.T) {
	root := t.TempDir()
	setProcessTempRoot(t, root)
	path := filepath.Join(root, "fak-preview.zip")
	if err := os.WriteFile(path, []byte("preview"), 0o600); err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-2 * time.Hour)
	if err := os.Chtimes(path, old, old); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runTempArtifacts(&stdout, &stderr, []string{"--min-age", "1h", "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, stderr.String())
	}
	var report tempartifact.Report
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode %q: %v", stdout.String(), err)
	}
	if report.Schema != tempartifact.Schema || report.Mode != "preview" || report.Root != filepath.Clean(root) {
		t.Fatalf("report header = %+v", report)
	}
	if len(report.Items) != 1 || report.Items[0].Path != path {
		t.Fatalf("items = %+v", report.Items)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("default preview removed candidate: %v", err)
	}
}

func TestTempArtifactsHumanReportCarriesPathAgeBytesReasonAndAggregates(t *testing.T) {
	root := t.TempDir()
	setProcessTempRoot(t, root)
	path := filepath.Join(root, "fak-human.tar")
	if err := os.WriteFile(path, []byte("human"), 0o600); err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-2 * time.Hour)
	if err := os.Chtimes(path, old, old); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	if code := runTempArtifacts(&stdout, &stderr, []string{"--min-age", "1h"}); code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, stderr.String())
	}
	for _, want := range []string{path, "age=", "bytes=5", "reason=", "summary matching=1/5B", "reaped=0/0B"} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("human report missing %q: %s", want, stdout.String())
		}
	}
}

func setProcessTempRoot(t *testing.T, root string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Setenv("TEMP", root)
		t.Setenv("TMP", root)
		return
	}
	t.Setenv("TMPDIR", root)
}
