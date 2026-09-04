package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/citeverify"
)

func TestCiteverifyCommand(t *testing.T) {
	root := t.TempDir()
	sourceFile := filepath.Join(root, "pkg", "sample.go")
	if err := os.MkdirAll(filepath.Dir(sourceFile), 0o755); err != nil {
		t.Fatal(err)
	}
	content := "package sample\nfunc MySpecialFunc() {}\n"
	if err := os.WriteFile(sourceFile, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name       string
		args       []string
		wantCode   int
		wantStdout string
	}{
		{
			name:       "supports exact line",
			args:       []string{"--root", root, "--claim", "`MySpecialFunc` exists", "pkg/sample.go:2"},
			wantCode:   0,
			wantStdout: "supports\n",
		},
		{
			name:       "contradicts wrong line",
			args:       []string{"--root", root, "--claim", "`MySpecialFunc` exists", "pkg/sample.go:1"},
			wantCode:   3,
			wantStdout: "contradicts\n",
		},
		{
			name:       "unknown file",
			args:       []string{"--root", root, "--claim", "`MySpecialFunc` exists", "pkg/missing.go:1"},
			wantCode:   1,
			wantStdout: "unknown\n",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			code := runCiteverify(&stdout, &stderr, tc.args)
			if code != tc.wantCode {
				t.Fatalf("runCiteverify() = %d, want %d; stderr: %s", code, tc.wantCode, stderr.String())
			}
			if stdout.String() != tc.wantStdout {
				t.Fatalf("runCiteverify() stdout = %q, want %q", stdout.String(), tc.wantStdout)
			}
		})
	}

	// Test JSON output
	var stdout, stderr bytes.Buffer
	code := runCiteverify(&stdout, &stderr, []string{
		"--root", root,
		"--claim", "`MySpecialFunc` exists",
		"--json",
		"pkg/sample.go:2",
	})
	if code != 0 {
		t.Fatalf("JSON runCiteverify() code = %d, stderr: %s", code, stderr.String())
	}
	var res citeverifyJSONResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("json unmarshal: %v", err)
	}
	if res.Status != citeverify.Supports {
		t.Fatalf("expected supports, got %s", res.Status)
	}

	// Test usage on empty
	stdout.Reset()
	stderr.Reset()
	code = runCiteverify(&stdout, &stderr, []string{})
	if code != 2 {
		t.Fatalf("expected usage exit code 2, got %d", code)
	}
	if !strings.Contains(stderr.String(), "fak citeverify — mechanically verify") {
		t.Fatalf("expected usage text, got: %s", stderr.String())
	}
}
