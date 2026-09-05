package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/codesearch"
)

func findCodesearchRepoRoot(t *testing.T) string {
	t.Helper()
	root := filepath.Clean(filepath.Join("..", ".."))
	if _, err := os.Stat(filepath.Join(root, "go.mod")); err == nil {
		return root
	}
	if _, err := os.Stat("go.mod"); err == nil {
		return "."
	}
	t.Fatalf("could not locate repository root containing go.mod")
	return ""
}

func TestCodesearchDogfood(t *testing.T) {
	root := findCodesearchRepoRoot(t)
	codesearchDir := filepath.Join(root, "internal", "codesearch")

	t.Run("grep", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := codesearch.Run([]string{"grep", "--root", codesearchDir, "func Run"}, &stdout, &stderr)
		if code != 0 {
			t.Fatalf("grep exit code %d, stderr: %s", code, stderr.String())
		}
		out := stdout.String()
		if !strings.Contains(out, "codesearch.go") {
			t.Errorf("grep 'func Run' over %s missed codesearch.go; got:\n%s", codesearchDir, out)
		}

		// Also verify a second regex pattern targeting WorkspaceIndex in workspace_index.go
		stdout.Reset()
		stderr.Reset()
		code = codesearch.Run([]string{"grep", "--root", codesearchDir, `type\s+WorkspaceIndex\s+struct`}, &stdout, &stderr)
		if code != 0 {
			t.Fatalf("grep exit code %d, stderr: %s", code, stderr.String())
		}
		out = stdout.String()
		if !strings.Contains(out, "workspace_index.go") {
			t.Errorf("grep 'type WorkspaceIndex struct' missed workspace_index.go; got:\n%s", out)
		}
	})

	t.Run("lit", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := codesearch.Run([]string{"lit", "--root", codesearchDir, "collectGoFiles"}, &stdout, &stderr)
		if code != 0 {
			t.Fatalf("lit exit code %d, stderr: %s", code, stderr.String())
		}
		out := stdout.String()
		if !strings.Contains(out, "codesearch.go") {
			t.Errorf("lit 'collectGoFiles' over %s missed codesearch.go; got:\n%s", codesearchDir, out)
		}

		// Verify literal substring search over workspace_index.go
		stdout.Reset()
		stderr.Reset()
		code = codesearch.Run([]string{"lit", "--root", codesearchDir, "type Result = trigram.Result"}, &stdout, &stderr)
		if code != 0 {
			t.Fatalf("lit exit code %d, stderr: %s", code, stderr.String())
		}
		out = stdout.String()
		if !strings.Contains(out, "workspace_index.go") {
			t.Errorf("lit 'type Result = trigram.Result' missed workspace_index.go; got:\n%s", out)
		}
	})

	t.Run("ast", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := codesearch.Run([]string{"ast", "--root", codesearchDir, "flag.NewFlagSet($_, $_)"}, &stdout, &stderr)
		if code != 0 {
			t.Fatalf("ast exit code %d, stderr: %s", code, stderr.String())
		}
		out := stdout.String()
		if !strings.Contains(out, "codesearch.go:") || !strings.Contains(out, "flag.NewFlagSet") {
			t.Errorf("ast query 'flag.NewFlagSet($_, $_)' did not match expected site in codesearch.go; got:\n%s", out)
		}

		stdout.Reset()
		stderr.Reset()
		code = codesearch.Run([]string{"ast", "--root", codesearchDir, "fmt.Fprintln($_, $_)"}, &stdout, &stderr)
		if code != 0 {
			t.Fatalf("ast exit code %d, stderr: %s", code, stderr.String())
		}
		out = stdout.String()
		if !strings.Contains(out, "codesearch.go:") || !strings.Contains(out, "fmt.Fprintln") {
			t.Errorf("ast query 'fmt.Fprintln($_, $_)' did not match expected site in codesearch.go; got:\n%s", out)
		}
	})
}
