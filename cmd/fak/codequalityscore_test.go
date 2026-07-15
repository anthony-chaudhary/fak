package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"
)

func writeCodeQualityCheckers(t *testing.T, root string) {
	t.Helper()
	for _, rel := range codeQualityCheckerPaths {
		path := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte("checker\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func TestCodeQualityScoreAcceptsUnchangedPinnedChecker(t *testing.T) {
	root := t.TempDir()
	writeCodeQualityCheckers(t, root)
	run := func(context.Context, string, []string) ([]byte, []byte, int, error) {
		return []byte(`{"ok":true}`), nil, 0, nil
	}
	var out, errOut bytes.Buffer
	if code := runCodeQualityScore(&out, &errOut, []string{"--workspace", root, "--json"}, run); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if out.String() != `{"ok":true}` {
		t.Fatalf("stdout=%q", out.String())
	}
}

func TestCodeQualityScoreRefusesCheckerDriftBeforeEmittingGrade(t *testing.T) {
	root := t.TempDir()
	writeCodeQualityCheckers(t, root)
	run := func(_ context.Context, _ string, _ []string) ([]byte, []byte, int, error) {
		path := filepath.Join(root, filepath.FromSlash(codeQualityCheckerPaths[0]))
		if err := os.WriteFile(path, []byte("weakened\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		return []byte(`{"ok":true,"forged":true}`), nil, 0, nil
	}
	var out, errOut bytes.Buffer
	if code := runCodeQualityScore(&out, &errOut, []string{"--workspace", root, "--json"}, run); code != 1 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if out.Len() != 0 {
		t.Fatalf("forged grade leaked before pin check: %q", out.String())
	}
	if !bytes.Contains(errOut.Bytes(), []byte("CHECKER_TAMPERED")) {
		t.Fatalf("stderr=%q", errOut.String())
	}
}
