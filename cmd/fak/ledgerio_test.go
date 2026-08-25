package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAppendReportHistoryPreservesCommandContract(t *testing.T) {
	t.Run("disabled", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "history.jsonl")
		called := false
		code := appendReportHistory(&bytes.Buffer{}, &bytes.Buffer{}, false, true, filepath.Dir(path), path,
			"sample report", "sample", "row", func(string) (string, error) {
				called = true
				return "", nil
			})
		if code != 0 || called {
			t.Fatalf("disabled append = (code %d, renderer called %v), want (0, false)", code, called)
		}
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("disabled append materialized %q: %v", path, err)
		}
	})

	t.Run("append and announce", func(t *testing.T) {
		root := t.TempDir()
		path := filepath.Join(root, "docs", "history.jsonl")
		var stdout, stderr bytes.Buffer
		code := appendReportHistory(&stdout, &stderr, true, true, root, path,
			"sample report", "sample", "row", func(value string) (string, error) { return value, nil })
		if code != 0 {
			t.Fatalf("append code = %d, stderr = %q", code, stderr.String())
		}
		body, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if got, want := string(body), "row\n"; got != want {
			t.Fatalf("ledger = %q, want %q", got, want)
		}
		if got := filepath.ToSlash(stdout.String()); !strings.Contains(got, "appended sample row -> docs/history.jsonl") {
			t.Fatalf("announcement = %q", got)
		}
	})

	t.Run("render failure", func(t *testing.T) {
		root := t.TempDir()
		path := filepath.Join(root, "history.jsonl")
		var stderr bytes.Buffer
		code := appendReportHistory(&bytes.Buffer{}, &stderr, true, false, root, path,
			"sample report", "sample", "row", func(string) (string, error) { return "", errors.New("boom") })
		if code != 1 {
			t.Fatalf("append code = %d, want 1", code)
		}
		if got, want := stderr.String(), "fak sample report: append ledger: boom\n"; got != want {
			t.Fatalf("stderr = %q, want %q", got, want)
		}
	})
}
