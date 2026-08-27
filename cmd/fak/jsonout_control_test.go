package main

import (
	"bytes"
	"errors"
	"testing"
)

type jsonOutputFailWriter struct{}

func (jsonOutputFailWriter) Write([]byte) (int, error) {
	return 0, errors.New("write failed")
}

func TestEmitJSONOrPrintlnPreservesOutputContract(t *testing.T) {
	t.Run("json", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := emitJSONOrPrintln(&stdout, &stderr, "fak representative", true,
			struct {
				OK bool `json:"ok"`
			}{OK: true},
			"human bytes",
		)

		if code != 0 {
			t.Fatalf("code = %d, want 0", code)
		}
		if got, want := stdout.String(), "{\n  \"ok\": true\n}\n"; got != want {
			t.Fatalf("stdout = %q, want %q", got, want)
		}
		if stderr.Len() != 0 {
			t.Fatalf("stderr = %q, want empty", stderr.String())
		}
	})

	t.Run("human", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := emitJSONOrPrintln(&stdout, &stderr, "fak representative", false, map[string]bool{"ok": true}, "human bytes")

		if code != 0 {
			t.Fatalf("code = %d, want 0", code)
		}
		if got, want := stdout.String(), "human bytes\n"; got != want {
			t.Fatalf("stdout = %q, want %q", got, want)
		}
		if stderr.Len() != 0 {
			t.Fatalf("stderr = %q, want empty", stderr.String())
		}
	})

	t.Run("write failure", func(t *testing.T) {
		var stderr bytes.Buffer
		code := emitJSONOrPrintln(jsonOutputFailWriter{}, &stderr, "fak representative", true, map[string]bool{"ok": true}, "human bytes")

		if code != 1 {
			t.Fatalf("code = %d, want 1", code)
		}
		if got, want := stderr.String(), "fak representative: encode json: write failed\n"; got != want {
			t.Fatalf("stderr = %q, want %q", got, want)
		}
	})
}

func TestCodexLoopIssueJSONWriteFailurePreservesCommandError(t *testing.T) {
	var stderr bytes.Buffer
	code := runCodexLoopSyncIssues(jsonOutputFailWriter{}, &stderr, codexLoopRecentReport{}, true, codexLoopIssueOptions{})

	if code != 1 {
		t.Fatalf("code = %d, want 1", code)
	}
	if got, want := stderr.String(), "fak sessions codex-loop: encode json: write failed\n"; got != want {
		t.Fatalf("stderr = %q, want %q", got, want)
	}
}
