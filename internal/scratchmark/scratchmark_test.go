package scratchmark

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDetectClassifiesMarkerCorpus(t *testing.T) {
	tests := []struct {
		name   string
		source string
		marker string
	}{
		{"temporary file", "// Temporary file; remove after the migration.\npackage probe\n", "temporary"},
		{"disposable script", "# disposable script used for one audit\nprint('ok')\n", "disposable"},
		{"throwaway block comment", "/* This is a throwaway probe. */\npackage probe\n", "throwaway"},
		{"scratch helper", "// This helper is scratch code.\npackage probe\n", "scratch"},
		{"crlf header", "// THIS SOURCE IS DISPOSABLE.\r\npackage probe\r\n", "disposable"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := Detect([]byte(tc.source))
			if !got.Marked || got.Kept {
				t.Fatalf("Detect() = %+v, want marked and not kept", got)
			}
			if got.Marker != tc.marker {
				t.Errorf("Marker = %q, want %q", got.Marker, tc.marker)
			}
			if got.Line != 1 {
				t.Errorf("Line = %d, want 1", got.Line)
			}
		})
	}
}

func TestDetectLeavesOrdinaryProseAndCodeClean(t *testing.T) {
	tests := []struct {
		name   string
		source string
	}{
		{"temporary resource", "// NewTempDir creates a temporary directory for the caller.\npackage clean\n"},
		{"disposable noun", "// Disposable income is unrelated example text.\npackage clean\n"},
		{"scratch buffer", "// Scratch buffers are pooled between calls.\npackage clean\n"},
		{"marker after code", "package clean\n\n// This file is temporary.\n"},
		{"marker in string", "package clean\n\nconst note = `disposable file`\n"},
		{"ordinary header", "// Package clean implements stable parsing.\npackage clean\n"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := Detect([]byte(tc.source)); got != (Result{}) {
				t.Fatalf("Detect() = %+v, want a clean result", got)
			}
		})
	}
}

func TestDetectKeepDirectiveOverridesMarker(t *testing.T) {
	source := "// This file is temporary.\n// " + KeepDirective + " -- retained as a fixture\npackage kept\n"
	got := Detect([]byte(source))
	if got.Marked || !got.Kept {
		t.Fatalf("Detect() = %+v, want kept and not marked", got)
	}
}

func TestDetectBoundsLeadingHeader(t *testing.T) {
	prefix := "// " + strings.Repeat("x", MaxHeaderBytes) + "\n"
	got := Detect([]byte(prefix + "// This file is disposable.\npackage late\n"))
	if got != (Result{}) {
		t.Fatalf("Detect() = %+v, want marker beyond %d-byte header ignored", got, MaxHeaderBytes)
	}
}

func TestScanReadsFileAndReportsUnreadablePath(t *testing.T) {
	path := filepath.Join(t.TempDir(), "probe.go")
	if err := os.WriteFile(path, []byte("// throwaway file\npackage probe\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := Scan(path)
	if err != nil {
		t.Fatalf("Scan(%q): %v", path, err)
	}
	if !got.Marked || got.Marker != "throwaway" {
		t.Fatalf("Scan(%q) = %+v, want throwaway marker", path, got)
	}

	missing := filepath.Join(t.TempDir(), "missing.go")
	if _, err := Scan(missing); err == nil {
		t.Fatalf("Scan(%q) error = nil, want unreadable-file error", missing)
	}
}
