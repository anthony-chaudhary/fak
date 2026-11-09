package validate

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// golistStreamsTestChatter is the verbatim stderr line reproduce 2026-11-09
// (landing ticket 20261104-router-keyring-drift-obs) saw corrupt the stream:
// the pre-fix WSL path ran CombinedOutput, so parseGoList choked on the first
// stderr token with `parsing go list json: invalid character 'g'`.
const golistStreamsTestChatter = "go: downloading go1.26.10 for linux/amd64\n"

// golistStreamsTestStdout emits the shape `go list -e -json ./...` produces:
// one JSON object per package, concatenated, with one in-module edge.
func golistStreamsTestStdout(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	for _, p := range []map[string]any{
		{
			"ImportPath": "example.com/mod",
			"Name":       "mod",
			"Dir":        "/tmp/mod",
			"Module":     map[string]string{"Path": "example.com/mod", "Dir": "/tmp/mod"},
			"GoFiles":    []string{"a.go"},
		},
		{
			"ImportPath": "example.com/mod/pkg",
			"Name":       "pkg",
			"Dir":        "/tmp/mod/pkg",
			"Module":     map[string]string{"Path": "example.com/mod", "Dir": "/tmp/mod"},
			"GoFiles":    []string{"b.go"},
			"Imports":    []string{"example.com/mod"},
		},
	} {
		if err := enc.Encode(p); err != nil {
			t.Fatalf("encode go list package: %v", err)
		}
	}
	return buf.Bytes()
}

// TestValidateGoListGraphWSLStreamsParseIgnoresStderrChatter is the symptom
// witness: download chatter on stderr must not corrupt the JSON package stream.
// fak-test:runtime fast est=10ms lane=default
func TestValidateGoListGraphWSLStreamsParseIgnoresStderrChatter(t *testing.T) {
	stdout := golistStreamsTestStdout(t)
	// The pre-fix combined-stream shape must still fail to parse: this is what
	// failed fak-private lands closed before the streams were separated.
	if _, _, _, err := parseGoList(bytes.NewReader(append([]byte(golistStreamsTestChatter), stdout...))); err == nil {
		t.Fatal("combined (pre-fix shape) stream parsed, want the parse-failure witness")
	}
	fileToPkg, edges, total, err := goListMetadataFromStreams(stdout, []byte(golistStreamsTestChatter), nil)
	if err != nil {
		t.Fatalf("goListMetadataFromStreams: %v", err)
	}
	if total != 2 {
		t.Fatalf("total = %d, want 2", total)
	}
	if fileToPkg["a.go"] != "example.com/mod" || fileToPkg["pkg/b.go"] != "example.com/mod/pkg" {
		t.Fatalf("fileToPkg = %v, want a.go -> example.com/mod and pkg/b.go -> example.com/mod/pkg", fileToPkg)
	}
	if deps := edges["example.com/mod/pkg"]; len(deps) != 1 || deps[0] != "example.com/mod" {
		t.Fatalf("edges[example.com/mod/pkg] = %v, want [example.com/mod]", deps)
	}
}

// TestValidateGoListGraphWSLStreamsErrorDetailCarriesStderr keeps diagnosis
// working: a failing run surfaces the captured stderr (trimmed, bounded) in the
// error detail, and the existing `go list produced no packages` shape is kept.
// fak-test:runtime fast est=10ms lane=default
func TestValidateGoListGraphWSLStreamsErrorDetailCarriesStderr(t *testing.T) {
	runErr := errors.New("exit status 1")

	_, _, _, err := goListMetadataFromStreams(nil, []byte("go: cannot find main module\n"), runErr)
	if err == nil {
		t.Fatal("goListMetadataFromStreams succeeded, want go-list failure")
	}
	if msg := err.Error(); !strings.Contains(msg, "go list produced no packages: exit status 1") || !strings.Contains(msg, "go: cannot find main module") {
		t.Fatalf("error = %q, want the existing prefix plus the captured stderr detail", msg)
	}

	// No stderr chatter: keep the pre-existing detail shape exactly.
	_, _, _, err = goListMetadataFromStreams(nil, nil, runErr)
	if err == nil {
		t.Fatal("goListMetadataFromStreams succeeded, want go-list failure")
	}
	if msg := err.Error(); !strings.Contains(msg, "go list produced no packages: exit status 1") || strings.Contains(msg, "stderr:") {
		t.Fatalf("error = %q, want the plain detail without a stderr section", msg)
	}

	// A genuinely corrupt stdout still fails parse, but carries the stderr detail.
	_, _, _, err = goListMetadataFromStreams([]byte(golistStreamsTestChatter+"{"), []byte("go: cannot find main module\n"), runErr)
	if err == nil {
		t.Fatal("goListMetadataFromStreams succeeded, want parse failure")
	}
	if msg := err.Error(); !strings.Contains(msg, "parsing go list json") || !strings.Contains(msg, "go: cannot find main module") {
		t.Fatalf("error = %q, want the parse failure plus the captured stderr detail", msg)
	}

	// The detail is bounded: a runaway module-download log must not flood the
	// land-failure line.
	huge := strings.Repeat(golistStreamsTestChatter, 200)
	_, _, _, err = goListMetadataFromStreams(nil, []byte(huge), runErr)
	if err == nil {
		t.Fatal("goListMetadataFromStreams succeeded, want go-list failure")
	}
	msg := err.Error()
	if !strings.Contains(msg, "[...truncated") {
		t.Fatalf("error = %q, want the truncation marker", msg)
	}
	if len(msg) > goListStderrDetailLimit+256 { // prefix + marker slack
		t.Fatalf("error detail length = %d, want bounded near %d", len(msg), goListStderrDetailLimit)
	}
}
