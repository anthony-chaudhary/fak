// fak-test:runtime fast est=1s
package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func writeEmptyCorpusFixture(t *testing.T, name, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(body), 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadCorpusRejectsEmpty(t *testing.T) {
	for _, tc := range []struct{ name, body string }{
		{"object.json", "{}"},
		{"null.json", "null"},
		{"arrays.json", "{\"calls\":[],\"results\":[]}"},
		{"metadata.json", "{\"sources\":[\"fixture\"]}"},
		{"object.jsonl", "{}"},
		{"null.jsonl", "null"},
		{"arrays.jsonl", "{\"calls\":[],\"results\":[]}"},
		{"blank.jsonl", " \n\t\n"},
		{"source.jsonl", "{\"type\":\"source\",\"source\":\"fixture\"}\n{\"type\":\"source\",\"source\":\"second\"}\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := loadCorpus(writeEmptyCorpusFixture(t, tc.name, tc.body))
			if !errors.Is(err, errEmptyCorpus) {
				t.Fatalf("expected empty-corpus sentinel, got %v", err)
			}
		})
	}
}

func TestLoadCorpusPreservesInputErrors(t *testing.T) {
	for _, tc := range []struct{ name, body string }{
		{"broken.json", "{"},
		{"broken.jsonl", "{\n"},
		{"unknown.jsonl", "{\"type\":\"unknown\"}\n{}"},
		{"missing-tool.jsonl", "{\"type\":\"call\"}\n{}"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := loadCorpus(writeEmptyCorpusFixture(t, tc.name, tc.body))
			if err == nil || errors.Is(err, errEmptyCorpus) {
				t.Fatalf("expected distinct parse error, got %v", err)
			}
		})
	}
	_, err := loadCorpus(filepath.Join(t.TempDir(), "missing.json"))
	if !errors.Is(err, os.ErrNotExist) || errors.Is(err, errEmptyCorpus) {
		t.Fatalf("expected file-not-found identity, got %v", err)
	}
}

func TestLoadCorpusAcceptsSingleWorkKind(t *testing.T) {
	for _, tc := range []struct {
		name, body     string
		calls, results int
	}{
		{"call.json", "{\"calls\":[{\"tool\":\"read\",\"args\":\"{}\"}]}", 1, 0},
		{"result.json", "{\"results\":[{\"name\":\"clean\",\"tool\":\"read\",\"payload\":\"safe\"}]}", 0, 1},
		{"call.jsonl", "{\"type\":\"call\",\"tool\":\"read\",\"args\":\"{}\"}\n", 1, 0},
		{"result.jsonl", "{\"type\":\"result\",\"name\":\"clean\",\"tool\":\"read\",\"payload\":\"safe\"}\n", 0, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cp, err := loadCorpus(writeEmptyCorpusFixture(t, tc.name, tc.body))
			if err != nil || len(cp.Calls) != tc.calls || len(cp.Results) != tc.results {
				t.Fatalf("expected calls=%d results=%d, got calls=%d results=%d error=%v", tc.calls, tc.results, len(cp.Calls), len(cp.Results), err)
			}
		})
	}
}

func TestRunEmptyCorpusPreservesReports(t *testing.T) {
	for _, existing := range []bool{false, true} {
		name := "absent"
		if existing {
			name = "existing"
		}
		t.Run(name, func(t *testing.T) {
			input := writeEmptyCorpusFixture(t, "empty.json", "{}")
			out := filepath.Join(t.TempDir(), "report.json")
			original := []byte("previous report")
			if existing {
				if err := os.WriteFile(out, original, 0600); err != nil {
					t.Fatal(err)
				}
			}
			var stdout, stderr bytes.Buffer
			err := run(&stdout, &stderr, []string{"-corpus", input, "-out", out})
			if !errors.Is(err, errEmptyCorpus) {
				t.Fatalf("expected empty-corpus refusal, got %v", err)
			}
			if stdout.Len() != 0 {
				t.Fatalf("refused input emitted a report: %q", stdout.Bytes())
			}
			got, readErr := os.ReadFile(out)
			if existing {
				if readErr != nil || !bytes.Equal(got, original) {
					t.Fatalf("existing report changed: %q error=%v", got, readErr)
				}
			} else if !errors.Is(readErr, os.ErrNotExist) {
				t.Fatalf("refused input created output: %q error=%v", got, readErr)
			}
		})
	}
}

func TestDefaultCorpusHasWork(t *testing.T) {
	cp, err := loadCorpus("../../testdata/poison.json")
	if err != nil || len(cp.Calls)+len(cp.Results) == 0 {
		t.Fatalf("default fixture must contain work: calls=%d results=%d error=%v", len(cp.Calls), len(cp.Results), err)
	}
}
