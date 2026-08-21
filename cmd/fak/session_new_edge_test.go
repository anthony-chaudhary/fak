package main

import (
	"bytes"
	"errors"
	"strings"
	"testing"
)

func TestSessionNewEdgeAdversarialInputs(t *testing.T) {
	cases := []struct {
		name     string
		argv     []string
		stdin    string
		wantCode int
		want     string
	}{
		{"empty positional", []string{"   "}, "", 2, "prompt is empty"},
		{"empty stdin", []string{"--stdin"}, "\n", 2, "prompt is empty"},
		{"conflicting sources", []string{"--stdin", "positional"}, "stdin", 2, "exactly one prompt source"},
		{"unknown terminal", []string{"--terminal", "not-a-terminal", "x"}, "", 1, "unsupported terminal"},
		{"hostile stays data", []string{"--dry-run", "$(Remove-Item -Recurse *) ; `whoami`\n--help"}, "", 0, "planned"},
		{"unicode", []string{"--dry-run", "λ 雪 🚀"}, "", 0, "planned"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			deps := sessionNewTestDeps("windows", tc.stdin)
			var stdout, stderr bytes.Buffer
			code := runSessionNewWith(&stdout, &stderr, tc.argv, deps)
			if code != tc.wantCode {
				t.Fatalf("code=%d want=%d stdout=%s stderr=%s", code, tc.wantCode, stdout.String(), stderr.String())
			}
			if combined := stdout.String() + stderr.String(); !strings.Contains(combined, tc.want) {
				t.Fatalf("output %q lacks %q", combined, tc.want)
			}
		})
	}
}

func TestSessionNewRefusalsNameRecovery(t *testing.T) {
	cases := []struct {
		name     string
		argv     []string
		mutate   func(*sessionNewDeps)
		recovery string
	}{
		{"missing prompt", nil, nil, "provide prompt text"},
		{"clipboard failure", []string{"--clipboard"}, func(d *sessionNewDeps) {
			d.readClipboard = func(string) (string, error) { return "", errors.New("clipboard offline") }
		}, "clipboard"},
		{"terminal missing", []string{"text"}, func(d *sessionNewDeps) {
			d.lookPath = func(string) (string, error) { return "", errors.New("not found") }
		}, "install"},
		{"start failure", []string{"--terminal", "windows-terminal", "text"}, func(d *sessionNewDeps) {
			d.start = func(string, string, []string) error { return errors.New("denied") }
		}, "start terminal"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			deps := sessionNewTestDeps("windows", "")
			if tc.mutate != nil {
				tc.mutate(&deps)
			}
			var stdout, stderr bytes.Buffer
			if code := runSessionNewWith(&stdout, &stderr, tc.argv, deps); code == 0 {
				t.Fatalf("expected refusal: %s", stdout.String())
			}
			if !strings.Contains(strings.ToLower(stderr.String()), strings.ToLower(tc.recovery)) {
				t.Fatalf("refusal %q lacks recovery %q", stderr.String(), tc.recovery)
			}
		})
	}
}
