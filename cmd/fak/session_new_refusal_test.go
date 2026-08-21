package main

import (
	"bytes"
	"errors"
	"strings"
	"testing"
)

func TestSessionNewErrorsNameActionableRecovery(t *testing.T) {
	cases := []struct {
		name       string
		argv       []string
		mutate     func(*sessionNewDeps)
		recoveries []string
	}{
		{"no source", nil, nil, []string{"provide prompt text", "--stdin", "--clipboard"}},
		{"clipboard unavailable", []string{"--clipboard"}, func(d *sessionNewDeps) {
			d.readClipboard = func(string) (string, error) { return "", errors.New("offline") }
		}, []string{"provide prompt text directly"}},
		{"terminal unavailable", []string{"text"}, func(d *sessionNewDeps) {
			d.lookPath = func(string) (string, error) { return "", errors.New("missing") }
		}, []string{"install", "--terminal"}},
		{"launch refused", []string{"--terminal", "windows-terminal", "text"}, func(d *sessionNewDeps) {
			d.start = func(string, string, []string) error { return errors.New("denied") }
		}, []string{"--dry-run", "--terminal"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			deps := sessionNewTestDeps("windows", "")
			if tc.mutate != nil {
				tc.mutate(&deps)
			}
			var stdout, stderr bytes.Buffer
			if code := runSessionNewWith(&stdout, &stderr, tc.argv, deps); code == 0 {
				t.Fatalf("expected refusal")
			}
			message := strings.ToLower(stderr.String())
			for _, recovery := range tc.recoveries {
				if !strings.Contains(message, strings.ToLower(recovery)) {
					t.Fatalf("message %q lacks recovery %q", stderr.String(), recovery)
				}
			}
		})
	}
}
