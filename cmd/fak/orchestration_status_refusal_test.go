package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestOrchestrationStatusRefusalsNameRecovery(t *testing.T) {
	cases := []struct {
		name string
		argv []string
		want []string
	}{
		{"missing session", nil, []string{"session id", "--session"}},
		{"unknown flag", []string{"--not-a-flag"}, []string{"usage:", "orchestration status"}},
		{"missing receipt", []string{"--session", "does-not-exist"}, []string{"receipt", "does-not-exist"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("CODEX_HOME", t.TempDir())
			var stdout, stderr bytes.Buffer
			if code := runOrchestrationStatus(&stdout, &stderr, tc.argv); code == 0 {
				t.Fatalf("expected refusal: %s", stdout.String())
			}
			message := stderr.String()
			for _, want := range tc.want {
				if !strings.Contains(message, want) {
					t.Fatalf("message %q lacks recovery %q", stderr.String(), want)
				}
			}
		})
	}
}
