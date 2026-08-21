package main

import (
	"strings"
	"testing"
)

func TestGuardProfileRefusalsNameRecovery(t *testing.T) {
	cases := []struct {
		name               string
		command            []string
		output, work, want string
	}{
		{"unknown output", []string{"codex"}, "missing", agentDefaultWorkProfile, "supported:"},
		{"unknown work", []string{"codex"}, agentDefaultOutputStyle, "missing", "supported:"},
		{"unsupported harness", []string{"other"}, agentDefaultOutputStyle, agentDefaultWorkProfile, "no witnessed profile injection seam"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, capture, err := injectGuardProfiles(tc.command, tc.output, tc.work, true)
			if err == nil || !strings.Contains(err.Error(), tc.want) || got != nil || capture != nil {
				t.Fatalf("got=(%v,%v,%v), want refusal containing %q", got, capture, err, tc.want)
			}
		})
	}
}
