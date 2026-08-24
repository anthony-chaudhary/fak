package main

import (
	"errors"
	"strings"
	"testing"
)

func TestGuardInfoRuntimeIdentityVerdicts(t *testing.T) {
	cases := []struct {
		name, source, running, want string
	}{
		{"match", "501d308a4a2b", "501d308a4a2b", "MATCH"},
		{"stale", "501d308a4a2b", "e2fb0c8c5b11", "STALE"},
		{"unknown source", "", "501d308a4a2b", "UNKNOWN"},
		{"dirty build", "501d308a4a2b", "501d308a4a2b+dirty", "UNKNOWN"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := guardInfoRuntimeIdentityOf(tc.source, "cmd/fak@r3292", tc.running, "sha256:abc123", "2026-08-24T20:00:00Z")
			if got.Verdict != tc.want {
				t.Fatalf("verdict = %q, want %q", got.Verdict, tc.want)
			}
			if got.ConfigDigest != "sha256:abc123" || got.SessionStarted == "unknown" {
				t.Fatalf("receipt fields lost: %+v", got)
			}
			if got.Verdict != "MATCH" && got.Action == "none" {
				t.Fatalf("%s identity lacks operator action: %+v", got.Verdict, got)
			}
		})
	}
}

func TestGuardInfoRuntimeIdentityRowIsOperatorVisibleAndPrivate(t *testing.T) {
	id := guardInfoRuntimeIdentityOf("501d308a4a2b", "cmd/fak@r3292", "e2fb0c8c5b11", guardInfoStartupDigest("fak guard: capability floor /private/operator/policy.json\nfak guard: active config digest sha256:abc123\n"), "2026-08-24T20:00:00Z")
	got := guardInfoRuntimeIdentityRow(id)
	for _, want := range []string{"identity STALE", "source 501d308a4a2b (cmd/fak@r3292)", "running e2fb0c8c5b11", "config sha256:", "session 2026-08-24T20:00:00Z", "go install ./cmd/fak"} {
		if !strings.Contains(got, want) {
			t.Fatalf("identity row missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "/private/") || strings.Contains(got, "policy.json") {
		t.Fatalf("identity row leaked configuration content/path: %s", got)
	}
}

func TestGuardInfoSourceHeadFrom(t *testing.T) {
	got := guardInfoSourceHeadFrom(func(args ...string) ([]byte, error) {
		if strings.Join(args, " ") != "rev-parse --verify HEAD" {
			t.Fatalf("unexpected git args: %v", args)
		}
		return []byte("501d308a4a2bd7aca968\n"), nil
	})
	if got != "501d308a4a2b" {
		t.Fatalf("source head = %q", got)
	}
	if got := guardInfoSourceHeadFrom(func(...string) ([]byte, error) { return nil, errors.New("no git") }); got != "unknown" {
		t.Fatalf("failed source lookup = %q, want unknown", got)
	}
}
