package main

import (
	"os"
	"testing"
)

// claudeSessionUUID exists so a guard-session descriptor carries the SAME id a wip
// checkpoint stamps (#5343). The join is only sound if the two agree, so the ordering is
// pinned here against wip.go's stamp rather than merely asserted in a comment.
func TestClaudeSessionUUIDMatchesWipCheckpointStampOrder(t *testing.T) {
	t.Setenv("CLAUDE_CODE_SESSION_ID", "code-uuid")
	t.Setenv("CLAUDE_SESSION_ID", "plain-uuid")
	t.Setenv("FAK_SESSION_ID", "trace-id")

	// wip.go stamps firstNonEmpty(CLAUDE_CODE_SESSION_ID, FAK_SESSION_ID); when both are
	// set the checkpoint carries CLAUDE_CODE_SESSION_ID, so the descriptor must too.
	stamp := firstNonEmpty(os.Getenv("CLAUDE_CODE_SESSION_ID"), os.Getenv("FAK_SESSION_ID"))
	if got := claudeSessionUUID(); got != stamp {
		t.Fatalf("descriptor uuid %q does not match wip checkpoint stamp %q; the join would silently never resolve", got, stamp)
	}
	if got := claudeSessionUUID(); got != "code-uuid" {
		t.Fatalf("claudeSessionUUID() = %q, want code-uuid", got)
	}
}

func TestClaudeSessionUUIDFallbackOrder(t *testing.T) {
	cases := []struct {
		name                   string
		code, plain, fak, want string
	}{
		{"code wins", "a", "b", "c", "a"},
		{"plain when code empty", "", "b", "c", "b"},
		{"fak is last resort", "", "", "c", "c"},
		{"empty when nothing set", "", "", "", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("CLAUDE_CODE_SESSION_ID", tc.code)
			t.Setenv("CLAUDE_SESSION_ID", tc.plain)
			t.Setenv("FAK_SESSION_ID", tc.fak)
			if got := claudeSessionUUID(); got != tc.want {
				t.Fatalf("claudeSessionUUID() = %q, want %q", got, tc.want)
			}
		})
	}
}

// FAK_SESSION_ID is deliberately LAST: under `fak guard` a child sees it set to the
// volatile trace id, not the stable transcript uuid. Preferring it would publish an id
// that changes every run and joins to nothing.
func TestClaudeSessionUUIDPrefersStableUUIDOverVolatileTraceID(t *testing.T) {
	t.Setenv("CLAUDE_CODE_SESSION_ID", "stable-transcript-uuid")
	t.Setenv("CLAUDE_SESSION_ID", "")
	t.Setenv("FAK_SESSION_ID", "volatile-trace-0001")
	if got := claudeSessionUUID(); got != "stable-transcript-uuid" {
		t.Fatalf("claudeSessionUUID() = %q, want the stable uuid, not the guard trace id", got)
	}
}
