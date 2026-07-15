package main

import (
	"strings"
	"testing"
)

func TestBlockedQwen36ModelCommandsFailClosedWithExactStagingGuidance(t *testing.T) {
	for _, label := range []string{"pull", "model load"} {
		message, blocked := blockedModelMessage("qwen3.6-27b", label)
		if !blocked {
			t.Fatalf("blockedModelMessage(%q) did not block", label)
		}
		for _, want := range []string{
			"known stale (HTTP 404)",
			"Qwen3.6-27B-Q4_K_M.gguf",
			"16547398784 bytes",
			"33625d8dc3a5dd8d88c324d47db58561b11f7072816287078bfe58b4c55782f9",
			"pass its local path directly",
		} {
			if !strings.Contains(message, want) {
				t.Fatalf("blockedModelMessage(%q) = %q; missing %q", label, message, want)
			}
		}
	}
}

func TestBlockedModelMessageLeavesOtherAliasesAlone(t *testing.T) {
	if message, blocked := blockedModelMessage("smollm2", "pull"); blocked || message != "" {
		t.Fatalf("blockedModelMessage(smollm2) = %q, %v; want empty, false", message, blocked)
	}
}
