package dojo

import (
	"testing"
)

func TestDetectResubmissionLoops_SessionIsolation(t *testing.T) {
	// Consecutive samples from two independent sessions where neither loops individually.
	// Session A has 2 identical tool calls after elision (streak = 2, threshold is 3 -> no loop).
	// Session A also ends with a yield issued with 1000 PromptTokens, but no subsequent continuation turn.
	// Session B starts with 1 identical tool call (same tool and arguments hash), streak = 1 -> no loop.
	// Session B also has PromptTokens = 1200, which without session isolation would falsely match
	// Session A's pending yield continuation (1200 >= 1000).
	samples := []ResubmissionLoopSample{
		// Session A
		{SessionID: "session_a", Turn: 1, ToolName: "read_file", ArgumentsHash: "hash_x", WasElided: true},
		{SessionID: "session_a", Turn: 2, ToolName: "read_file", ArgumentsHash: "hash_x", WasElided: false, YieldIssued: true, PromptTokens: 1000},
		// Session B
		{SessionID: "session_b", Turn: 1, ToolName: "read_file", ArgumentsHash: "hash_x", WasElided: false, PromptTokens: 1200},
	}

	toolLoops, yieldLoops := DetectResubmissionLoops(samples)
	if toolLoops != 0 {
		t.Errorf("toolLoops = %d, want 0 (cross-session false positive detected)", toolLoops)
	}
	if yieldLoops != 0 {
		t.Errorf("yieldLoops = %d, want 0 (cross-session false positive detected)", yieldLoops)
	}
}
