package agent

// inkernel_eligible_test.go — the planner side of the #3391 eligibility-filtered
// hit-rate denominator: which prompt tokens the planner books as ABLE to hit the cached
// KV prefix. Reuse disabled → none, ever. Reuse enabled → the first prefill is
// ineligible (nothing admitted yet, so nothing could match — the always-cold head the
// raw ratio unfairly counts against the cache), and every turn after the admission
// latch flips counts its whole prompt.

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

func TestKVPrefixEligibleWitness(t *testing.T) {
	off := &InKernelPlanner{} // no tree: prefix reuse disabled
	if got := off.kvPrefixEligiblePromptTokens(500); got != 0 {
		t.Fatalf("reuse-disabled planner eligible = %d, want 0 (no token can ever hit)", got)
	}
	off.noteKVPrefixAdmitted() // must not latch a reuse-disabled planner
	if got := off.kvPrefixEligiblePromptTokens(500); got != 0 {
		t.Fatalf("reuse-disabled planner eligible after note = %d, want 0", got)
	}

	on := &InKernelPlanner{tree: radixkv.New(0)} // CPU path: reuse supported
	if got := on.kvPrefixEligiblePromptTokens(500); got != 0 {
		t.Fatalf("first-prefill eligible = %d, want 0 (empty cache: nothing could match)", got)
	}
	on.noteKVPrefixAdmitted() // a successful reuse-enabled turn admitted its prompt
	if got := on.kvPrefixEligiblePromptTokens(700); got != 700 {
		t.Fatalf("post-admission eligible = %d, want the whole prompt (700)", got)
	}
}
