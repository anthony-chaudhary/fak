package gateway

import (
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// syntheticServedPrompt returns one user message the served-admission estimator
// prices at exactly promptTokens (chars/4, role included).
func syntheticServedPrompt(promptTokens int) []agent.Message {
	return []agent.Message{{Role: "user", Content: strings.Repeat("x", 4*promptTokens-len("user"))}}
}

// admitUntilFull offers n identical served requests to a fresh controller and
// returns how many entered the running set at once.
func admitUntilFull(t *testing.T, policy AdmissionPolicy, promptTokens, maxTokens, n int) (admitted, footprint int) {
	t.Helper()
	ctl := NewAdmissionController(policy)
	footprint = estimateServedAdmissionTokensWithCap(syntheticServedPrompt(promptTokens), nil, maxTokens, policy.PreallocCeiling)
	for i := 0; i < n; i++ {
		if ctl.Offer(SeqRequest{TraceID: fmt.Sprintf("req-%d", i), Tokens: footprint}) == VerdictAdmitted {
			admitted++
		}
	}
	if got := ctl.Stats().Running; got != admitted {
		t.Fatalf("running set = %d, admitted verdicts = %d", got, admitted)
	}
	return admitted, footprint
}

// TestNativeAdmissionConcurrencyUnderShippingPolicy pins how many served requests
// the native admission gate runs at once. The token budget, not MaxNumSeqs, is the
// binding axis for any realistic request: each request is charged its prompt plus
// min(max_tokens, PreallocCeiling), so 8192 (the fallback when no native context
// resolves) runs at most three requests that ask for >=2048 tokens of output, and a
// context-resolved budget (fak serve copies the resolved window into the budget)
// scales concurrency with that window, never with slots x window.
//
// fak-test:runtime fast est=50ms lane=default
func TestNativeAdmissionConcurrencyUnderShippingPolicy(t *testing.T) {
	shipping := DefaultAdmissionPolicy()
	resolved32k := DefaultAdmissionPolicy()
	resolved32k.TokenBudget = 32768
	resolved32k.TokenBudgetProvenance = "context"

	cases := []struct {
		name          string
		policy        AdmissionPolicy
		promptTokens  int
		maxTokens     int
		wantFootprint int
		wantAdmitted  int
	}{
		{"fallback/tiny-no-max-tokens", shipping, 8, 0, 9, 256},
		{"fallback/chat-1k-prompt-512-out", shipping, 1024, 512, 1536, 5},
		{"fallback/empty-ish-prompt-2k-out", shipping, 16, 2048, 2064, 3},
		{"fallback/chat-2k-prompt-4k-out-capped", shipping, 2048, 4096, 4096, 2},
		{"fallback/agent-6k-prompt-8k-out-capped", shipping, 6144, 8192, 8192, 1},
		{"context-32k/chat-2k-prompt-4k-out-capped", resolved32k, 2048, 4096, 4096, 8},
		{"context-32k/agent-6k-prompt-8k-out-capped", resolved32k, 6144, 8192, 8192, 4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			admitted, footprint := admitUntilFull(t, tc.policy, tc.promptTokens, tc.maxTokens, 300)
			if footprint != tc.wantFootprint {
				t.Fatalf("per-request footprint = %d, want %d", footprint, tc.wantFootprint)
			}
			if admitted != tc.wantAdmitted {
				t.Fatalf("concurrently admitted = %d, want %d (budget=%d max_num_seqs=%d)",
					admitted, tc.wantAdmitted, tc.policy.TokenBudget, tc.policy.MaxNumSeqs)
			}
		})
	}
}
