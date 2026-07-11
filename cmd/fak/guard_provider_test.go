package main

import "testing"

// guardSpendPricingContext is the seam that makes a fable guard session price at
// its own 2x-Opus rate instead of being under-booked as the agent-name Opus
// default. These cases pin the four branches: real model preferred when priced,
// agent-name fallback when the model is unknown or absent.
func TestGuardSpendPricingContext(t *testing.T) {
	cases := []struct {
		name       string
		provider   string
		guardModel string
		command    []string
		want       string
	}{
		{
			name:     "fable in child argv is preferred over the agent name",
			provider: "anthropic",
			command:  []string{"claude", "--model", "claude-fable-5", "-p", "hi"},
			want:     "claude-fable-5",
		},
		{
			name:       "fak --model override is preferred",
			provider:   "anthropic",
			guardModel: "claude-fable-5",
			command:    []string{"claude", "-p", "hi"},
			want:       "claude-fable-5",
		},
		{
			name:     "explicit opus resolves to itself (known price)",
			provider: "anthropic",
			command:  []string{"claude", "--model", "claude-opus-4-8"},
			want:     "claude-opus-4-8",
		},
		{
			name:     "no --model falls back to the agent name (Opus default)",
			provider: "anthropic",
			command:  []string{"claude", "-p", "hi"},
			want:     "claude",
		},
		{
			name:     "unknown model falls back to the agent name, never dollar-blind",
			provider: "anthropic",
			command:  []string{"claude", "--model", "totally-made-up-model"},
			want:     "claude",
		},
		{
			name:     "model after -- is not treated as the upstream model",
			provider: "anthropic",
			command:  []string{"claude", "--", "--model", "claude-fable-5"},
			want:     "claude",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := guardSpendPricingContext(tc.provider, tc.guardModel, tc.command)
			if got != tc.want {
				t.Fatalf("guardSpendPricingContext(%q, %q, %v) = %q, want %q",
					tc.provider, tc.guardModel, tc.command, got, tc.want)
			}
		})
	}
}
