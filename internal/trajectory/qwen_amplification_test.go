package trajectory

import "testing"

func TestQwenAmplificationDecision(t *testing.T) {
	usage := &QwenCanonicalUsage{InputTokens: 100, CacheReadTokens: 80, OutputTokens: 20, UsefulWitnesses: 2}
	tests := []struct {
		name     string
		policy   QwenAmplificationPolicy
		usage    *QwenCanonicalUsage
		want     QwenAmplificationAction
		eligible bool
	}{
		{"below budget", QwenAmplificationPolicy{InputTokenBudget: 101}, usage, QwenAmplificationObserve, true},
		{"breach observe", QwenAmplificationPolicy{InputTokenBudget: 99}, usage, QwenAmplificationAlert, true},
		{"breach enforce hold", QwenAmplificationPolicy{InputTokenBudget: 99, Enforce: true}, usage, QwenAmplificationHold, true},
		{"missing usage", QwenAmplificationPolicy{InputTokenBudget: 99}, nil, QwenAmplificationObserve, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.policy.Decide("fak-native/qwen", tt.usage, "")
			if got.Action != tt.want || got.Eligible != tt.eligible {
				t.Fatalf("got action=%q eligible=%v", got.Action, got.Eligible)
			}
			if tt.usage == nil && got.NonEligibleReason != QwenAmplificationMissingUsage {
				t.Fatalf("missing usage reason = %q", got.NonEligibleReason)
			}
		})
	}
}

func TestQwenAmplificationOverrideReceipt(t *testing.T) {
	p := QwenAmplificationPolicy{InputTokenBudget: 99, Enforce: true}
	got := p.Decide("fak-native/qwen", &QwenCanonicalUsage{InputTokens: 100}, "approved experiment")
	if got.Action != QwenAmplificationAlert {
		t.Fatalf("action = %q", got.Action)
	}
	if got.Override == nil || got.Override.Reason != "approved experiment" {
		t.Fatalf("override receipt = %#v", got.Override)
	}
}

func TestQwenAmplificationEnforceRequiresFakNativeEngine(t *testing.T) {
	p := QwenAmplificationPolicy{InputTokenBudget: 1, Enforce: true}
	for _, engine := range []string{"", "other/qwen"} {
		got := p.Decide(engine, &QwenCanonicalUsage{InputTokens: 2}, "")
		if got.Eligible || got.NonEligibleReason != QwenAmplificationInvalidEngine {
			t.Fatalf("engine %q: %#v", engine, got)
		}
	}
}
