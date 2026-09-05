package main

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// TestProfileStandardDefaultAllow verifies that fak guard resolves standard profile
// to permissive default-open posture, admitting unlisted benign tools while strictly
// fail-closing on dangerous gotchas with POLICY_BLOCK.
func TestProfileStandardDefaultAllow(t *testing.T) {
	ctx := context.Background()

	// 1. Load floor with standard profile (or default empty profile)
	rt, _, _, _ := loadGuardCapabilityFloor("", "default_open", "standard")
	if rt.Adjudicator.Posture != adjudicator.PostureDefaultOpen {
		t.Fatalf("expected PostureDefaultOpen under standard profile, got %v", rt.Adjudicator.Posture)
	}

	adj := adjudicator.New(rt.Adjudicator)

	// Benign unlisted tools must be allowed
	benignCall := guardToolCall(t, "custom_unlisted_tool", map[string]any{"action": "ping"})
	v := adj.Adjudicate(ctx, benignCall)
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("unlisted tool under standard profile: got Kind=%v (%s), want VerdictAllow", v.Kind, abi.ReasonName(v.Reason))
	}

	// Dangerous gotchas must be blocked with POLICY_BLOCK
	gotchaCall := guardToolCall(t, "Bash", map[string]any{"command": "rm -rf /"})
	vGotcha := adj.Adjudicate(ctx, gotchaCall)
	if vGotcha.Kind != abi.VerdictDeny || vGotcha.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("gotcha under standard profile: got Kind=%v, Reason=%v, want Deny/POLICY_BLOCK", vGotcha.Kind, vGotcha.Reason)
	}

	// 2. Load floor with strict profile (opt-in)
	rtStrict, _, _, _ := loadGuardCapabilityFloor("", "fail_closed", "strict")
	if rtStrict.Adjudicator.Posture != adjudicator.PostureFailClosed {
		t.Fatalf("expected PostureFailClosed under strict profile, got %v", rtStrict.Adjudicator.Posture)
	}

	adjStrict := adjudicator.New(rtStrict.Adjudicator)
	vStrict := adjStrict.Adjudicate(ctx, benignCall)
	if vStrict.Kind != abi.VerdictDeny || vStrict.Reason != abi.ReasonDefaultDeny {
		t.Fatalf("unlisted tool under strict profile: got Kind=%v, Reason=%v, want Deny/DEFAULT_DENY", vStrict.Kind, vStrict.Reason)
	}
}
