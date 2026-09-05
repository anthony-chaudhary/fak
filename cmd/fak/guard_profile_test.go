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

// TestProfileApplyFloorWithProfileDefaults verifies that applyFloorWithProfile
// (the capability floor installer for fak serve and serve --stdio) defaults to
// ProfileStandard (permissive default-open posture) when path and profile are empty,
// admitting benign custom tools while blocking dangerous gotchas, and honors an
// explicit strict or prod profile by rejecting unlisted tools with DEFAULT_DENY (#11370, #11358).
func TestProfileApplyFloorWithProfileDefaults(t *testing.T) {
	ctx := context.Background()

	t.Setenv("FAK_PROFILE", "")

	prior := adjudicator.Default.PolicySnapshot()
	defer func() {
		adjudicator.Default.SetPolicy(prior)
	}()

	// 1. When path == "" and profile == "", applyFloorWithProfile applies ProfileStandard (posture=default_open)
	applyFloorWithProfile("", "")

	snap := adjudicator.Default.PolicySnapshot()
	if snap.Posture != adjudicator.PostureDefaultOpen {
		t.Fatalf("expected PostureDefaultOpen under default applyFloorWithProfile, got %v", snap.Posture)
	}

	benignCall := guardToolCall(t, "custom_unlisted_tool", map[string]any{"action": "ping"})
	vBenign := adjudicator.Default.Adjudicate(ctx, benignCall)
	if vBenign.Kind != abi.VerdictAllow {
		t.Fatalf("unlisted benign tool under default applyFloorWithProfile: got %v (%s), want VerdictAllow",
			vBenign.Kind, abi.ReasonName(vBenign.Reason))
	}
	if vBenign.Meta["posture"] != "default_open" {
		t.Fatalf("unlisted benign tool meta[posture]: got %q, want default_open", vBenign.Meta["posture"])
	}

	// Dangerous gotchas must remain strictly blocked with POLICY_BLOCK
	gotchaCall := guardToolCall(t, "Bash", map[string]any{"command": "rm -rf /"})
	vGotcha := adjudicator.Default.Adjudicate(ctx, gotchaCall)
	if vGotcha.Kind != abi.VerdictDeny || vGotcha.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("dangerous gotcha under default applyFloorWithProfile: got %v/%s, want Deny/POLICY_BLOCK",
			vGotcha.Kind, abi.ReasonName(vGotcha.Reason))
	}

	// 2. When explicit strict profile is passed, unlisted tools remain rejected with DEFAULT_DENY
	applyFloorWithProfile("", "strict")

	snapStrict := adjudicator.Default.PolicySnapshot()
	if snapStrict.Posture != adjudicator.PostureFailClosed {
		t.Fatalf("expected PostureFailClosed under explicit strict profile, got %v", snapStrict.Posture)
	}

	vStrict := adjudicator.Default.Adjudicate(ctx, benignCall)
	if vStrict.Kind != abi.VerdictDeny || vStrict.Reason != abi.ReasonDefaultDeny {
		t.Fatalf("unlisted tool under explicit strict profile: got %v/%s, want Deny/DEFAULT_DENY",
			vStrict.Kind, abi.ReasonName(vStrict.Reason))
	}

	// 3. When explicit prod profile is passed, unlisted tools remain rejected with DEFAULT_DENY
	applyFloorWithProfile("", "prod")

	snapProd := adjudicator.Default.PolicySnapshot()
	if snapProd.Posture != adjudicator.PostureFailClosed {
		t.Fatalf("expected PostureFailClosed under explicit prod profile, got %v", snapProd.Posture)
	}

	vProd := adjudicator.Default.Adjudicate(ctx, benignCall)
	if vProd.Kind != abi.VerdictDeny || vProd.Reason != abi.ReasonDefaultDeny {
		t.Fatalf("unlisted tool under explicit prod profile: got %v/%s, want Deny/DEFAULT_DENY",
			vProd.Kind, abi.ReasonName(vProd.Reason))
	}
}
