package policy

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

func TestProfileStandardDefaultAllow(t *testing.T) {
	ctx := context.Background()

	// 1. Verify ProfileStandard sets PostureDefaultOpen, admits unlisted benign tools
	// while refusing dangerous gotchas with fail-closed POLICY_BLOCK.
	rtStd, err := RuntimeForProfile(ProfileStandard)
	if err != nil {
		t.Fatalf("RuntimeForProfile(ProfileStandard): %v", err)
	}
	if rtStd.Adjudicator.Posture != adjudicator.PostureDefaultOpen {
		t.Fatalf("expected rtStd.Adjudicator.Posture == PostureDefaultOpen, got %v", rtStd.Adjudicator.Posture)
	}
	if rtStd.PolicyContext.Posture != abi.PostureDefaultOpen {
		t.Fatalf("expected rtStd.PolicyContext.Posture == PostureDefaultOpen, got %v", rtStd.PolicyContext.Posture)
	}
	if rtStd.PolicyContext.Profile != "standard" {
		t.Fatalf("expected rtStd.PolicyContext.Profile == standard, got %q", rtStd.PolicyContext.Profile)
	}
	if rtStd.StrictGatedSinks {
		t.Fatalf("expected rtStd.StrictGatedSinks == false")
	}

	adjStd := adjudicator.New(rtStd.Adjudicator)

	// Benign unlisted tools must be admitted with default_open
	callBenign := &abi.ToolCall{
		Tool: "custom_linter",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"path":"main.go"}`)},
	}
	vBenign := adjStd.Adjudicate(ctx, callBenign)
	if vBenign.Kind != abi.VerdictAllow {
		t.Fatalf("unlisted benign tool under standard profile: got Kind=%v, want VerdictAllow", vBenign.Kind)
	}
	if vBenign.Meta["posture"] != "default_open" {
		t.Fatalf("unlisted benign tool under standard profile: got Meta[posture]=%q, want default_open", vBenign.Meta["posture"])
	}

	// Dangerous gotchas must be refused under standard profile
	callGotcha := &abi.ToolCall{
		Tool: "Bash",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"rm -rf /"}`)},
	}
	vGotcha := adjStd.Adjudicate(ctx, callGotcha)
	if vGotcha.Kind != abi.VerdictDeny || vGotcha.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("dangerous gotcha under standard profile: got Kind=%v, Reason=%v, want Deny/POLICY_BLOCK", vGotcha.Kind, vGotcha.Reason)
	}

	// 2. Verify ProfileStrict opts into PostureFailClosed
	rtStrict, err := RuntimeForProfile(ProfileStrict)
	if err != nil {
		t.Fatalf("RuntimeForProfile(ProfileStrict): %v", err)
	}
	if rtStrict.Adjudicator.Posture != adjudicator.PostureFailClosed {
		t.Fatalf("expected rtStrict.Adjudicator.Posture == PostureFailClosed, got %v", rtStrict.Adjudicator.Posture)
	}
	adjStrict := adjudicator.New(rtStrict.Adjudicator)
	vStrict := adjStrict.Adjudicate(ctx, callBenign)
	if vStrict.Kind != abi.VerdictDeny || vStrict.Reason != abi.ReasonDefaultDeny {
		t.Fatalf("unlisted tool under strict profile: got %v/%s, want Deny/DEFAULT_DENY", vStrict.Kind, abi.ReasonName(vStrict.Reason))
	}
}

func TestProfileDevDefaultAllow(t *testing.T) {
	ctx := context.Background()

	// 1. Verify ProfileDev sets PostureDefaultOpen, admits unlisted benign tools
	// while still refusing dangerous gotchas.
	rtDev, err := RuntimeForProfile(ProfileDev)
	if err != nil {
		t.Fatalf("RuntimeForProfile(ProfileDev): %v", err)
	}
	if rtDev.Adjudicator.Posture != adjudicator.PostureDefaultOpen {
		t.Fatalf("expected rtDev.Adjudicator.Posture == PostureDefaultOpen, got %v", rtDev.Adjudicator.Posture)
	}
	if rtDev.PolicyContext.Posture != abi.PostureDefaultOpen {
		t.Fatalf("expected rtDev.PolicyContext.Posture == PostureDefaultOpen, got %v", rtDev.PolicyContext.Posture)
	}
	if rtDev.PolicyContext.Profile != "dev" {
		t.Fatalf("expected rtDev.PolicyContext.Profile == dev, got %q", rtDev.PolicyContext.Profile)
	}
	if rtDev.StrictGatedSinks {
		t.Fatalf("expected rtDev.StrictGatedSinks == false")
	}

	adjDev := adjudicator.New(rtDev.Adjudicator)

	// Benign unlisted tools must be admitted with default_open
	benignTools := []struct {
		name string
		call *abi.ToolCall
	}{
		{
			name: "custom_linter",
			call: &abi.ToolCall{
				Tool: "custom_linter",
				Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"path":"main.go"}`)},
			},
		},
		{
			name: "echo_helper",
			call: &abi.ToolCall{
				Tool: "echo_helper",
				Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"text":"hello world"}`)},
			},
		},
	}

	for _, tc := range benignTools {
		v := adjDev.Adjudicate(ctx, tc.call)
		if v.Kind != abi.VerdictAllow {
			t.Fatalf("tool %s under dev profile: got Kind=%v, want VerdictAllow", tc.name, v.Kind)
		}
		if v.Meta["posture"] != "default_open" {
			t.Fatalf("tool %s under dev profile: got Meta[posture]=%q, want default_open", tc.name, v.Meta["posture"])
		}
	}

	// Dangerous gotchas must be refused even under dev profile
	gotchas := []struct {
		name string
		call *abi.ToolCall
	}{
		{
			name: "shell_rm_rf",
			call: &abi.ToolCall{
				Tool: "Bash",
				Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"rm -rf /"}`)},
			},
		},
		{
			name: "raw_disk",
			call: &abi.ToolCall{
				Tool: "Bash",
				Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"mkfs.ext4 /dev/sda1"}`)},
			},
		},
		{
			name: "system_disruption",
			call: &abi.ToolCall{
				Tool: "Bash",
				Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"kill -9 1"}`)},
			},
		},
	}

	for _, tc := range gotchas {
		v := adjDev.Adjudicate(ctx, tc.call)
		if v.Kind != abi.VerdictDeny {
			t.Fatalf("dangerous gotcha %s under dev profile: got Kind=%v, want VerdictDeny", tc.name, v.Kind)
		}
	}

	// 2. Verify ProfileProd sets PostureFailClosed and refuses unlisted tools with DEFAULT_DENY.
	rtProd, err := RuntimeForProfile(ProfileProd)
	if err != nil {
		t.Fatalf("RuntimeForProfile(ProfileProd): %v", err)
	}
	if rtProd.Adjudicator.Posture != adjudicator.PostureFailClosed {
		t.Fatalf("expected rtProd.Adjudicator.Posture == PostureFailClosed, got %v", rtProd.Adjudicator.Posture)
	}
	if rtProd.PolicyContext.Posture != abi.PostureFailClosed {
		t.Fatalf("expected rtProd.PolicyContext.Posture == PostureFailClosed, got %v", rtProd.PolicyContext.Posture)
	}
	if rtProd.PolicyContext.Profile != "prod" {
		t.Fatalf("expected rtProd.PolicyContext.Profile == prod, got %q", rtProd.PolicyContext.Profile)
	}
	if !rtProd.StrictGatedSinks {
		t.Fatalf("expected rtProd.StrictGatedSinks == true")
	}

	adjProd := adjudicator.New(rtProd.Adjudicator)
	vProd := adjProd.Adjudicate(ctx, benignTools[0].call)
	if vProd.Kind != abi.VerdictDeny || vProd.Reason != abi.ReasonDefaultDeny {
		t.Fatalf("unlisted tool under prod profile: got %v/%s, want Deny/DEFAULT_DENY", vProd.Kind, abi.ReasonName(vProd.Reason))
	}

	// 3. Verify ProfileAudit sets PostureAdmitAndLog.
	rtAudit, err := RuntimeForProfile(ProfileAudit)
	if err != nil {
		t.Fatalf("RuntimeForProfile(ProfileAudit): %v", err)
	}
	if rtAudit.Adjudicator.Posture != adjudicator.PostureAdmitAndLog {
		t.Fatalf("expected rtAudit.Adjudicator.Posture == PostureAdmitAndLog, got %v", rtAudit.Adjudicator.Posture)
	}
	if rtAudit.PolicyContext.Posture != abi.PostureAdmitAndLog {
		t.Fatalf("expected rtAudit.PolicyContext.Posture == PostureAdmitAndLog, got %v", rtAudit.PolicyContext.Posture)
	}
	if rtAudit.PolicyContext.Profile != "audit" {
		t.Fatalf("expected rtAudit.PolicyContext.Profile == audit, got %q", rtAudit.PolicyContext.Profile)
	}
	if rtAudit.StrictGatedSinks {
		t.Fatalf("expected rtAudit.StrictGatedSinks == false")
	}
}

func TestParseAndValidateProfile(t *testing.T) {
	for _, tc := range []struct {
		input   string
		want    Profile
		wantErr bool
	}{
		{"dev", ProfileDev, false},
		{"DEV", ProfileDev, false},
		{"  dev  ", ProfileDev, false},
		{"prod", ProfileProd, false},
		{"PROD", ProfileProd, false},
		{"audit", ProfileAudit, false},
		{"AUDIT", ProfileAudit, false},
		{"", "", false},
		{"   ", "", false},
		{"invalid", "", true},
		{"unknown", "", true},
	} {
		got, err := ParseProfile(tc.input)
		if (err != nil) != tc.wantErr {
			t.Errorf("ParseProfile(%q) error = %v, wantErr = %v", tc.input, err, tc.wantErr)
		}
		if got != tc.want {
			t.Errorf("ParseProfile(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}

	if err := ValidateProfile(ProfileDev); err != nil {
		t.Errorf("ValidateProfile(ProfileDev) failed: %v", err)
	}
	if err := ValidateProfile(ProfileProd); err != nil {
		t.Errorf("ValidateProfile(ProfileProd) failed: %v", err)
	}
	if err := ValidateProfile(ProfileAudit); err != nil {
		t.Errorf("ValidateProfile(ProfileAudit) failed: %v", err)
	}
	if err := ValidateProfile(""); err != nil {
		t.Errorf("ValidateProfile(\"\") failed: %v", err)
	}
	if err := ValidateProfile("nonexistent"); err == nil {
		t.Error("ValidateProfile(nonexistent) expected error, got nil")
	}
}

func TestManifestProfileDefaults(t *testing.T) {
	// Manifest with profile "dev" and no posture should default to default_open
	manifestJSON := `{"version":"fak-policy/v1","profile":"dev"}`
	m, err := ParseManifest([]byte(manifestJSON))
	if err != nil {
		t.Fatalf("ParseManifest: %v", err)
	}
	if m.Profile != "dev" {
		t.Fatalf("m.Profile = %q, want dev", m.Profile)
	}

	rt, err := m.ToRuntime()
	if err != nil {
		t.Fatalf("ToRuntime: %v", err)
	}
	if rt.Adjudicator.Posture != adjudicator.PostureDefaultOpen {
		t.Fatalf("rt.Adjudicator.Posture = %v, want PostureDefaultOpen", rt.Adjudicator.Posture)
	}
	if rt.PolicyContext.Profile != "dev" {
		t.Fatalf("rt.PolicyContext.Profile = %q, want dev", rt.PolicyContext.Profile)
	}

	// Manifest with profile "dev" but explicit posture "fail_closed" should keep fail_closed
	overrideJSON := `{"version":"fak-policy/v1","profile":"dev","posture":"fail_closed"}`
	mOver, err := ParseManifest([]byte(overrideJSON))
	if err != nil {
		t.Fatalf("ParseManifest override: %v", err)
	}
	rtOver, err := mOver.ToRuntime()
	if err != nil {
		t.Fatalf("ToRuntime override: %v", err)
	}
	if rtOver.Adjudicator.Posture != adjudicator.PostureFailClosed {
		t.Fatalf("rtOver.Adjudicator.Posture = %v, want PostureFailClosed (explicit override)", rtOver.Adjudicator.Posture)
	}

	// Round-trip through FromRuntime
	mRound := FromRuntime(rt)
	if mRound.Profile != "dev" {
		t.Fatalf("FromRuntime lost profile: got %q, want dev", mRound.Profile)
	}
	if !strings.Contains(string(mRound.JSON()), `"profile": "dev"`) {
		t.Fatalf("mRound.JSON() missing profile: %s", string(mRound.JSON()))
	}
}
