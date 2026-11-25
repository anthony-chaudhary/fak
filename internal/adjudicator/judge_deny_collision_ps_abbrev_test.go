package adjudicator

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Two Deny keys that fold to the same name must resolve to the hard reason no
// matter which casing is exact-lowercase or which order the map yields.
func TestDenyCaseCollisionKeepsHardReason(t *testing.T) {
	ctx := context.Background()
	for _, deny := range []map[string]abi.ReasonCode{
		{"wipe": abi.ReasonSelfModify, "Wipe": abi.ReasonPolicyBlock},
		{"WIPE": abi.ReasonSelfModify, "Wipe": abi.ReasonPolicyBlock},
		{"Wipe": abi.ReasonSelfModify, "WIPE": abi.ReasonPolicyBlock},
	} {
		raw := Policy{
			Posture:         PostureDefaultOpen,
			Deny:            deny,
			AdvisoryReasons: map[abi.ReasonCode]bool{abi.ReasonSelfModify: true},
		}
		for i := 0; i < 64; i++ {
			a := New(raw)
			for _, tool := range []string{"wipe", "Wipe", "WIPE"} {
				if v := a.Adjudicate(ctx, inlineCall(tool, `{}`)); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
					t.Fatalf("Deny %v, call %q: got %v/%v, want Deny/POLICY_BLOCK", deny, tool, v.Kind, v.Reason)
				}
				if !raw.NeverAdmits(tool) || !a.NeverAdmits(tool) {
					t.Fatalf("Deny %v: NeverAdmits(%q) = false, want true", deny, tool)
				}
			}
		}
	}
}

// PowerShell binds abbreviated and common-parameter aliases; a value flag the
// extractor does not recognise must not shift the positional path onto its value.
func TestPowerShellAbbreviatedParamsStillGuarded(t *testing.T) {
	ctx := context.Background()
	a := New(Policy{
		Posture:          PostureDefaultOpen,
		BlockedPathGlobs: []string{".env"},
		SelfModifyGlobs:  []string{"internal/abi/"},
	})
	for _, cmd := range []string{
		`Set-Content -Va x .env`,
		`Set-Content -Va x -Pat .env`,
		`Add-Content -Val x .env`,
		`sc -va x .env`,
		`Set-Content -ErrorAction Stop .env x`,
		`Set-Content -ea 0 .env x`,
		`Remove-Item -Fil *.go .env`,
		`Move-Item tmp.txt -Dest .env`,
		`"x" | Out-File -Enc utf8 .env`,
	} {
		b, _ := json.Marshal(map[string]any{"command": cmd})
		if v := a.Adjudicate(ctx, inlineCall("PowerShell", string(b))); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
			t.Errorf("%q: got %v/%v, want Deny/POLICY_BLOCK", cmd, v.Kind, v.Reason)
		}
	}
	for _, cmd := range []string{
		`Set-Content -Va "see .env" notes.txt`,
		`Set-Content -Path notes.txt -Va "see .env"`,
		`Copy-Item .env -Dest C:/temp/y`,
		`Get-Content .env -Tail 5`,
	} {
		b, _ := json.Marshal(map[string]any{"command": cmd})
		if v := a.Adjudicate(ctx, inlineCall("PowerShell", string(b))); v.Kind != abi.VerdictAllow {
			t.Errorf("%q: got %v/%v, want Allow", cmd, v.Kind, v.Reason)
		}
	}
}
