package adjudicator

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestBlockedPathAndSelfModifyPowerShell(t *testing.T) {
	ctx := context.Background()
	a := New(Policy{
		Posture:          PostureDefaultOpen,
		BlockedPathGlobs: []string{".env", ".aws/", "id_rsa"},
		SelfModifyGlobs:  []string{"internal/abi/"},
	})

	blockedCmds := []string{
		`Set-Content -Path .env -Value "SECRET=1"`,
		`set-content .env "SECRET=1"`,
		`Set-Content -LiteralPath C:\Users\alice\.aws\credentials -Value "x"`,
		`"SECRET=1" | Out-File -FilePath .env`,
		`"SECRET=1" | out-file .env`,
		`"SECRET=1" | Add-Content -Path .env`,
		`Copy-Item -Path tmp.txt -Destination .aws\credentials`,
		`Copy-Item tmp.txt .env`,
		`Move-Item -Path .env -Destination C:\temp\stolen`,
		`Remove-Item -Force .env`,
		`New-Item -Path .aws -Name credentials -ItemType File`,
		`Clear-Content -Path .env`,
		`Rename-Item -Path .env -NewName .env.bak`,
		`"x" | Tee-Object -FilePath .env`,
		`sc .env "SECRET=1"`,
		`ri -Force .env`,
		`pwsh -Command "Set-Content -Path .env -Value x"`,
		`echo x > .aws\credentials`,
	}
	for _, cmd := range blockedCmds {
		b, _ := json.Marshal(map[string]any{"command": cmd})
		v := a.Adjudicate(ctx, inlineCall("Bash", string(b)))
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
			t.Errorf("PowerShell blocked-path write %q: got %v/%v, want Deny/POLICY_BLOCK", cmd, v.Kind, v.Reason)
		}
	}

	selfModifyCmds := []string{
		`Set-Content -Path internal\abi\kernel.go -Value "package abi"`,
		`Remove-Item internal/abi/kernel.go`,
		`Clear-Content internal\abi\kernel.go`,
		`"x" | Out-File internal/abi/kernel.go`,
		`Rename-Item -Path internal/abi/kernel.go -NewName kernel.go.bak`,
	}
	for _, cmd := range selfModifyCmds {
		b, _ := json.Marshal(map[string]any{"command": cmd})
		v := a.Adjudicate(ctx, inlineCall("Bash", string(b)))
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSelfModify {
			t.Errorf("PowerShell self-modify write %q: got %v/%v, want Deny/SELF_MODIFY", cmd, v.Kind, v.Reason)
		}
	}

	// Read-only PowerShell cmdlets and copies FROM guarded paths to safe destinations stay allowed.
	allowedCmds := []string{
		`Get-Content .env`,
		`Get-Content internal\abi\kernel.go`,
		`Copy-Item .env C:\temp\backup.txt`,
		`Set-Content -Path src\main.go -Value "package main"`,
	}
	for _, cmd := range allowedCmds {
		b, _ := json.Marshal(map[string]any{"command": cmd})
		v := a.Adjudicate(ctx, inlineCall("Bash", string(b)))
		if v.Kind != abi.VerdictAllow {
			t.Errorf("allowed PowerShell cmd %q: got %v/%v, want Allow", cmd, v.Kind, v.Reason)
		}
	}
}

func TestBlockedPathAndSelfModifyNonKeywordWriteTools(t *testing.T) {
	ctx := context.Background()
	a := New(Policy{
		Posture:          PostureDefaultOpen,
		BlockedPathGlobs: []string{".env", ".aws/"},
		SelfModifyGlobs:  []string{"internal/abi/"},
	})

	blockedCalls := []struct {
		tool string
		args string
	}{
		{"save_file", `{"path":".env","content":"SECRET=1"}`},
		{"replace_in_file", `{"file_path":".env","old_str":"a","new_str":"b"}`},
		{"remove_file", `{"path":".env"}`},
		{"rename_file", `{"path":"tmp.txt","new_path":".env"}`},
		{"move_file", `{"path":"tmp.txt","destination":".aws/credentials"}`},
		{"copy_file", `{"path":"tmp.txt","dest":".env"}`},
		{"unlink_file", `{"path":".env"}`},
		{"custom_blob_stash", `{"path":".env","content":"SECRET=1"}`},
	}
	for _, tc := range blockedCalls {
		v := a.Adjudicate(ctx, inlineCall(tc.tool, tc.args))
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
			t.Errorf("non-keyword write %s(%s): got %v/%v, want Deny/POLICY_BLOCK", tc.tool, tc.args, v.Kind, v.Reason)
		}
	}

	selfModCalls := []struct {
		tool string
		args string
	}{
		{"save_file", `{"path":"internal/abi/kernel.go","content":"package abi"}`},
		{"replace_in_file", `{"filePath":"internal/abi/kernel.go","replacement":"x"}`},
		{"remove_file", `{"path":"internal/abi/kernel.go"}`},
		{"custom_blob_stash", `{"path":"internal/abi/kernel.go","data":"x"}`},
	}
	for _, tc := range selfModCalls {
		v := a.Adjudicate(ctx, inlineCall(tc.tool, tc.args))
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSelfModify {
			t.Errorf("non-keyword self-modify %s(%s): got %v/%v, want Deny/SELF_MODIFY", tc.tool, tc.args, v.Kind, v.Reason)
		}
	}

	// Read-only tools targeting guarded paths remain admitted (Allow or transparent Transform) under PostureDefaultOpen.
	for _, tc := range []struct {
		tool string
		args string
	}{
		{"Read", `{"path":".env"}`},
		{"read_file", `{"path":"internal/abi/kernel.go"}`},
		{"get_file", `{"file_path":".aws/credentials"}`},
	} {
		v := a.Adjudicate(ctx, inlineCall(tc.tool, tc.args))
		if v.Kind != abi.VerdictAllow && v.Kind != abi.VerdictTransform {
			t.Errorf("read tool %s(%s): got %v/%v, want Allow or Transform", tc.tool, tc.args, v.Kind, v.Reason)
		}
	}
}
