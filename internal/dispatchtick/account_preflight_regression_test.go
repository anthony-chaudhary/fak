package dispatchtick

import (
	"strings"
	"testing"
)

// TestNeedsLoginAccountIsRefusedBeforeSpawn captures #4772 at the routing-to-spawn
// boundary. A stale registry can still claim Available=true after the Claude guard
// has proved that the seat needs login and cannot serve. The guard posture must win.
func TestNeedsLoginAccountIsRefusedBeforeSpawn(t *testing.T) {
	cannotServe := false
	stale := AccountRow{
		Account:     ".claude-stale",
		Tag:         "stale",
		Product:     "claude",
		ModelTier:   1,
		Available:   true,
		LoginStatus: "needs_login",
		CanServe:    &cannotServe,
	}

	route := RouteAccount(AccountRouteInput{
		Rows:     []AccountRow{stale},
		Product:  "claude",
		WorkKind: "engineering",
	})
	if route.OK {
		t.Fatalf("stale route = %+v, want needs-login seat rejected", route)
	}
	if len(route.BlockedTargetAccounts) != 1 {
		t.Fatalf("blocked accounts = %+v, want stale seat retained as refusal evidence", route.BlockedTargetAccounts)
	}
	blocked := BlockedAccountFromRow(route.BlockedTargetAccounts[0])
	if blocked.LoginStatus != "needs_login" || blocked.CanServe == nil || *blocked.CanServe {
		t.Fatalf("blocked account = %+v, want needs_login and can_serve=false", blocked)
	}

	in := preflightInput()
	in.Kernel.Target = IntPtr(0)
	in.Account = AccountCheck{
		Available:       route.OK,
		Reason:          route.Reason,
		Blocked:         []string{stale.Tag},
		BlockedAccounts: []BlockedAccount{blocked},
	}
	verdict := EvaluatePreflight(in)

	spawnCalls := 0
	spawnIfAllowed := func() {
		if verdict.Verdict == PreflightOKVerdict {
			spawnCalls++
		}
	}
	spawnIfAllowed()

	if verdict.Verdict != PreflightRefuseNoAccount {
		t.Fatalf("verdict = %s (%s), want REFUSE_NO_ACCOUNT", verdict.Verdict, verdict.Reason)
	}
	if spawnCalls != 0 {
		t.Fatalf("spawn calls = %d, want refusal before spawn", spawnCalls)
	}
	if !strings.Contains(verdict.Reason, "no live credentials") {
		t.Fatalf("reason = %q, want login failure in captured refusal", verdict.Reason)
	}
	account, ok := verdict.Map()["account"].(map[string]any)
	if !ok {
		t.Fatalf("account evidence = %#v, want structured refusal", verdict.Map()["account"])
	}
	blockedAccounts, ok := account["blocked_accounts"].([]BlockedAccount)
	if !ok || len(blockedAccounts) != 1 || blockedAccounts[0].LoginStatus != "needs_login" {
		t.Fatalf("blocked account evidence = %#v, want typed needs_login posture", account["blocked_accounts"])
	}
}

func TestLoginGateKeepsHealthyClaudeAndCodexRoutable(t *testing.T) {
	canServe := true
	tests := []struct {
		name string
		row  AccountRow
	}{
		{
			name: "claude",
			row: AccountRow{
				Account: ".claude-ready", Tag: "ready", Product: "claude",
				ModelTier: 1, Available: true, LoginStatus: "ready", CanServe: &canServe,
			},
		},
		{
			name: "codex",
			row: AccountRow{
				Account: ".codex", Tag: "codex", Product: "codex",
				ModelTier: 1, Available: true,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			route := RouteAccount(AccountRouteInput{
				Rows: []AccountRow{tt.row}, Product: tt.row.Product, WorkKind: "engineering",
			})
			if !route.OK || route.Account.Tag != tt.row.Tag {
				t.Fatalf("route = %+v, want healthy %s seat selected", route, tt.name)
			}
		})
	}
}
