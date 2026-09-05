package adjudicator

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestBlockedPathGlobsDeniesPathAndCommand(t *testing.T) {
	p := Policy{
		Posture:          PostureDefaultOpen,
		BlockedPathGlobs: []string{".env", ".aws/", "id_rsa"},
	}
	a := New(p)
	ctx := context.Background()

	// 1. Direct path arg write
	cPath := inlineCall("Write", `{"path":".env","content":"SECRET=1"}`)
	vPath := a.Adjudicate(ctx, cPath)
	if vPath.Kind != abi.VerdictDeny {
		t.Fatalf("write to .env: want VerdictDeny, got %v", vPath.Kind)
	}
	if vPath.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("write to .env: want ReasonPolicyBlock, got %v", vPath.Reason)
	}
	if vPath.By != "monitor/credential-block" {
		t.Fatalf("write to .env: want By 'monitor/credential-block', got %q", vPath.By)
	}
	if vPath.Meta[abi.MetaDenyRule] != abi.DenyRuleCredentialPathBlock {
		t.Fatalf("write to .env: want deny_rule %q, got %q", abi.DenyRuleCredentialPathBlock, vPath.Meta[abi.MetaDenyRule])
	}

	// 2. Command write targeting blocked path
	cCmd := inlineCall("Bash", `{"command":"cat /dev/null > .aws/credentials"}`)
	vCmd := a.Adjudicate(ctx, cCmd)
	if vCmd.Kind != abi.VerdictDeny {
		t.Fatalf("command write to .aws/credentials: want VerdictDeny, got %v", vCmd.Kind)
	}
	if vCmd.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("command write to .aws/credentials: want ReasonPolicyBlock, got %v", vCmd.Reason)
	}
	if vCmd.By != "monitor/credential-block" {
		t.Fatalf("command write to .aws/credentials: want By 'monitor/credential-block', got %q", vCmd.By)
	}
	if vCmd.Meta[abi.MetaDenyRule] != abi.DenyRuleCredentialPathBlock {
		t.Fatalf("command write to .aws/credentials: want deny_rule %q, got %q", abi.DenyRuleCredentialPathBlock, vCmd.Meta[abi.MetaDenyRule])
	}

	// 3. Clean path under default open is allowed
	cClean := inlineCall("Write", `{"path":"src/main.go","content":"package main"}`)
	vClean := a.Adjudicate(ctx, cClean)
	if vClean.Kind != abi.VerdictAllow {
		t.Fatalf("clean write under default open: want VerdictAllow, got %v", vClean.Kind)
	}
}
