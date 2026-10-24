package main

import (
	"bytes"
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// guard_allow_approve_test.go — `fak guard allow --approve`: the human operator's
// per-call override, exercised against the SHIPPED default floor.

func approveTestCall(tool, args string) *abi.ToolCall {
	return &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(args)}}
}

func defaultFloorWith(t *testing.T, ov guardAllowOverlay) *adjudicator.Adjudicator {
	t.Helper()
	rt, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		t.Fatalf("parse default floor: %v", err)
	}
	guardApplyAllowOverlay(&rt, ov)
	return adjudicator.New(rt.Adjudicator)
}

// The end-to-end operator flow: the default floor refuses rm -rf and offers an id;
// the operator approves it through the overlay; the identical call is admitted and
// a different destructive call is not.
func TestGuardAllowApproveAdmitsTheRefusedCallOnTheDefaultFloor(t *testing.T) {
	const rmBuild = `{"command":"rm -rf build"}`
	refused := defaultFloorWith(t, guardAllowOverlay{}).Adjudicate(context.Background(), approveTestCall("Bash", rmBuild))
	if refused.Kind != abi.VerdictDeny || refused.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("default floor: kind = %v reason %s, want Deny/POLICY_BLOCK", refused.Kind, abi.ReasonName(refused.Reason))
	}
	id := refused.Meta[adjudicator.MetaOperatorApprovalID]
	if !guardOperatorApprovalIDRe.MatchString(id) {
		t.Fatalf("refusal offers approval id %q, want ap-<12 hex>", id)
	}

	path := filepath.Join(t.TempDir(), "allow.json")
	ov := guardAllowOverlay{Version: guardAllowOverlayVersion}
	var out, errb bytes.Buffer
	if code := runGuardAllowApprove(&out, &errb, path, &ov, []string{id}, 0, false, false); code != 0 {
		t.Fatalf("approve exit %d: %s", code, errb.String())
	}
	loaded, err := loadGuardAllowOverlay(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if !guardAllowNamePresent(loaded.Approve, id) {
		t.Fatalf("overlay approve list %v lacks %s", loaded.Approve, id)
	}
	stamp, err := time.Parse(time.RFC3339, loaded.Expiry[id])
	if err != nil {
		t.Fatalf("approval has no expiry stamp: %v", loaded.Expiry)
	}
	if left := time.Until(stamp); left <= 0 || left > guardOperatorApprovalDefaultTTL+time.Minute {
		t.Fatalf("approval expiry %s is not about the %s default", stamp, guardOperatorApprovalDefaultTTL)
	}

	a := defaultFloorWith(t, loaded)
	if v := a.Adjudicate(context.Background(), approveTestCall("Bash", rmBuild)); v.Kind != abi.VerdictAllow || v.Meta["posture"] != adjudicator.PostureOperatorApproved {
		t.Fatalf("approved call: kind = %v meta %v, want an operator-approved Allow", v.Kind, v.Meta)
	}
	if v := a.Adjudicate(context.Background(), approveTestCall("Bash", `{"command":"rm -rf src"}`)); v.Kind != abi.VerdictDeny {
		t.Fatalf("a different rm -rf: kind = %v, want it still refused", v.Kind)
	}
}

func TestGuardAllowApproveRejectsMalformedIDs(t *testing.T) {
	path := filepath.Join(t.TempDir(), "allow.json")
	for _, args := range [][]string{nil, {"Bash"}, {"ap-XYZ"}, {"ap-0123456789abc"}} {
		ov := guardAllowOverlay{Version: guardAllowOverlayVersion}
		var out, errb bytes.Buffer
		if code := runGuardAllowApprove(&out, &errb, path, &ov, args, 0, false, false); code != 2 {
			t.Errorf("args %v: exit %d, want 2", args, code)
		}
	}
	ov := guardAllowOverlay{Version: guardAllowOverlayVersion}
	var out, errb bytes.Buffer
	if code := runGuardAllowApprove(&out, &errb, path, &ov, []string{"ap-0123456789ab"}, 0, false, true); code != 2 {
		t.Errorf("conflicting mode: exit %d, want 2", code)
	}
}

func TestGuardAllowApproveRevokeAndExpiry(t *testing.T) {
	const id = "ap-0123456789ab"
	path := filepath.Join(t.TempDir(), "allow.json")
	ov := guardAllowOverlay{Version: guardAllowOverlayVersion}
	var out, errb bytes.Buffer
	if code := runGuardAllowApprove(&out, &errb, path, &ov, []string{id}, time.Hour, false, false); code != 0 {
		t.Fatalf("approve exit %d: %s", code, errb.String())
	}
	if code := runGuardAllowApprove(&out, &errb, path, &ov, []string{id}, 0, true, false); code != 0 {
		t.Fatalf("revoke exit %d: %s", code, errb.String())
	}
	loaded, err := loadGuardAllowOverlay(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(loaded.Approve) != 0 || loaded.Expiry[id] != "" {
		t.Fatalf("revoked approval still present: %+v", loaded)
	}

	expired := guardAllowOverlay{Version: guardAllowOverlayVersion, Approve: []string{id}, Expiry: map[string]string{id: "2000-01-01T00:00:00Z"}}
	if got, _ := guardAllowDropExpired(expired, time.Now()); len(got.Approve) != 0 {
		t.Fatalf("expired approval not dropped: %v", got.Approve)
	}
}

// The agent must not be able to approve its own refused call: the default floor
// refuses the approve verb from every shell tool, however it is spelled.
func TestDefaultFloorRefusesAgentSelfApproval(t *testing.T) {
	a := defaultFloorWith(t, guardAllowOverlay{})
	for _, tc := range []struct{ tool, args string }{
		{"Bash", `{"command":"fak guard allow --approve ap-0123456789ab"}`},
		{"Bash", `{"command":"fak-run fak guard allow --session --approve ap-0123456789ab"}`},
		{"Bash", `{"command":"go run ./cmd/fak guard allow --approve ap-0123456789ab"}`},
		{"PowerShell", `{"command":"fak.exe guard allow --approve ap-0123456789ab"}`},
		{"shell_command", `{"command":"fak guard allow --approve ap-0123456789ab"}`},
		{"exec_command", `{"cmd":"fak guard allow --approve ap-0123456789ab"}`},
	} {
		v := a.Adjudicate(context.Background(), approveTestCall(tc.tool, tc.args))
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
			t.Errorf("%s %s: kind = %v reason %s, want Deny/POLICY_BLOCK", tc.tool, tc.args, v.Kind, abi.ReasonName(v.Reason))
		}
	}
	// Reading the overlay is still fine.
	if v := a.Adjudicate(context.Background(), approveTestCall("Bash", `{"command":"fak guard allow --list"}`)); v.Kind != abi.VerdictAllow {
		t.Errorf("fak guard allow --list: kind = %v reason %s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}
}
