package gateway

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// approvedBashAdj admits every bash call; the approved command carries the
// floor's operator-approval record, as adjudicator.finish stamps it.
type approvedBashAdj struct{ approvedCmd string }

func (approvedBashAdj) Caps() []abi.Capability { return nil }
func (a approvedBashAdj) Adjudicate(ctx context.Context, call *abi.ToolCall) abi.Verdict {
	if call.Tool != "bash" {
		return (toolAdj{}).Adjudicate(ctx, call)
	}
	v := abi.Verdict{Kind: abi.VerdictAllow, By: "monitor"}
	var args map[string]any
	_ = json.Unmarshal(call.Args.Inline, &args)
	if cmd, _ := args["command"].(string); cmd == a.approvedCmd {
		v.Meta = map[string]string{
			"posture":                          adjudicator.PostureOperatorApproved,
			adjudicator.MetaOperatorApprovalID: adjudicator.OperatorApprovalID(call.Tool, args),
		}
	}
	return v
}

// An operator-approved call passes the workspace lease admission that refuses a
// shell command with an unprovable write footprint (witnessed live 2026-10-23:
// the approved rm -rf was admitted by the floor, then refused here). An
// unapproved call is still refused, now offering its approval id.
// fak-test:runtime fast est=1s
func TestLeaseAdmissionHonorsOperatorApproval(t *testing.T) {
	const approved = "rm -rf build"
	newTestServer(t)
	abi.RegisterAdjudicator(-1, approvedBashAdj{approvedCmd: approved})
	srv, err := New(Config{EngineID: "test", Model: "test-model", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	root := t.TempDir()
	SetWorkspaceLeaseAdmissionProvider(func(context.Context) (WorkspaceLeaseAdmissionView, error) {
		return WorkspaceLeaseAdmissionView{WorkspaceRoot: root}, nil
	})
	t.Cleanup(func() { SetWorkspaceLeaseAdmissionProvider(nil) })

	ok, rpcErr := callFakAdjudicate(t, srv, `{"tool":"bash","arguments":{"command":"`+approved+`"}}`)
	if rpcErr != nil || ok.Verdict.Kind != "ALLOW" || ok.Verdict.By != "lease-admission-operator-approved" {
		t.Fatalf("approved call: response=%+v error=%+v, want ALLOW by lease-admission-operator-approved", ok, rpcErr)
	}
	if ok.Verdict.Detail["would_deny_by"] != "lease-admission" || ok.Verdict.Detail["workspace_admission"] != adjudicator.PostureOperatorApproved {
		t.Fatalf("approved call evidence = %v", ok.Verdict.Detail)
	}

	const other = "python -c 'print(1)'"
	refused, rpcErr := callFakAdjudicate(t, srv, `{"tool":"bash","arguments":{"command":"python -c 'print(1)'"}}`)
	if rpcErr != nil || refused.Verdict.Kind != "DENY" || refused.Verdict.By != "lease-admission" {
		t.Fatalf("unapproved call: response=%+v error=%+v, want DENY by lease-admission", refused, rpcErr)
	}
	want := adjudicator.OperatorApprovalID("bash", map[string]any{"command": other})
	if got := refused.Verdict.Detail[detailOperatorApprovalID]; got != want {
		t.Fatalf("unapproved lease refusal offers id %q, want %q", got, want)
	}
}
