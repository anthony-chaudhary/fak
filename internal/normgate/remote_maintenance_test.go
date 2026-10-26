package normgate_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob"
	"github.com/anthony-chaudhary/fak/internal/normgate"
	"github.com/anthony-chaudhary/fak/internal/remotemaintresult"
	"github.com/anthony-chaudhary/fak/pkg/remotemaint"
)

// fak-test:runtime fast est=1s lane=short
func TestRemoteMaintenanceControlTargetPIIContext(t *testing.T) {
	tests := []struct {
		name, profile, target               string
		mode                                string
		wantAllow, wantSecret, wantGoverned bool
	}{
		{name: "source bound target remains byte identical", wantAllow: true},
		{name: "no context falls back", mode: "no-context"},
		{name: "forged metadata falls back", mode: "forged-meta"},
		{name: "changed call falls back", mode: "changed-call"},
		{name: "changed payload falls back", mode: "changed-payload"},
		{name: "unrelated receipt email still transforms", profile: "ops@example.com"},
		{name: "source target overlapping SSN still transforms", target: "svc@123-45-6789.example"},
		{name: "source target overlapping reversed SSN stays governed", target: "svc@9876-54-321.example", wantGoverned: true},
		{name: "source target overlapping base64 SSN stays governed", target: "svc@MTIzLTQ1LTY3ODkg.example", wantGoverned: true},
		{name: "source hostname containing encoded email stays governed", target: "svc@YWxpY2VAZXhhbXBsZS5jb20.example", wantGoverned: true},
		{name: "secret shaped username stays on secret floor", target: "AKIAQWERTYUIOPASDFGH@node.local", wantSecret: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			call, grant, body := remoteMaintenanceControlFixture(t, test.profile, test.target)
			ctx := remotemaintresult.Bind(context.Background(), call, grant, body)
			switch test.mode {
			case "no-context":
				ctx = context.Background()
			case "forged-meta":
				ctx = context.Background()
				call.Meta = map[string]string{"remote_maintenance_control_target": "public"}
			case "changed-call":
				changed := *call
				call = &changed
			case "changed-payload":
				body = append(append([]byte(nil), body...), ' ')
			}
			original := append([]byte(nil), body...)
			result := &abi.Result{Call: call, Payload: abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body)), Taint: abi.TaintTainted, Scope: abi.ScopeAgent}}
			verdict := normgate.New().Admit(ctx, call, result)
			if test.wantAllow {
				if verdict.Kind != abi.VerdictAllow || verdict.By != "normgate" || !bytes.Equal(result.Payload.Inline, original) {
					t.Fatalf("verdict=%+v payload=%q, want byte-identical Allow", verdict, result.Payload.Inline)
				}
				return
			}
			if test.wantSecret {
				if verdict.Kind == abi.VerdictAllow || verdict.Reason != abi.ReasonSecretRedacted && verdict.Reason != abi.ReasonSecretExfil {
					t.Fatalf("verdict=%+v, want existing secret floor", verdict)
				}
				return
			}
			if test.wantGoverned {
				if verdict.Kind == abi.VerdictAllow || verdict.Kind == abi.VerdictDefer || bytes.Equal([]byte(resolve(t, ctx, result.Payload)), original) {
					t.Fatalf("verdict=%+v payload remained unchanged, want existing PII floor", verdict)
				}
				return
			}
			if verdict.Kind != abi.VerdictTransform || verdict.Reason != abi.ReasonPIIRedacted {
				t.Fatalf("verdict=%+v, want existing PII transform", verdict)
			}
		})
	}
}

func remoteMaintenanceControlFixture(t *testing.T, profile, target string) (*abi.ToolCall, remotemaint.Grant, []byte) {
	t.Helper()
	if profile == "" {
		profile = "raider-ssh-first-v1"
	}
	if target == "" {
		target = "fak@node.local"
	}
	now := time.Now().UTC()
	expires := now.Add(5 * time.Minute)
	grant := remotemaint.Grant{
		ProfileID: profile, OperationDigest: fmt.Sprintf("%064x", 1), ContextID: fmt.Sprintf("%064x", 2), Target: target,
		HostKey: "SHA256:XtPqkn4E/+n6N9tBuK7dbyLtGF67osyUMnIaWCq43rM", Operation: remotemaint.OperationSystemdProfileInstallReload,
		PayloadSHA256: fmt.Sprintf("%064x", 3), Holder: "raider-recovery", BootID: "boot-1", PriorStateSHA256: fmt.Sprintf("%064x", 4),
		Paths: []string{"/etc/systemd/system/fak.service.d/50-fak.conf"}, PathPrefixes: []remotemaint.PathPrefix{{Directory: "/var/tmp", FilenamePrefix: "fak-raider-recovery."}},
		Units: []string{"fak.service"}, LeasePins: []remotemaint.LeasePin{{NodeID: "owned-raider", ID: "lease-1", Epoch: 1, ExpiresAt: expires}}, ExpiresAt: expires,
	}
	receipt := remotemaint.Receipt{Schema: remotemaint.ReceiptSchema, GrantDigest: remotemaint.GrantDigest(grant), ProfileID: grant.ProfileID, OperationDigest: grant.OperationDigest, ContextID: grant.ContextID, Target: grant.Target, HostKey: grant.HostKey, Operation: grant.Operation, BootID: grant.BootID, StartedAt: now, EndedAt: now.Add(time.Second), Outcome: remotemaint.OutcomeApplied, Verified: true, BackupDisposition: remotemaint.BackupNotRequired}
	body, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	return &abi.ToolCall{Tool: "fak_remote_maintenance", TraceID: "trace-control-result", SeqNo: 7}, grant, body
}
