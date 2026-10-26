package remotemaintresult_test

import (
	"bytes"
	"context"
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ifc"
	"github.com/anthony-chaudhary/fak/internal/remotemaintresult"
	"github.com/anthony-chaudhary/fak/pkg/remotemaint"
)

// fak-test:runtime fast est=0.1s
func TestSourceBoundPIIProbe(t *testing.T) {
	now := time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)
	grant := remotemaint.Grant{
		ProfileID: "reviewed", OperationDigest: strings.Repeat("b", 64), ContextID: strings.Repeat("c", 64),
		Target: "svc@node.local", HostKey: "SHA256:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
		Operation: remotemaint.OperationSystemdProfileInstallReload, PayloadSHA256: strings.Repeat("a", 64),
		Holder: "guard", BootID: "boot-1", PriorStateSHA256: strings.Repeat("d", 64),
		Paths:        []string{"/etc/systemd/system/fak.service"},
		PathPrefixes: []remotemaint.PathPrefix{{Directory: "/var/tmp", FilenamePrefix: "fak-stage."}},
		Units:        []string{"fak.service"}, ExpiresAt: now.Add(time.Minute),
		LeasePins: []remotemaint.LeasePin{{NodeID: "node-1", ID: "lease-1", Epoch: 1, ExpiresAt: now.Add(time.Hour)}},
	}
	receipt := remotemaint.Receipt{
		Schema: remotemaint.ReceiptSchema, GrantDigest: remotemaint.GrantDigest(grant), ProfileID: grant.ProfileID,
		OperationDigest: grant.OperationDigest, ContextID: grant.ContextID, Target: grant.Target, HostKey: grant.HostKey,
		Operation: grant.Operation, BootID: grant.BootID, StartedAt: now, EndedAt: now,
		Outcome: remotemaint.OutcomeApplied, Verified: true,
	}
	marshal := func(t *testing.T, r remotemaint.Receipt) []byte {
		t.Helper()
		b, err := json.Marshal(r)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	body := marshal(t, receipt)
	call := &abi.ToolCall{Tool: "fak_remote_maintenance", TraceID: "trace-1", SeqNo: 7,
		Meta: map[string]string{"remote_maintenance_grant": receipt.GrantDigest, "source_bound": "true"}}
	ctx := remotemaintresult.Bind(context.Background(), call, grant, body)
	refuse := func(t *testing.T, ctx context.Context, c *abi.ToolCall, raw []byte) {
		t.Helper()
		if probe, ok := remotemaintresult.PIIProbe(ctx, c, raw); ok || probe != nil {
			t.Fatalf("unexpected probe: %s", probe)
		}
		if remotemaintresult.TrustedControlReceipt(ctx, c, raw) {
			t.Fatal("unbound control receipt gained trust")
		}
	}
	t.Run("only-target-changes", func(t *testing.T) {
		original := bytes.Clone(body)
		probe, ok := remotemaintresult.PIIProbe(ctx, call, body)
		if !ok {
			t.Fatal("source-bound receipt refused")
		}
		if !remotemaintresult.TrustedControlReceipt(ctx, call, body) {
			t.Fatal("exact install control receipt was not recognized")
		}
		var got remotemaint.Receipt
		if err := json.Unmarshal(probe, &got); err != nil {
			t.Fatal(err)
		}
		if got.Target == receipt.Target || strings.Contains(got.Target, "@") {
			t.Fatal("probe retains email-shaped target")
		}
		got.Target = receipt.Target
		if !reflect.DeepEqual(got, receipt) || !bytes.Equal(body, original) {
			t.Fatal("probe changed another field or original payload")
		}
		probe[0] = '!'
		if !bytes.Equal(body, original) {
			t.Fatal("probe aliases payload")
		}
	})
	t.Run("no-typed-authority", func(t *testing.T) {
		refuse(t, context.Background(), call, body)
		refuse(t, nil, call, body)
		refuse(t, ctx, nil, body)
		copyCall := *call
		refuse(t, ctx, &copyCall, body)
		for _, mutate := range []func(){
			func() { call.TraceID = "replay" }, func() { call.SeqNo++ }, func() { call.Tool = "email" },
		} {
			saved := *call
			mutate()
			refuse(t, ctx, call, body)
			*call = saved
		}
	})
	t.Run("exact-bytes", func(t *testing.T) {
		refuse(t, ctx, call, append(bytes.Clone(body), '\n'))
		changed := receipt
		changed.Target = "other@node.local"
		refuse(t, ctx, call, marshal(t, changed))
		changed = receipt
		changed.Outcome = remotemaint.OutcomeUnknownEffect
		refuse(t, ctx, call, marshal(t, changed))
	})
	t.Run("strict-json", func(t *testing.T) {
		for _, raw := range [][]byte{
			append(bytes.Clone(body), []byte(" {}")...),
			append(bytes.Clone(body[:len(body)-1]), []byte(`,"unknown":true}`)...),
			append(bytes.Clone(body[:len(body)-1]), []byte(`,"target":"svc@node.local"}`)...),
			bytes.Replace(body, []byte(`"schema":`), []byte(`"Schema":`), 1),
		} {
			bound := remotemaintresult.Bind(context.Background(), call, grant, raw)
			refuse(t, bound, call, raw)
		}
	})
	t.Run("receipt-identity", func(t *testing.T) {
		for _, change := range []func(*remotemaint.Receipt){
			func(r *remotemaint.Receipt) { r.Schema += "x" }, func(r *remotemaint.Receipt) { r.GrantDigest += "x" },
			func(r *remotemaint.Receipt) { r.ProfileID += "x" }, func(r *remotemaint.Receipt) { r.OperationDigest += "x" },
			func(r *remotemaint.Receipt) { r.ContextID += "x" }, func(r *remotemaint.Receipt) { r.Target += "x" },
			func(r *remotemaint.Receipt) { r.HostKey += "x" }, func(r *remotemaint.Receipt) { r.Operation += "x" },
			func(r *remotemaint.Receipt) { r.BootID += "x" },
		} {
			r := receipt
			change(&r)
			raw := marshal(t, r)
			refuse(t, remotemaintresult.Bind(context.Background(), call, grant, raw), call, raw)
		}
	})
	t.Run("full-grant-binding", func(t *testing.T) {
		for _, change := range []func(*remotemaint.Grant){
			func(g *remotemaint.Grant) { g.Holder += "x" }, func(g *remotemaint.Grant) { g.PayloadSHA256 += "x" },
			func(g *remotemaint.Grant) { g.PriorStateSHA256 += "x" }, func(g *remotemaint.Grant) { g.ExpiresAt = g.ExpiresAt.Add(time.Second) },
			func(g *remotemaint.Grant) { g.Paths = []string{"/other"} }, func(g *remotemaint.Grant) { g.Units = []string{"other.service"} },
			func(g *remotemaint.Grant) {
				g.PathPrefixes = []remotemaint.PathPrefix{{Directory: "/other", FilenamePrefix: "x"}}
			},
			func(g *remotemaint.Grant) { g.LeasePins = []remotemaint.LeasePin{{NodeID: "other"}} },
		} {
			g := grant
			change(&g)
			refuse(t, remotemaintresult.Bind(context.Background(), call, g, body), call, body)
		}
	})
	t.Run("source-target-structure", func(t *testing.T) {
		for _, target := range []string{"node.local", "a@@node.local", "-oProxyCommand@node.local", "svc@bad host", "svc@node.local/secret", "svc@node.local:22", "svc@-node.local"} {
			g, r := grant, receipt
			g.Target = target
			r.Target, r.GrantDigest = target, remotemaint.GrantDigest(g)
			raw := marshal(t, r)
			refuse(t, remotemaintresult.Bind(context.Background(), call, g, raw), call, raw)
		}
		wrongCall := *call
		wrongCall.Tool = "Fak_Remote_Maintenance"
		refuse(t, remotemaintresult.Bind(context.Background(), &wrongCall, grant, body), &wrongCall, body)
	})
	t.Run("other-email-survives", func(t *testing.T) {
		g, r := grant, receipt
		g.BootID = "other@example.test"
		r.BootID, r.GrantDigest = g.BootID, remotemaint.GrantDigest(g)
		raw := marshal(t, r)
		probe, ok := remotemaintresult.PIIProbe(remotemaintresult.Bind(context.Background(), call, g, raw), call, raw)
		if !ok || !bytes.Contains(probe, []byte(g.BootID)) {
			t.Fatal("probe concealed unrelated email")
		}
	})
	t.Run("native-control-union", func(t *testing.T) {
		root := t.TempDir()
		for _, operation := range []string{remotemaint.OperationMaintainLeaseAcquire, remotemaint.OperationMaintainLeaseRelease} {
			g, r := grant, receipt
			g.Operation, g.Target, g.HostKey, g.BootID, g.PayloadSHA256, g.PriorStateSHA256 = operation, "", "", "", "", ""
			g.Paths, g.PathPrefixes, g.Units = []string{filepath.Join(root, ".fak", "nvlease")}, nil, nil
			g.Reservation = &remotemaint.TrustedReservation{WorkspaceRoot: root, NodeIDs: []string{"node-1"}, Kind: "maintain", TTLSeconds: 120, CatalogGeneration: "catalog", ExpectedStoreEpoch: 1}
			r.ReservationLeases = []remotemaint.LeasePin{{NodeID: "node-1", ID: "native-lease", Epoch: 2, ExpiresAt: now.Add(time.Minute)}}
			if operation == remotemaint.OperationMaintainLeaseAcquire {
				g.LeasePins = nil
			} else {
				g.Reservation.TTLSeconds = 0
				g.Reservation.ExpectedStoreEpoch = 2
				g.LeasePins = append([]remotemaint.LeasePin(nil), r.ReservationLeases...)
			}
			r.Operation, r.Target, r.HostKey, r.BootID, r.GrantDigest = operation, "", "", "", remotemaint.GrantDigest(g)
			raw := marshal(t, r)
			bound := remotemaintresult.Bind(context.Background(), call, g, raw)
			if !remotemaintresult.TrustedControlReceipt(bound, call, raw) {
				t.Fatalf("closed native receipt not recognized: %s", operation)
			}
			if probe, ok := remotemaintresult.PIIProbe(bound, call, raw); ok || probe != nil {
				t.Fatal("native control binding granted SSH PII exception")
			}
		}
	})
	t.Run("observations-and-unknown-stay-tainted", func(t *testing.T) {
		for _, change := range []func(*remotemaint.Receipt){
			func(r *remotemaint.Receipt) { r.Verified = false },
			func(r *remotemaint.Receipt) { r.Outcome = remotemaint.OutcomeUnknownEffect },
			func(r *remotemaint.Receipt) {
				r.Readback = &remotemaint.SystemdProfileReadback{UnitGraph: "remote observations"}
			},
		} {
			r := receipt
			change(&r)
			raw := marshal(t, r)
			bound := remotemaintresult.Bind(context.Background(), call, grant, raw)
			if remotemaintresult.TrustedControlReceipt(bound, call, raw) {
				t.Fatal("non-control evidence gained trust")
			}
		}
		g, r := grant, receipt
		g.Operation, r.Operation = remotemaint.OperationSystemdProfileReadback, remotemaint.OperationSystemdProfileReadback
		r.GrantDigest, r.Outcome = remotemaint.GrantDigest(g), remotemaint.OutcomeUnchanged
		raw := marshal(t, r)
		if remotemaintresult.TrustedControlReceipt(remotemaintresult.Bind(context.Background(), call, g, raw), call, raw) {
			t.Fatal("readback operation gained control trust without a graph")
		}
	})
	t.Run("stamp-preserves-quarantine-and-trace-high-water", func(t *testing.T) {
		ledger := ifc.NewLedger()
		stamp := ifc.NewStampGate(ledger, ifc.Policy{})
		result := func(taint abi.TaintLabel, raw []byte) *abi.Result {
			return &abi.Result{Payload: abi.Ref{Kind: abi.RefInline, Inline: raw, Taint: taint}}
		}
		r := result(abi.TaintTainted, body)
		stamp.Admit(ctx, call, r)
		if r.Payload.Taint != abi.TaintTrusted || ledger.Level(call.TraceID) != abi.TaintTrusted {
			t.Fatal("source-bound control receipt tainted a clean trace")
		}
		ledger.Raise(call.TraceID, abi.TaintTainted)
		stamp.Admit(ctx, call, result(abi.TaintTainted, body))
		if ledger.Level(call.TraceID) != abi.TaintTainted {
			t.Fatal("control receipt lowered preexisting trace taint")
		}
		r = result(abi.TaintQuarantined, body)
		stamp.Admit(ctx, call, r)
		if r.Payload.Taint != abi.TaintQuarantined || ledger.Level(call.TraceID) != abi.TaintQuarantined {
			t.Fatal("control binding bypassed quarantine")
		}
		r = result(abi.TaintTainted, append(bytes.Clone(body), '\n'))
		stamp.Admit(ctx, call, r)
		if r.Payload.Taint != abi.TaintTainted {
			t.Fatal("changed result bytes retained control trust")
		}
	})
}
