package cachemeta

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fabricmap"
)

// fak-test:runtime fast est=20ms
func TestKVFabricRequiresConsumerReadyReceipt(t *testing.T) {
	now := time.Now()
	for _, test := range []struct {
		name    string
		change  func(*KVFabricReceipt)
		partial bool
		want    LookupKind
	}{
		{name: "complete", want: LookupHit},
		{name: "landing-only", partial: true, want: LookupFault},
		{name: "missing", change: func(f *KVFabricReceipt) { f.Execution.Hops = nil }, want: LookupFault},
		{name: "expired", change: func(f *KVFabricReceipt) { f.ValidUntil = now.Add(-time.Second) }, want: LookupFault},
		{name: "wrong-ready", change: func(f *KVFabricReceipt) { f.Execution.To = "landing" }, want: LookupFault},
		{name: "unbound-content", change: func(f *KVFabricReceipt) { f.Execution.Hops[1].Integrity.Digest = "other" }, want: LookupFault},
	} {
		t.Run(test.name, func(t *testing.T) {
			route := fabricmap.Route{
				From: "producer", To: "ready", Bytes: 96,
				Links: []fabricmap.Link{
					{ID: "network", From: "producer", To: "landing", Transport: "shm"},
					{ID: "local", From: "landing", To: "ready", Transport: "materialize"},
				},
			}
			authorized, err := fabricmap.AuthorizeRoute(route, fabricmap.Access{
				Principal: "receiver", Tenant: "test", Operation: fabricmap.OperationCopy,
			}, fabricmap.LabelAuthorizer{})
			if err != nil {
				t.Fatal(err)
			}
			execution, err := fabricmap.Execute(context.Background(), authorized, route.Bytes, fabricmap.ExecutorFunc(func(_ context.Context, request fabricmap.AuthorizationRequest, bytes uint64) (fabricmap.HopReceipt, error) {
				if test.partial && request.HopIndex == 1 {
					return fabricmap.HopReceipt{}, errors.New("local realization failed")
				}
				link := request.Link
				return fabricmap.HopReceipt{
					HopIndex: request.HopIndex, LinkID: link.ID, From: link.From, To: link.To, Transport: link.Transport,
					Bytes: bytes, StartedAt: now, CompletedAt: now, State: fabricmap.HopComplete,
					Integrity: fabricmap.Integrity{Algorithm: KVFabricIntegrity, Digest: "actual-rows"},
				}, nil
			}))
			if (err != nil) != test.partial {
				t.Fatalf("execution error=%v, partial=%v", err, test.partial)
			}
			proof := &KVFabricReceipt{
				PlanID: "opaque-plan", Provider: "receiver", Generation: 3,
				ValidUntil: now.Add(time.Minute), Landing: "landing", ReadyAt: now,
				SpanDigest: "span", ContentDigest: "actual-rows", LogicalBytes: 96,
				Route: route, Execution: execution,
			}
			if test.change != nil {
				test.change(proof)
			}
			entry := FromKVTransfer(KVTransfer{SpanDigest: "span", Outcome: KVTransferOK, Fabric: proof})
			if verdict := KVTransferVerdict(entry); verdict.Kind != test.want {
				t.Fatalf("verdict=%s, want %s", verdict.Kind, test.want)
			}
			if entry.Labels["fabric_plan_id"] != "opaque-plan" || entry.Labels["fabric_receipt"] == "" {
				t.Fatal("normalized event lost plan/per-hop evidence")
			}
		})
	}
	encoded, err := json.Marshal(KVTransfer{SpanDigest: "legacy"})
	if err != nil || bytes.Contains(encoded, []byte("\"Fabric\"")) {
		t.Fatalf("nil fabric changed legacy wire metadata: %s, %v", encoded, err)
	}
	if got := KVTransferVerdict(FromKVTransfer(KVTransfer{SpanDigest: "legacy"})); got.Kind != LookupHit {
		t.Fatalf("legacy one-hop transfer changed: %+v", got)
	}
}
