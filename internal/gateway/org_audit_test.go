package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/auditreceipt"
)

func TestGatewayAdjudicationEmitsAllowAndDenyOrgReceipts(t *testing.T) {
	got := make(chan auditreceipt.Receipt, 2)
	sink := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var rec auditreceipt.Receipt
		if err := json.NewDecoder(r.Body).Decode(&rec); err != nil {
			t.Errorf("decode: %v", err)
			return
		}
		got <- rec
		w.WriteHeader(http.StatusNoContent)
	}))
	defer sink.Close()
	exp, err := auditreceipt.New(auditreceipt.Config{Endpoint: sink.URL, DeviceID: "device-1", BufferPath: filepath.Join(t.TempDir(), "audit.jsonl"), Capacity: 64, Timeout: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	srv := &Server{orgAudit: exp}
	srv.logGatewayOperation("adjudicate", "trace", "search_kb", WireVerdict{Kind: "ALLOW"}, nil, time.Millisecond)
	srv.logGatewayOperation("adjudicate", "trace", "refund_payment", WireVerdict{Kind: "DENY", Reason: "POLICY_BLOCK"}, nil, time.Millisecond)
	seen := map[string]auditreceipt.Receipt{}
	for range 2 {
		select {
		case r := <-got:
			seen[r.Verdict] = r
		case <-time.After(time.Second):
			t.Fatal("receipt timeout")
		}
	}
	if seen["ALLOW"].Tool != "search_kb" || seen["DENY"].Reason != "POLICY_BLOCK" {
		t.Fatalf("receipts=%+v", seen)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := exp.Close(ctx); err != nil {
		t.Fatal(err)
	}
}
