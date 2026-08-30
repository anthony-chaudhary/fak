package vdso

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestProducerDiagnosticsFreshHitParity(t *testing.T) {
	ctx := context.Background()
	v := New(8)
	call := roCall("read_diagnostic_fixture", `{"id":"9948"}`)
	diagnostics, err := ProducerDiagnosticReceipt(ProducerDiagnostic{
		Severity: "warning",
		Code:     "schema.deprecated_field",
		Message:  "field legacy_name was accepted through the compatibility path",
	})
	if err != nil {
		t.Fatalf("ProducerDiagnosticReceipt: %v", err)
	}
	fresh := &abi.Result{
		Call:    call,
		Status:  abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"value":"fresh"}`)},
		Meta:    map[string]string{MetaProducerDiagnostics: diagnostics},
	}

	receipt, err := v.StoreResult(ctx, call, fresh)
	if err != nil {
		t.Fatalf("StoreResult: %v", err)
	}
	if receipt.ProducerDiagnostics != diagnostics {
		t.Fatalf("receipt diagnostics=%q, want fresh diagnostics %q", receipt.ProducerDiagnostics, diagnostics)
	}

	hit, ok := v.Lookup(ctx, call)
	if !ok {
		t.Fatal("Lookup missed cached result")
	}
	if got := hit.Meta[MetaProducerDiagnostics]; got != fresh.Meta[MetaProducerDiagnostics] {
		t.Fatalf("hit diagnostics=%q, want byte-identical fresh diagnostics %q", got, fresh.Meta[MetaProducerDiagnostics])
	}
	var decoded producerDiagnosticReceipt
	if err := json.Unmarshal([]byte(hit.Meta[MetaProducerDiagnostics]), &decoded); err != nil {
		t.Fatalf("decode hit diagnostics: %v", err)
	}
	if len(decoded.Diagnostics) != 1 {
		t.Fatalf("hit diagnostics count=%d, want exactly one replay", len(decoded.Diagnostics))
	}
}

func TestProducerDiagnosticsCleanHitHasNoStaleReceipt(t *testing.T) {
	ctx := context.Background()
	v := New(8)
	warned := roCall("read_diagnostic_fixture", `{"id":"warned"}`)
	clean := roCall("read_diagnostic_fixture", `{"id":"clean"}`)
	diagnostics, err := ProducerDiagnosticReceipt(ProducerDiagnostic{
		Severity: "warning",
		Code:     "input.normalized",
		Message:  "input used the canonical normalization path",
	})
	if err != nil {
		t.Fatalf("ProducerDiagnosticReceipt: %v", err)
	}

	if _, err := v.StoreResult(ctx, warned, &abi.Result{
		Call: warned, Status: abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte("warned")},
		Meta:    map[string]string{MetaProducerDiagnostics: diagnostics},
	}); err != nil {
		t.Fatalf("StoreResult warned: %v", err)
	}
	if _, err := v.StoreResult(ctx, clean, &abi.Result{
		Call: clean, Status: abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte("clean")},
	}); err != nil {
		t.Fatalf("StoreResult clean: %v", err)
	}

	hit, ok := v.Lookup(ctx, clean)
	if !ok {
		t.Fatal("Lookup missed clean cached result")
	}
	if got, exists := hit.Meta[MetaProducerDiagnostics]; exists || got != "" {
		t.Fatalf("clean hit retained stale diagnostics: exists=%v value=%q", exists, got)
	}
}

func TestProducerDiagnosticsReopenParity(t *testing.T) {
	ctx := context.Background()
	first := New(8)
	call := roCall("read_diagnostic_fixture", `{"id":"reopen"}`)
	diagnostics, err := ProducerDiagnosticReceipt(ProducerDiagnostic{
		Severity: "warning",
		Code:     "shape.compatibility",
		Message:  "shape was accepted by the stable compatibility rule",
	})
	if err != nil {
		t.Fatalf("ProducerDiagnosticReceipt: %v", err)
	}
	receipt, err := first.StoreResult(ctx, call, &abi.Result{
		Call: call, Status: abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte("reopened")},
		Meta:    map[string]string{MetaProducerDiagnostics: diagnostics},
	})
	if err != nil {
		t.Fatalf("StoreResult: %v", err)
	}

	reopened := New(8)
	if err := reopened.RestoreResult(ctx, call, receipt); err != nil {
		t.Fatalf("RestoreResult: %v", err)
	}
	hit, ok := reopened.Lookup(ctx, call)
	if !ok {
		t.Fatal("Lookup missed restored cached result")
	}
	if got := hit.Meta[MetaProducerDiagnostics]; got != diagnostics {
		t.Fatalf("restored diagnostics=%q, want %q", got, diagnostics)
	}
}

func TestProducerDiagnosticsRejectUnsafeReceipts(t *testing.T) {
	tests := []struct {
		name    string
		message string
	}{
		{name: "secret", message: "authorization: bearer redacted-but-still-forbidden"},
		{name: "timestamp", message: "generated at 2026-08-29T12:34:56Z"},
		{name: "request id", message: "request_id=abc123"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ProducerDiagnosticReceipt(ProducerDiagnostic{
				Severity: "warning",
				Code:     "producer.warning",
				Message:  tt.message,
			})
			if err == nil {
				t.Fatal("ProducerDiagnosticReceipt accepted unsafe diagnostic")
			}
		})
	}

	v := New(8)
	call := roCall("read_diagnostic_fixture", `{"id":"invalid"}`)
	_, err := v.StoreResult(context.Background(), call, &abi.Result{
		Call: call, Status: abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte("invalid")},
		Meta:    map[string]string{MetaProducerDiagnostics: `{"version":2,"diagnostics":[{"severity":"warning","code":"producer.warning","message":"stable"}]}`},
	})
	if err == nil || !strings.Contains(err.Error(), "incompatible") {
		t.Fatalf("StoreResult incompatible receipt error=%v, want incompatible rejection", err)
	}
}

func TestProducerDiagnosticsLegacyReceiptCompatibility(t *testing.T) {
	ctx := context.Background()
	v := New(8)
	call := roCall("read_diagnostic_fixture", `{"id":"legacy"}`)
	legacy := ResultStoreReceipt{
		Ref:         abi.Ref{Kind: abi.RefInline, Inline: []byte("legacy")},
		Resident:    true,
		Replication: ResultResidentOnly,
	}
	if err := v.RestoreResult(ctx, call, legacy); err != nil {
		t.Fatalf("RestoreResult legacy receipt: %v", err)
	}
	hit, ok := v.Lookup(ctx, call)
	if !ok {
		t.Fatal("Lookup missed legacy restored result")
	}
	if _, exists := hit.Meta[MetaProducerDiagnostics]; exists {
		t.Fatal("legacy receipt synthesized producer diagnostics")
	}
}
