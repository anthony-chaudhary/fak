package vdso

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestLookupReceiptMatchesOnlyExactRealVDSOHit(t *testing.T) {
	v := New(DefaultCacheSize)
	v.RegisterStatic("list_all", []byte(`{"items":["a"]}`))
	call := &abi.ToolCall{
		Tool: "list_all",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)},
		Meta: map[string]string{"readOnlyHint": "true", "idempotentHint": "true"},
	}

	ctxA, receiptA := WithLookupReceipt(context.Background())
	resultA, hit := v.Lookup(ctxA, call)
	if !hit || resultA == nil {
		t.Fatal("real vDSO lookup missed")
	}
	ctxB, receiptB := WithLookupReceipt(context.Background())
	resultB, hit := v.Lookup(ctxB, call)
	if !hit || resultB == nil {
		t.Fatal("second real vDSO lookup missed")
	}
	if !receiptA.Matches(resultA) || receiptA.Matches(resultB) {
		t.Fatal("first receipt did not bind only its exact returned result")
	}
	if !receiptB.Matches(resultB) || receiptB.Matches(resultA) {
		t.Fatal("second receipt did not bind only its exact returned result")
	}
	clone := *resultA
	clone.Meta = map[string]string{"served_by": "vdso", "tier": "3"}
	if receiptA.Matches(&clone) || receiptA.Matches(nil) {
		t.Fatal("copied result or metadata claim forged a lookup receipt")
	}

	if result, ok := v.Lookup(ctxA, &abi.ToolCall{Tool: "missing"}); ok || result != nil {
		t.Fatal("unknown tool unexpectedly hit")
	}
	if receiptA.Matches(resultA) {
		t.Fatal("later miss retained a stale successful lookup receipt")
	}
}
