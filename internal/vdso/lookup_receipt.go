package vdso

import (
	"context"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// LookupReceipt identifies the exact result of a real successful VDSO lookup.
// It is scoped to one call and must not be retained in a request history. Result
// metadata cannot manufacture this evidence; only Lookup writes its private state.
type LookupReceipt struct {
	result atomic.Pointer[abi.Result]
}

type lookupReceiptContextKey struct{}

// WithLookupReceipt arms a fresh receipt for one lookup sequence. The derived
// context must reach Lookup, including through the default registered tier wrappers.
// Callers still apply their acceptance gates before using Matches as evidence.
func WithLookupReceipt(ctx context.Context) (context.Context, *LookupReceipt) {
	if ctx == nil {
		ctx = context.Background()
	}
	receipt := &LookupReceipt{}
	return context.WithValue(ctx, lookupReceiptContextKey{}, receipt), receipt
}

// Matches reports whether result is the exact nonnil result returned by the
// latest successful real Lookup in this call. A subsequent real lookup clears
// previous evidence, including on a miss. A zero or nil receipt matches nothing.
func (r *LookupReceipt) Matches(result *abi.Result) bool {
	return r != nil && result != nil && r.result.Load() == result
}

func lookupReceiptFromContext(ctx context.Context) *LookupReceipt {
	if ctx == nil {
		return nil
	}
	receipt, _ := ctx.Value(lookupReceiptContextKey{}).(*LookupReceipt)
	return receipt
}
