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
	snapshot atomic.Pointer[lookupObservation]
}

type lookupObservation struct {
	result             *abi.Result
	historicalEngineNS int64
	lookupNS           int64
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
	if r == nil || result == nil {
		return false
	}
	value := r.snapshot.Load()
	return value != nil && value.result == result
}

// Timing returns the prior measured engine span and current lookup span for the
// exact accepted result. Only measured resident tier-2 entries supply a baseline;
// static, restored, search-cache, and unmeasured results remain unavailable.
func (r *LookupReceipt) Timing(result *abi.Result) (historicalEngineNS, lookupNS int64, ok bool) {
	if r == nil || result == nil {
		return 0, 0, false
	}
	value := r.snapshot.Load()
	if value == nil || value.result != result || value.historicalEngineNS <= 0 || value.lookupNS < 0 {
		return 0, 0, false
	}
	return value.historicalEngineNS, value.lookupNS, true
}

func lookupReceiptFromContext(ctx context.Context) *LookupReceipt {
	if ctx == nil {
		return nil
	}
	receipt, _ := ctx.Value(lookupReceiptContextKey{}).(*LookupReceipt)
	return receipt
}
