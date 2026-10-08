package ctxmmu

import (
	"context"
	"sort"
	"sync"
)

// Reasons an oversize-but-benign page-out was suppressed. They are the closed label
// set of the gateway's suppressed-paging counter.
const (
	// PagingSuppressedHarnessCannotRestore: the client advertised no restore tool, so a
	// {"_paged":true} stub would be an unrecoverable hole the model loops on.
	PagingSuppressedHarnessCannotRestore = "harness_cannot_restore"
	// PagingSuppressedTrailingResult: the result answers the current turn's tool call;
	// stubbing what the model just asked for forces an immediate re-request.
	PagingSuppressedTrailingResult = "trailing_result"
	// PagingSuppressedOperatorOff: the operator disabled paging (env or request header).
	PagingSuppressedOperatorOff = "operator_off"
)

type oversizePagingKey struct{}

// WithOversizePagingSuppressed returns a ctx under which Admit never pages an
// oversize-but-benign result out to a pointer stub; the bytes are admitted as-is.
// Quarantine, injection, secret, and semantic screening are unaffected. An empty
// reason returns ctx unchanged.
func WithOversizePagingSuppressed(ctx context.Context, reason string) context.Context {
	if reason == "" {
		return ctx
	}
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, oversizePagingKey{}, reason)
}

// OversizePagingSuppressedReason reports the suppression reason carried by ctx, or ""
// when oversize paging is enabled (the default).
func OversizePagingSuppressedReason(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	reason, _ := ctx.Value(oversizePagingKey{}).(string)
	return reason
}

var suppressedPaging struct {
	mu       sync.Mutex
	byReason map[string]int64
}

func recordSuppressedPaging(reason string) {
	suppressedPaging.mu.Lock()
	if suppressedPaging.byReason == nil {
		suppressedPaging.byReason = map[string]int64{}
	}
	suppressedPaging.byReason[reason]++
	suppressedPaging.mu.Unlock()
}

// SuppressedPagingCount is one row of the process-wide suppressed-paging counter.
type SuppressedPagingCount struct {
	Reason string
	Count  int64
}

// SuppressedPagingCounts returns, sorted by reason, how many oversize page-outs Admit
// skipped because the ctx suppressed them. The known reasons are always present.
func SuppressedPagingCounts() []SuppressedPagingCount {
	suppressedPaging.mu.Lock()
	defer suppressedPaging.mu.Unlock()
	rows := map[string]int64{
		PagingSuppressedHarnessCannotRestore: 0,
		PagingSuppressedTrailingResult:       0,
		PagingSuppressedOperatorOff:          0,
	}
	for k, v := range suppressedPaging.byReason {
		rows[k] = v
	}
	out := make([]SuppressedPagingCount, 0, len(rows))
	for k, v := range rows {
		out = append(out, SuppressedPagingCount{Reason: k, Count: v})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Reason < out[j].Reason })
	return out
}

// oversizePagingSuppressed reports whether ctx suppresses the oversize page-out and,
// when it does, counts the skipped page-out under its reason.
func (m *MMU) oversizePagingSuppressed(ctx context.Context) bool {
	reason := OversizePagingSuppressedReason(ctx)
	if reason == "" {
		return false
	}
	recordSuppressedPaging(reason)
	return true
}
