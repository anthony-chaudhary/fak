package agent

import (
	"context"
	"errors"
)

// inkernel_cache_prime.go semantics (CW-03, fak#13351), implemented here alongside the
// decode seam it reuses:
//
// PrimeCacheState runs the EXACT production prefill/admission path (the same lookup,
// suffix prefill, snapshot and tree admission that generateReusedContextWithBias
// performs in steps 1–3) and then EXITS BEFORE the decode lane. Its purpose is cache
// MATERIALIZATION: the outcome is persisted state a later demand turn restores, not a
// generated token. Consequently a prime:
//
//   - never samples, emits, runs a tool or bills demand usage (the purpose returns at
//     the step-3b seam, before any decodeLane exists);
//   - is excluded from demand turn-tax: recordTurnTax is a per-demand-turn decision and
//     a populate is not demand traffic, so a prime does not append to the ledger;
//   - reports a closed receipt: Admitted / Cold / Unsupported.

// ErrCachePrimeUnsupported is the fail-closed refusal for a planner whose prefix-reuse
// path cannot populate a cache (no tree, or a model/backend pair the reuse path does not
// support). It mirrors the "explicit unsupported status" the issue requires: a caller
// must never read a zero-value receipt as a successful prime.
var ErrCachePrimeUnsupported = errors.New("agent: cache populate is unsupported on this planner")

// Closed receipt vocabulary. cachePrimeReasonAdmitted means full prompt state is
// resident; cachePrimeReasonCold means the path ran but admitted no state (a legitimate,
// explicitly-labelled outcome, never a silent claim); cachePrimeReasonUnsupported means
// the planner cannot prime at all.
const (
	cachePrimeReasonAdmitted    = "admitted"
	cachePrimeReasonCold        = "cold"
	cachePrimeReasonUnsupported = "unsupported"
)

// CachePrimeReceipt is the closed result of a cache populate. Admitted is the single
// load-bearing bit: only an admitted receipt with matching TokenCount may be treated as
// a usable warm. Usable() folds the invariant so a caller cannot misread a cold or
// partial prime as a hit.
type CachePrimeReceipt struct {
	// Admitted is true only when the full prompt state was admitted to the cache.
	Admitted bool
	// Reason is the closed reason token (admitted | cold | unsupported).
	Reason string
	// Tokens is the number of prompt tokens whose state is resident (0 unless admitted).
	Tokens int
	// PromptTokens is the prompt length the prime was asked to materialize.
	PromptTokens int
	// ReusedTokens is the prefix the production generation path actually accepted.
	ReusedTokens int
	// PrefilledTokens is the prompt suffix the production path computed.
	PrefilledTokens int
}

// Usable reports whether this receipt describes a restorable warm. It is the AND of the
// admitted bit and a non-empty admitted prefix, so a cancelled or partial prime can
// never read as usable.
func (r CachePrimeReceipt) Usable() bool {
	return r.Admitted && r.Reason == cachePrimeReasonAdmitted && r.Tokens > 0
}

// primeCacheStateOnce runs one populate through the production path with the decode lane
// suppressed. It sets the request-local purpose, calls the same method the served path
// calls, and clears the purpose before returning (including on error), so the flag can
// never leak into an ordinary turn.
//
// The returned int is the RESIDENT prefix length AFTER admission, read back from the
// same cache the next demand turn will consult (p.cachedPrefixLen). The generate
// result's own `matched` is the prefix served BEFORE this prime (normally 0) and is not
// the materialization outcome; the read-back is.
func (p *InKernelPlanner) primeCacheStateOnce(ctx context.Context, ids []int) (int, error) {
	resident, _, err := p.primeCacheStateMeasured(ctx, ids)
	return resident, err
}

func (p *InKernelPlanner) primeCacheStateMeasured(ctx context.Context, ids []int) (resident, reused int, err error) {
	p.cachePopulate = true
	defer func() { p.cachePopulate = false }()
	_, _, _, reused, _, _, _, _, err = p.generateReusedContextWithBias(
		ctx, ids, 0, 0, 0, 0, nil, 0, 0, map[int]bool{}, nil)
	if err != nil {
		return 0, reused, err
	}
	if err := ctx.Err(); err != nil {
		return 0, reused, err
	}
	return p.cachedPrefixLen(ids), reused, nil
}

// PrimeCacheState materializes the KV/expert state for ids without generating a token.
// It is the CW-03 cache-populate purpose: full lookup + suffix prefill + full-state
// admission, then exit before sampling or decode. The returned receipt reports whether
// the full prompt state is now resident; an unsupported planner is refused with
// ErrCachePrimeUnsupported, and a cancelled context returns its error with no usable
// receipt (the failed preparation never claims a cache hit).
func (p *InKernelPlanner) PrimeCacheState(ctx context.Context, ids []int) (CachePrimeReceipt, error) {
	if p == nil {
		return CachePrimeReceipt{Reason: cachePrimeReasonUnsupported}, ErrCachePrimeUnsupported
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return CachePrimeReceipt{Reason: cachePrimeReasonCold}, err
	}
	if len(ids) == 0 || p.tree == nil || !inKernelPlannerPrefixReuseSupported(p.m, p.backend) {
		return CachePrimeReceipt{Reason: cachePrimeReasonUnsupported, PromptTokens: len(ids)}, ErrCachePrimeUnsupported
	}
	matched, reused, err := p.primeCacheStateMeasured(ctx, ids)
	prefilled := len(ids) - reused
	if prefilled < 0 {
		prefilled = 0
	}
	if err != nil {
		return CachePrimeReceipt{Reason: cachePrimeReasonCold, PromptTokens: len(ids), ReusedTokens: reused, PrefilledTokens: prefilled}, err
	}
	if matched < len(ids) {
		return CachePrimeReceipt{Admitted: false, Reason: cachePrimeReasonCold, Tokens: matched, PromptTokens: len(ids), ReusedTokens: reused, PrefilledTokens: prefilled}, nil
	}
	return CachePrimeReceipt{Admitted: true, Reason: cachePrimeReasonAdmitted, Tokens: matched, PromptTokens: len(ids), ReusedTokens: reused, PrefilledTokens: prefilled}, nil
}
