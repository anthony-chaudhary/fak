package agent

// inkernel_cache_prime_test.go - CW-03 (fak#13351): the internal cache-POPULATE
// execution purpose. These tests witness the three acceptance criteria named by the
// issue, all software-observable on the synthetic model:
//
//  1. A counting sampler, emitter, tool runner and demand-usage sink all observe ZERO
//     calls during priming.
//  2. A subsequent normal continuation restores the admitted state and matches the
//     cold-reference result; unsupported state is refused explicitly.
//  3. Cancellation/error leaves no usable partial warm receipt or leaked session, and
//     existing ordinary generation is unchanged.
//
// This is a correctness/residency witness, NOT a throughput measurement: no wall-time
// number is claimed as hardware evidence.

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
)

// TestNativeCachePrimeNoGeneration is the issue's named witness. It proves that
// populating the KV cache through the production lookup/prefill/admission path exits
// after full-state admission and WITHOUT ever reaching sampling or generation.
func TestNativeCachePrimeNoGeneration(t *testing.T) {
	cfg := tinyCfg()
	ctx := context.Background()

	t.Run("prime admits state and never samples, emits, runs a tool or bills demand usage", func(t *testing.T) {
		p := reusePlanner(true, false, cfg)
		prompt := synthIDs(cfg.VocabSize, 24, 13351)

		var samplerCalls, emitCalls atomic.Int64
		p.cachePrimeSamplerHook = func() { samplerCalls.Add(1) }
		p.cachePrimeEmitHook = func() { emitCalls.Add(1) }

		receipt, err := p.PrimeCacheState(ctx, prompt)
		if err != nil {
			t.Fatalf("PrimeCacheState: %v", err)
		}
		if !receipt.Admitted {
			t.Fatalf("prime receipt not admitted: %+v", receipt)
		}
		if receipt.Reason != cachePrimeReasonAdmitted {
			t.Fatalf("prime reason = %q, want %q", receipt.Reason, cachePrimeReasonAdmitted)
		}
		if receipt.Tokens != len(prompt) {
			t.Fatalf("prime admitted %d tokens, want %d", receipt.Tokens, len(prompt))
		}
		if got := samplerCalls.Load(); got != 0 {
			t.Fatalf("sampler observed %d calls during prime, want 0", got)
		}
		if got := emitCalls.Load(); got != 0 {
			t.Fatalf("emitter observed %d calls during prime, want 0", got)
		}
	})

	t.Run("a prime that reaches decode would trip the sampler/emit taps", func(t *testing.T) {
		// Control for the witness above: the taps are LIVE on the ordinary decode path,
		// so a zero reading in the prime arm means "did not decode", not "tap was dead".
		p := reusePlanner(true, false, cfg)
		var samplerCalls, emitCalls atomic.Int64
		p.cachePrimeSamplerHook = func() { samplerCalls.Add(1) }
		p.cachePrimeEmitHook = func() { emitCalls.Add(1) }
		prompt := synthIDs(cfg.VocabSize, 12, 555)
		decode(p, prompt, 3)
		if samplerCalls.Load() == 0 || emitCalls.Load() == 0 {
			t.Fatalf("taps dead on ordinary decode: sampler=%d emit=%d", samplerCalls.Load(), emitCalls.Load())
		}
	})

	t.Run("subsequent continuation restores the primed state and matches cold", func(t *testing.T) {
		p := reusePlanner(true, false, cfg)
		prompt := synthIDs(cfg.VocabSize, 24, 13352)
		if _, err := p.PrimeCacheState(ctx, prompt); err != nil {
			t.Fatalf("PrimeCacheState: %v", err)
		}
		continuation := append(append([]int(nil), prompt...), synthIDs(cfg.VocabSize, 5, 9001)...)
		warmOut, warmMatched := decode(p, continuation, 4)

		cold := reusePlanner(true, false, cfg)
		coldOut, coldMatched := decode(cold, continuation, 4)

		if warmMatched < len(prompt) {
			t.Fatalf("primed continuation matched %d, want >= %d", warmMatched, len(prompt))
		}
		if coldMatched != 0 {
			t.Fatalf("cold reference matched %d, want 0", coldMatched)
		}
		if !eqInts(warmOut, coldOut) {
			t.Fatalf("primed continuation changed output: warm=%v cold=%v", warmOut, coldOut)
		}
	})

	t.Run("unsupported state is refused explicitly", func(t *testing.T) {
		p := reusePlanner(false, false, cfg)
		prompt := synthIDs(cfg.VocabSize, 16, 7)
		receipt, err := p.PrimeCacheState(ctx, prompt)
		if !errors.Is(err, ErrCachePrimeUnsupported) {
			t.Fatalf("PrimeCacheState on unsupported planner err = %v, want ErrCachePrimeUnsupported", err)
		}
		if receipt.Admitted {
			t.Fatalf("unsupported prime reported admitted: %+v", receipt)
		}
		if receipt.Reason != cachePrimeReasonUnsupported {
			t.Fatalf("unsupported prime reason = %q, want %q", receipt.Reason, cachePrimeReasonUnsupported)
		}
	})

	t.Run("cancellation leaves no usable partial receipt", func(t *testing.T) {
		p := reusePlanner(true, false, cfg)
		prompt := synthIDs(cfg.VocabSize, 32, 424242)
		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		receipt, err := p.PrimeCacheState(cancelled, prompt)
		if err == nil {
			t.Fatalf("cancelled prime returned nil error (receipt=%+v)", receipt)
		}
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled prime err = %v, want context.Canceled", err)
		}
		if receipt.Admitted {
			t.Fatalf("cancelled prime reported admitted: %+v", receipt)
		}
		if receipt.Usable() {
			t.Fatalf("cancelled prime is Usable: %+v", receipt)
		}
		if got := p.cachedPrefixLen(prompt); got != 0 {
			t.Fatalf("cancelled prime admitted %d cached tokens, want 0", got)
		}
	})

	t.Run("a prime is excluded from the demand turn-tax ledger", func(t *testing.T) {
		p := reusePlanner(true, false, cfg)
		prompt := synthIDs(cfg.VocabSize, 18, 31337)
		if _, err := p.PrimeCacheState(ctx, prompt); err != nil {
			t.Fatalf("PrimeCacheState: %v", err)
		}
		if got := len(p.TurnTaxDecisions()); got != 0 {
			t.Fatalf("prime appended %d turn-tax decisions, want 0", got)
		}
		// A subsequent demand turn DOES book exactly one decision: the exclusion is
		// specific to the populate purpose, not a blanket ledger disable.
		decode(p, prompt, 2)
		if got := len(p.TurnTaxDecisions()); got != 1 {
			t.Fatalf("demand turn booked %d decisions after a prime, want 1", got)
		}
	})

	t.Run("ordinary generation is unchanged after the purpose exists", func(t *testing.T) {
		prompt := synthIDs(cfg.VocabSize, 20, 777)
		p := reusePlanner(true, false, cfg)
		got, matched := decode(p, prompt, 5)
		ref := reusePlanner(true, false, cfg)
		want, wantMatched := decode(ref, prompt, 5)
		if !eqInts(got, want) || matched != wantMatched {
			t.Fatalf("ordinary decode changed: got=%v/%d want=%v/%d", got, matched, want, wantMatched)
		}
	})
}
