package model

import (
	"math"
	"testing"
)

// TestQwen35HybridLongHorizonDecodeSelfConsistent is a regression guard for issue #4273. The
// reported Qwen3.6-27B degeneracy (output collapses into a verbatim repeat loop) is, per the
// on-box q36rawrepro witness, specific to DECODE length through the qwen35-hybrid path
// (Gated-DeltaNet linear-attention layers + gated full attention): the completion opens coherent
// and grounded, then degrades ~40 tokens into decode and cycles to the token cap — the same short
// prompt that decodes only ~2 tokens before EOS stays clean. That localizes the suspect to
// accumulating numeric behavior in the per-token Step path (linearAttnStep — the GDN rank-1
// delta-rule recurrent update + depthwise conv cache) and the full-attention KV carry, exercised
// only over a long decode. The existing hybrid witnesses stop at N<=16, so this closes the
// coverage gap at the failure horizon (N=1300) by pinning three invariants the dev box CAN check
// on synthetic weights:
//
//  1. Every logit stays finite over the whole long prefill and across 500 decoded tokens — a
//     non-finite blow-up in the recurrent scan is the machine-checkable shape of "degenerates
//     over length".
//  2. The incremental Step path agrees with a fresh cacheless Forward (a separate entrypoint with
//     no Session/Cache) at checkpoints out to N=1300, within the established hybrid tolerance —
//     the per-token GDN step + conv + KV carry reproduces the cacheless math at length.
//  3. A whole-prompt Prefill(N) stays BIT-IDENTICAL to a split Prefill(k)+Step*(N-k) — the carry
//     across the prefill/decode boundary never drifts as context grows.
//
// Invalidating assumption (per the generation frame, kept explicit): on the qwen35-hybrid path
// prefill is itself a sequential token-loop, so Forward / Prefill / Step share one numeric path
// and these deltas are ~0 on synthetic weights. This guard therefore CANNOT reproduce the real
// 27B collapse and GREEN does NOT prove #4273 fixed — that needs the on-box 16.7GB Q4_K_M GGUF,
// where the drift lives in real-weight fp32 accumulation (host-gated). What GREEN proves is that
// the hybrid long-horizon code path is finite and self-consistent, so a future regression (a
// ring-buffer wrap, a scratch overflow, a step-path that diverges from the cacheless reference)
// is caught here, and this is the scaffold the eventual on-box real-weight repro attaches to.
func TestQwen35HybridLongHorizonDecodeSelfConsistent(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)

	const N = 1300
	const split = 800 // past the last full-attention refresh, so the GDN and KV carries both cross the boundary

	// Deterministic pseudo-random token stream past the ~1.3k-token failure horizon. A fixed LCG
	// keeps it stable (bit-exactness demands determinism) without Date/rand; every id lands in
	// [1, VocabSize-2], never the 0 / EOS edges.
	ids := make([]int, N)
	x := uint32(0x9e3779b9)
	for i := range ids {
		x = x*1664525 + 1013904223
		ids[i] = int(x%uint32(cfg.VocabSize-2)) + 1
	}

	assertFinite := func(logits []float32, where string) {
		t.Helper()
		for i, v := range logits {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("#4273: %s logit[%d]=%v non-finite", where, i, v)
			}
		}
	}

	// Whole-prompt prefill reference (invariant 1 + the bit-exact target for invariant 3).
	full := m.NewSession().Prefill(ids)
	assertFinite(full, "whole prefill (N=1300)")

	// Incremental decode: Prefill(split) then Step one token at a time to N. Check finiteness at
	// every step, and at checkpoints cross-check against a fresh cacheless Forward (invariant 2).
	checkpoints := []int{split + 1, split + 50, split + 150, split + 300, N}
	ci := 0
	s := m.NewSession()
	s.Prefill(ids[:split])
	var inc []float32
	for p := split; p < N; p++ {
		inc = s.Step(ids[p])
		pos := p + 1
		assertFinite(inc, "incremental decode")
		if ci < len(checkpoints) && pos == checkpoints[ci] {
			ref := m.Forward(ids[:pos]).Logits[pos-1]
			if d := maxAbsDelta(ref, inc); d > 2e-5 {
				t.Fatalf("#4273: incremental decode diverges from cacheless Forward at pos=%d (%d decoded): max|delta|=%g", pos, pos-split, d)
			}
			ci++
		}
	}

	// Invariant 3: whole prefill == split prefill/decode, bit-for-bit, at the horizon.
	assertFloat32BitsEqual(t, "#4273 hybrid long-horizon split prefill/decode (N=1300)", full, inc)
}
