package model

import "testing"

// TestGenerateBatchReclaimsFinishedSlotsQ8 is the quantized-decode companion to
// TestGenerateBatchReclaimsFinishedSlots: it proves the #36 finished-slot reclamation in
// GenerateBatch holds bit-exactly on the Q8 (int8 SDOT) decode path — the path the hybrid-Metal
// / NEON serving lane actually runs — so the issue's "bit-identical to serial Generate for
// surviving sequences" guarantee spans BOTH the f32 and the quantized kernels, not just f32.
// There is no prior GenerateBatch-vs-serial witness on the Q8 path, so this also closes that
// coverage gap: the batched Q8 generation must match Q8 serial Generate token-for-token.
func TestGenerateBatchReclaimsFinishedSlotsQ8(t *testing.T) {
	cfg := Config{
		HiddenSize: 64, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 128, VocabSize: 257, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1, // start EOS-free so the full sequences are visible
	}
	m := NewSynthetic(cfg)
	m.Quantize()
	V := cfg.VocabSize
	const B = 4
	const P = 5 // equal prompt length so cache lengths are directly comparable across lanes
	const n = 12

	prompts := make([][]int, B)
	for b := 0; b < B; b++ {
		p := make([]int, P)
		for i := range p {
			p[i] = (b*61 + i*37 + 7) % V
		}
		prompts[b] = p
	}

	// full[b]: each lane's untruncated Q8 serial continuation (EOS disabled).
	full := make([][]int, B)
	for b := 0; b < B; b++ {
		s := m.NewSession()
		s.Quant = true
		full[b] = s.Generate(prompts[b], n)
		if len(full[b]) != n {
			t.Fatalf("setup: lane %d Q8 full continuation has %d tokens, want %d", b, len(full[b]), n)
		}
	}

	// Pick an EOS id that makes the run ragged: at least one lane emits it within n steps and at
	// least one never does.
	eos := -1
	for cand := 0; cand < V && eos < 0; cand++ {
		finishes, survives := false, false
		for b := 0; b < B; b++ {
			if firstIndexOf(full[b], cand) >= 0 {
				finishes = true
			} else {
				survives = true
			}
		}
		if finishes && survives {
			eos = cand
		}
	}
	if eos < 0 {
		t.Fatal("setup: no EOS id yields a ragged Q8 finish across these synthetic lanes")
	}

	ref := make([][]int, B)
	finishedEarly, survived := 0, 0
	for b := 0; b < B; b++ {
		if k := firstIndexOf(full[b], eos); k >= 0 {
			ref[b] = full[b][:k+1]
			finishedEarly++
		} else {
			ref[b] = full[b]
			survived++
		}
	}
	if finishedEarly == 0 || survived == 0 {
		t.Fatalf("setup: Q8 run is not ragged (finishedEarly=%d survived=%d)", finishedEarly, survived)
	}

	// Post-prefill cache length under Q8 (the same prefill GenerateBatch runs), so the
	// reclamation assertion is independent of prefill internals.
	probe := m.NewBatchSession(B)
	probe.SetQuant(true)
	probe.PrefillEach(prompts)
	prefillLen := probe.Seqs[0].Cache.Len()

	m.Cfg.EOSTokenID = eos
	bs := m.NewBatchSession(B)
	bs.SetQuant(true)
	got := bs.GenerateBatch(prompts, n)

	for b := 0; b < B; b++ {
		if !equalInts(got[b], ref[b]) {
			t.Fatalf("lane %d: Q8 GenerateBatch %v != Q8 serial reference %v", b, got[b], ref[b])
		}
		cacheLen := bs.Seqs[b].Cache.Len()
		if len(ref[b]) < n {
			wantLen := prefillLen + len(ref[b]) - 1
			if cacheLen != wantLen {
				t.Fatalf("Q8 lane %d finished after %d tokens but cache.Len=%d, want %d (slot not reclaimed — re-fed EOS?)",
					b, len(ref[b]), cacheLen, wantLen)
			}
			if cacheLen >= prefillLen+n {
				t.Fatalf("Q8 lane %d finished early yet cache advanced to %d (>= prefill+n=%d): done slot still stepped",
					b, cacheLen, prefillLen+n)
			}
		} else {
			if cacheLen != prefillLen+n {
				t.Fatalf("Q8 surviving lane %d cache.Len=%d, want %d (prefill %d + %d steps)",
					b, cacheLen, prefillLen+n, prefillLen, n)
			}
		}
	}
}
