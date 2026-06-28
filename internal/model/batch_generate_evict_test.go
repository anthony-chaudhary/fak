package model

import "testing"

// TestGenerateBatchReclaimsFinishedSlots is the #36 witness that a finished sequence's slot is
// FREED rather than kept rectangular by re-feeding its own EOS. It drives GenerateBatch with an
// EOS id chosen so the lanes finish RAGGEDLY (some emit it within n steps, at least one never
// does), then pins two properties the old "re-feed EOS into the done slot" path could not both
// satisfy:
//
//  1. CORRECTNESS — every user's output is bit-identical to serial Generate, INCLUDING the
//     lanes that finish early (each stops at exactly the same EOS) and the survivors (whose
//     decode must be unperturbed by their co-batch neighbours retiring). Dropping a finished
//     lane out of the active set must not move a single surviving token.
//  2. RECLAMATION — a finished lane's KV cache STOPS GROWING at the step it finishes
//     (cache.Len == prefill + emitted - 1), strictly short of prefill + n. Under the old path a
//     done slot was re-fed its EOS every remaining step, so its cache would have advanced the
//     full prefill + n; this assertion fails on that behaviour and passes only once the slot is
//     genuinely released — the work the issue says static batching was wasting on done slots.
//
// Synthetic weights suffice: argmax is independent of which id is EOS, so the full untruncated
// sequences fix each lane's tokens, EOS only truncates them, and per-lane KV ownership makes the
// reclamation a structural fact (each lane's cache is its own object, appended only when active).
func TestGenerateBatchReclaimsFinishedSlots(t *testing.T) {
	cfg := Config{
		HiddenSize: 64, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 128, VocabSize: 200, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1, // start with no EOS so the full sequences are visible
	}
	m := NewSynthetic(cfg)
	V := cfg.VocabSize
	const B = 4
	const P = 5 // equal prompt length so cache lengths are directly comparable across lanes
	const n = 12

	prompts := make([][]int, B)
	for b := 0; b < B; b++ {
		p := make([]int, P)
		for i := range p {
			p[i] = (b*61 + i*37 + 7) % V // distinct content per lane → distinct argmax streams
		}
		prompts[b] = p
	}

	// full[b]: each lane's untruncated greedy continuation (EOS disabled), so we know exactly
	// what token each lane emits at each step regardless of which id we later make EOS.
	full := make([][]int, B)
	for b := 0; b < B; b++ {
		full[b] = m.NewSession().Generate(prompts[b], n)
		if len(full[b]) != n {
			t.Fatalf("setup: lane %d full continuation has %d tokens, want %d", b, len(full[b]), n)
		}
	}

	// Choose an EOS id that makes the run ragged: at least one lane emits it within n steps
	// (finishes early) AND at least one lane never does (survives the full run).
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
		t.Fatal("setup: no EOS id yields a ragged finish across these synthetic lanes")
	}

	// ref[b]: the serial reference under EOS=eos — full[b] truncated at (and including) the first
	// occurrence of eos. This is exactly what serial Generate produces once eos stops it.
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
		t.Fatalf("setup: run is not ragged (finishedEarly=%d survived=%d)", finishedEarly, survived)
	}

	// Measure the post-prefill cache length the batched path starts from, so the reclamation
	// assertion is independent of prefill internals (it is the same prefill GenerateBatch runs).
	probe := m.NewBatchSession(B)
	probe.PrefillEach(prompts)
	prefillLen := probe.Seqs[0].Cache.Len()

	// Now make eos a real EOS and run the batched generation. Finished lanes must be reclaimed.
	m.Cfg.EOSTokenID = eos
	bs := m.NewBatchSession(B)
	got := bs.GenerateBatch(prompts, n)

	for b := 0; b < B; b++ {
		if !equalInts(got[b], ref[b]) {
			t.Fatalf("lane %d: GenerateBatch %v != serial reference %v", b, got[b], ref[b])
		}
		cacheLen := bs.Seqs[b].Cache.Len()
		if len(ref[b]) < n {
			// finished early: the slot was released, so its cache stopped at the finishing step.
			wantLen := prefillLen + len(ref[b]) - 1
			if cacheLen != wantLen {
				t.Fatalf("lane %d finished after %d tokens but cache.Len=%d, want %d (slot not reclaimed — was it re-fed EOS?)",
					b, len(ref[b]), cacheLen, wantLen)
			}
			if cacheLen >= prefillLen+n {
				t.Fatalf("lane %d finished early yet cache advanced to %d (>= prefill+n=%d): the done slot was still being stepped",
					b, cacheLen, prefillLen+n)
			}
		} else {
			// survived the full run: stepped every iteration.
			if cacheLen != prefillLen+n {
				t.Fatalf("surviving lane %d cache.Len=%d, want %d (prefill %d + %d steps)",
					b, cacheLen, prefillLen+n, prefillLen, n)
			}
		}
	}
}

// firstIndexOf returns the index of the first occurrence of v in s, or -1.
func firstIndexOf(s []int, v int) int {
	for i, x := range s {
		if x == v {
			return i
		}
	}
	return -1
}

// equalInts reports whether two int slices are element-wise equal.
func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
