package model

import (
	"math"
	"testing"
)

// tensor_parallel_vocab_test.go — the gates for the VOCAB-PARALLEL tensor-parallel subsystem
// (tensor_parallel_vocab.go, the vLLM VocabParallelEmbedding/ParallelLMHead port). The gate
// grammar matches tensor_parallel_forward_test.go:
//
//   - the masked-input helper == upstream get_masked_input_and_mask's degenerate form
//     (band test, out-of-band -> row 0 + mask);
//   - the vocab-parallel embedding == (max|Δ|=0) the monolith embedBand at ANY rank count
//     (only the owning rank contributes a nonzero row, so the AllReduce adds zeros);
//   - the vocab-parallel head == (max|Δ|=0) the monolith logitsFromHidden tail at ANY rank
//     count (each logit is one rank's fdot over the same weight rows; the gather concat in
//     rank order IS token-id order — the identity form of upstream's
//     get_sharded_to_full_mapping);
//   - ForwardTP(VocabRanks=N) == (max|Δ|=0) ForwardTP(VocabRanks<=1) end to end, which is
//     itself bit-exact vs Forward at ranks=1 (the transitive HF-oracle gate);
//   - every unsupported seam refuses with a named error (out-of-range id, quant-resident
//     head, prism transform, LoRA delta, packed Q2K table).

// tpVocabIDs spans the whole fixture vocabulary, including band boundaries for every rank
// count under test (0 and V-1 are the outer edges; the interior ids cross every band seam).
func tpVocabIDs(vocab int) []int {
	ids := make([]int, 0, 12)
	for _, id := range []int{0, 1, 3, 7, 8, 15, 16, 23, 24, 31, 39} {
		if id < vocab {
			ids = append(ids, id)
		}
	}
	return append(ids, vocab-1)
}

// TestMaskedVocabShardInputPortsUpstreamMask pins the degenerate form of upstream
// get_masked_input_and_mask: in-band ids shift to the local band index, out-of-band ids map
// to local row 0 with the input mask set (the rows masked_fill_ zeroes before the reduce).
func TestMaskedVocabShardInputPortsUpstreamMask(t *testing.T) {
	s := TPShard{Rank: 1, Lo: 8, Hi: 24}
	ids := []int{0, 7, 8, 9, 23, 24, 39}
	local, mask := maskedVocabShardInput(s, ids)
	wantLocal := []int{0, 0, 0, 1, 15, 0, 0}
	wantMask := []bool{true, true, false, false, false, true, true}
	for i := range ids {
		if local[i] != wantLocal[i] || mask[i] != wantMask[i] {
			t.Fatalf("id %d: got local=%d mask=%v, want local=%d mask=%v", ids[i], local[i], mask[i], wantLocal[i], wantMask[i])
		}
	}
}

// TestVocabParallelEmbeddingMatchesMonolith gates the vocab-parallel embedding against the
// monolith embedBand (which includes the arch embed scale) at max|Δ|=0 for every rank count,
// including non-divisible tilings (3 and 5 over V=40 exercise the remainder-on-first-shards
// bands upstream never produces with its padded divide).
func TestVocabParallelEmbeddingMatchesMonolith(t *testing.T) {
	m := NewSynthetic(tpFwdBaseCfg())
	ids := tpVocabIDs(m.Cfg.VocabSize)
	ref := m.embedBand(ids)
	for _, r := range []int{1, 2, 3, 5, 8} {
		plan, err := tpVocabPlan(m.Cfg, r)
		if err != nil {
			t.Fatalf("tpVocabPlan(r=%d): %v", r, err)
		}
		got, err := m.tpEmbedLayer(ids, plan, LocalCollective{})
		if err != nil {
			t.Fatalf("tpEmbedLayer(r=%d): %v", r, err)
		}
		for tPos := range ids {
			for i := range ref[tPos] {
				if d := math.Abs(float64(got[tPos][i] - ref[tPos][i])); d != 0 {
					t.Fatalf("r=%d pos=%d idx=%d: max|Δ|=%g, want 0 (bit-exact)", r, tPos, i, d)
				}
			}
		}
	}
}

// TestVocabParallelHeadLogitsMatchesMonolith gates the vocab-parallel head against the
// monolith tail logitsFromHidden (final norm -> head matmul -> logit scale) at max|Δ|=0,
// first with the synthetic model's TIED head (head == embed table), then with an UNTIED
// head (an injected lm_head.weight distinct from the embedding).
func TestVocabParallelHeadLogitsMatchesMonolith(t *testing.T) {
	cfg := tpFwdBaseCfg()
	H, V := cfg.HiddenSize, cfg.VocabSize
	m := NewSynthetic(cfg)
	// Any hidden vectors work as the tail's input; use deterministic pseudo-random ones so
	// the head matmul is exercised on non-degenerate activations.
	x := make([][]float32, 4)
	for t := range x {
		x[t] = tpRand(H, uint64(900+t))
	}
	for _, r := range []int{1, 2, 3, 5, 8} {
		plan, err := tpVocabPlan(m.Cfg, r)
		if err != nil {
			t.Fatalf("tpVocabPlan(r=%d): %v", r, err)
		}
		got, err := m.tpHeadLogits(x, plan, LocalCollective{})
		if err != nil {
			t.Fatalf("tpHeadLogits(r=%d): %v", r, err)
		}
		for tPos := range x {
			ref := m.logitsFromHidden(x[tPos])
			for i := range ref {
				if d := math.Abs(float64(got[tPos][i] - ref[i])); d != 0 {
					t.Fatalf("tied head r=%d pos=%d vocab=%d: max|Δ|=%g, want 0 (bit-exact)", r, tPos, i, d)
				}
			}
		}
	}

	// UNTIED head: inject a distinct lm_head.weight; headName() must now resolve to it and
	// the sharded bands must still reproduce the monolith tail bit-exactly.
	head := tpRand(V*H, 777)
	tpInjectTensors(m, map[string]tpTensor{"lm_head.weight": {shape: []int{V, H}, vals: head}})
	if got := m.headName(); got != "lm_head.weight" {
		t.Fatalf("headName after inject = %q, want lm_head.weight", got)
	}
	for _, r := range []int{1, 2, 3, 5, 8} {
		plan, err := tpVocabPlan(m.Cfg, r)
		if err != nil {
			t.Fatalf("tpVocabPlan untied r=%d: %v", r, err)
		}
		got, err := m.tpHeadLogits(x, plan, LocalCollective{})
		if err != nil {
			t.Fatalf("tpHeadLogits untied r=%d: %v", r, err)
		}
		for tPos := range x {
			ref := m.logitsFromHidden(x[tPos])
			for i := range ref {
				if d := math.Abs(float64(got[tPos][i] - ref[i])); d != 0 {
					t.Fatalf("untied head r=%d pos=%d vocab=%d: max|Δ|=%g, want 0 (bit-exact)", r, tPos, i, d)
				}
			}
		}
	}
}

// TestForwardTPVocabRanksMatchesRanks1AndForward gates the WIRED path end to end:
// ForwardTP with the vocab tail armed reproduces the monolith-tail ForwardTP bit-exactly
// (logits AND every hidden layer), and with attn/ffn at 1 rank it reproduces Forward
// bit-exactly — extending the transitive HF-oracle chain to the vocab-parallel tail.
func TestForwardTPVocabRanksMatchesRanks1AndForward(t *testing.T) {
	m := NewSynthetic(tpFwdBaseCfg())
	ids := tpVocabIDs(m.Cfg.VocabSize)
	ref, err := m.ForwardTP(ids, TPConfig{VocabRanks: 1})
	if err != nil {
		t.Fatalf("ForwardTP(vocab=1): %v", err)
	}
	for _, r := range []int{2, 3, 5, 8} {
		got, err := m.ForwardTP(ids, TPConfig{VocabRanks: r})
		if err != nil {
			t.Fatalf("ForwardTP(vocab=%d): %v", r, err)
		}
		for tPos := range ids {
			for i := range ref.Logits[tPos] {
				if d := math.Abs(float64(got.Logits[tPos][i] - ref.Logits[tPos][i])); d != 0 {
					t.Fatalf("vocab r=%d logits pos=%d vocab=%d: max|Δ|=%g, want 0", r, tPos, i, d)
				}
			}
		}
		for l := range ref.Hidden {
			for i := range ref.Hidden[l] {
				if got.Hidden[l][i] != ref.Hidden[l][i] {
					t.Fatalf("vocab r=%d hidden layer %d idx=%d: %g != %g", r, l, i, got.Hidden[l][i], ref.Hidden[l][i])
				}
			}
		}
	}

	// The composed gate: attn=ffn=1 rank + vocab-parallel tail == the monolithic Forward
	// (bit-exact), so the vocab tail inherits the HF oracle transitively.
	fwd := m.Forward(ids)
	got, err := m.ForwardTP(ids, TPConfig{VocabRanks: 4})
	if err != nil {
		t.Fatalf("ForwardTP(vocab=4, attn=1, ffn=1): %v", err)
	}
	for tPos := range ids {
		for i := range fwd.Logits[tPos] {
			if d := math.Abs(float64(got.Logits[tPos][i] - fwd.Logits[tPos][i])); d != 0 {
				t.Fatalf("ForwardTP(vocab=4) vs Forward logits pos=%d vocab=%d: max|Δ|=%g, want 0", tPos, i, d)
			}
		}
	}
}

// TestForwardTPVocabFailClosed pins the named refusals: an out-of-range id, a missing head
// tensor, a quant-resident head, a prism input transform on the head, an active LoRA delta
// on the head, and the packed Q2K embedding table — each refuses the whole forward up front
// instead of mis-serving it.
func TestForwardTPVocabFailClosed(t *testing.T) {
	cfg := tpFwdBaseCfg()
	H, V := cfg.HiddenSize, cfg.VocabSize

	if _, err := NewSynthetic(cfg).ForwardTP([]int{0, V}, TPConfig{VocabRanks: 2}); err == nil {
		t.Fatal("out-of-range id: want a named refusal, got nil")
	} else {
		want := "outside [0,"
		if len(err.Error()) < len(want) || !contains(err.Error(), want) {
			t.Fatalf("out-of-range id refusal = %q, want it to name the range %q", err.Error(), want)
		}
	}
	if _, err := NewSynthetic(cfg).ForwardTP([]int{0, -1}, TPConfig{VocabRanks: 2}); err == nil {
		t.Fatal("negative id: want a named refusal, got nil")
	}

	// Missing head tensor, unit level: a bare model has neither an f32 head nor any
	// quant store, so the head refusal names the missing tensor.
	mb := &Model{Cfg: cfg}
	if err := mb.requireTPHeadResidentF32(); err == nil || !contains(err.Error(), "no resident tensor") {
		t.Fatalf("missing head: want the no-resident-tensor refusal, got %v", err)
	}
	// ForwardTP level, tied model: the head IS the embed table, so stripping the manifest
	// fires the EMBEDDING residency refusal first (the correct first refusal).
	m := NewSynthetic(cfg)
	delete(m.manifest, "model.embed_tokens.weight")
	if _, err := m.ForwardTP([]int{1, 2}, TPConfig{VocabRanks: 2}); err == nil || !contains(err.Error(), "not resident f32") {
		t.Fatalf("stripped tied table: want the embedding residency refusal, got %v", err)
	}

	// Quant-resident head (untied): lm_head present ONLY in the Q8 store — the monolith
	// tail would read it via residentMatRows; the sharded band matmul cannot.
	mq := NewSynthetic(cfg)
	mq.q8w = map[string]*q8Tensor{"lm_head.weight": {out: V, in: H, nblk: 1}}
	if _, err := mq.ForwardTP([]int{1, 2}, TPConfig{VocabRanks: 2}); err == nil || !contains(err.Error(), "quant-resident") {
		t.Fatalf("quant-resident head: want the quant-resident refusal, got %v", err)
	}

	// Prism input transform on the head (the seam residentMatRows applies and the band
	// matmul bypasses).
	mp := NewSynthetic(cfg)
	mp.prism = &prismHadamardState{weightWidth: map[string]int{"model.embed_tokens.weight": H}}
	if _, err := mp.ForwardTP([]int{1, 2}, TPConfig{VocabRanks: 2}); err == nil || !contains(err.Error(), "prism") {
		t.Fatalf("prism head: want the prism refusal, got %v", err)
	}

	// Active LoRA delta on the head.
	ml := NewSynthetic(cfg)
	ml.lora = NewLoRASet()
	rank := 4
	if err := ml.lora.Add(&LoRAAdapter{
		Adapter: "t", Target: "model.embed_tokens.weight",
		In: H, Out: V, Rank: rank, Alpha: 8,
		A: tpRand(rank*H, 31), B: tpRand(V*rank, 32),
	}); err != nil {
		t.Fatalf("lora add: %v", err)
	}
	if _, err := ml.ForwardTP([]int{1, 2}, TPConfig{VocabRanks: 2}); err == nil || !contains(err.Error(), "LoRA") {
		t.Fatalf("lora head: want the LoRA refusal, got %v", err)
	}

	// Packed Q2K embedding table: no f32 rows to shard.
	mq2 := NewSynthetic(cfg)
	mq2.Q2KEmbedding = &Q2KEmbedding{}
	if _, err := mq2.ForwardTP([]int{1, 2}, TPConfig{VocabRanks: 2}); err == nil || !contains(err.Error(), "Q2K") {
		t.Fatalf("Q2K embedding: want the packed-table refusal, got %v", err)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
