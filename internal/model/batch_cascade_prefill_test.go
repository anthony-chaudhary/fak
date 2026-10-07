package model

import (
	"math"
	"strings"
	"testing"
)

// TestCascadePrefillMatchesIndependentPrefill is the correctness rung for the one-pass
// shared-prefix (cascade) prefill: every lane's last-token logits AND every lane's KV
// rows (post-RoPE K, pre-RoPE Kraw, V, positions) must be BIT-identical to an
// independent fresh Session.Prefill(prefix ++ suffix_b) — the same f32 Float32bits
// contract the rectangular PrefillEach lane clears. A follow-on decode step must agree
// too, so the cascade cache is a drop-in for a normally-prefilled one.
func TestCascadePrefillMatchesIndependentPrefill(t *testing.T) {
	cfg := Config{
		HiddenSize: 64, NumLayers: 3, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 128, VocabSize: 200, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	m := NewSynthetic(cfg)
	V := cfg.VocabSize
	for _, tc := range []struct{ P, S, B int }{{11, 5, 4}, {33, 1, 3}, {1, 7, 2}, {20, 9, 1}} {
		prefix := make([]int, tc.P)
		for i := range prefix {
			prefix[i] = (i*37 + 9) % V
		}
		suffixes := make([][]int, tc.B)
		for b := range suffixes {
			suffixes[b] = make([]int, tc.S)
			for i := range suffixes[b] {
				suffixes[b][i] = (b*53 + i*17 + 3) % V
			}
		}
		bs, logits, err := m.CascadePrefill(prefix, suffixes)
		if err != nil {
			t.Fatalf("P=%d S=%d B=%d: CascadePrefill: %v", tc.P, tc.S, tc.B, err)
		}
		if len(logits) != tc.B {
			t.Fatalf("logits rows = %d, want %d", len(logits), tc.B)
		}
		for b := 0; b < tc.B; b++ {
			ref := m.NewSession()
			want := ref.Prefill(append(append([]int(nil), prefix...), suffixes[b]...))
			assertBitsEqual(t, "logits", tc.P, tc.S, b, logits[b], want)
			got := bs.Seqs[b].Cache
			if got.Len() != ref.Cache.Len() {
				t.Fatalf("lane %d cache len %d != %d", b, got.Len(), ref.Cache.Len())
			}
			for i := range got.pos {
				if got.pos[i] != ref.Cache.pos[i] {
					t.Fatalf("lane %d pos[%d] = %d, want %d", b, i, got.pos[i], ref.Cache.pos[i])
				}
			}
			for l := 0; l < cfg.NumLayers; l++ {
				assertBitsEqual(t, "K", tc.P, tc.S, b, got.K[l], ref.Cache.K[l])
				assertBitsEqual(t, "Kraw", tc.P, tc.S, b, got.Kraw[l], ref.Cache.Kraw[l])
				assertBitsEqual(t, "V", tc.P, tc.S, b, got.V[l], ref.Cache.V[l])
			}
			// One decode step on top of the cascade cache matches the reference session.
			next := (b*7 + 1) % V
			assertBitsEqual(t, "step", tc.P, tc.S, b, bs.Seqs[b].Step(next), ref.Step(next))
		}
	}
}

func assertBitsEqual(t *testing.T, what string, P, S, lane int, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("P=%d S=%d lane %d %s: len %d != %d", P, S, lane, what, len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("P=%d S=%d lane %d %s[%d]: cascade %v != independent prefill %v", P, S, lane, what, i, got[i], want[i])
		}
	}
}

// TestCascadePrefillRefusesUnsupportedShapes pins the gate: a non-batch-lane
// architecture and a ragged suffix family are refused by name, never silently run.
func TestCascadePrefillRefusesUnsupportedShapes(t *testing.T) {
	cfg := Config{
		HiddenSize: 32, NumLayers: 1, NumHeads: 2, NumKVHeads: 1, HeadDim: 16,
		IntermediateSize: 64, VocabSize: 50, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	m := NewSynthetic(cfg)
	if _, _, err := m.CascadePrefill([]int{1, 2}, [][]int{{3, 4}, {5}}); err == nil || !strings.Contains(err.Error(), "equal length") {
		t.Fatalf("ragged suffixes: err = %v, want equal-length refusal", err)
	}
	if _, _, err := m.CascadePrefill(nil, [][]int{{3}}); err == nil {
		t.Fatal("empty prefix accepted")
	}
	if _, _, err := m.CascadePrefill(make([]int, batchRectPrefillMaxTokens+1), [][]int{{3}}); err == nil {
		t.Fatal("over-cap prefix accepted")
	}
	alibi := cfg
	alibi.Alibi = true
	ma := &Model{Cfg: alibi}
	if _, _, err := ma.CascadePrefill([]int{1}, [][]int{{2}}); err == nil || !strings.Contains(err.Error(), "cascade prefill refused") {
		t.Fatalf("alibi arch: err = %v, want refusal", err)
	}
}
