// Tests for cmd/o1proof1b's pure helpers (lcgIDs, cat, maxAbsDiff, ms,
// qwen25_1_5b) and for the two run arms (boundedRun, naiveRun) against a
// TINY synthetic model — the same shape internal/kvmmu's tests use — so the
// O(1)-vs-O(M) residency contract the witness binary prints is asserted
// here mechanically, without building the ~1.5B-parameter config.
package main

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// tinyCfg mirrors internal/kvmmu/kvmmu_test.go's synthCfg: small enough to
// build in milliseconds, real enough that Append/ApplyPlan run end to end.
func tinyCfg() model.Config {
	return model.Config{
		HiddenSize: 32, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 48, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1, ModelType: "llama",
	}
}

func TestQwen25_1_5bShape(t *testing.T) {
	cfg := qwen25_1_5b()
	// The doc comment pins this to sessionbench's "qwen25-1.5b" shape; if any
	// of these drift, the witness is no longer the claimed ~1.5B-scale run.
	if cfg.HiddenSize != 1536 {
		t.Errorf("HiddenSize = %d, want 1536", cfg.HiddenSize)
	}
	if cfg.NumLayers != 28 {
		t.Errorf("NumLayers = %d, want 28", cfg.NumLayers)
	}
	if cfg.NumHeads != 12 || cfg.NumKVHeads != 2 || cfg.HeadDim != 128 {
		t.Errorf("heads = %d/%d headdim = %d, want 12/2 headdim 128",
			cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim)
	}
	if cfg.IntermediateSize != 8960 {
		t.Errorf("IntermediateSize = %d, want 8960", cfg.IntermediateSize)
	}
	if cfg.VocabSize != 151936 {
		t.Errorf("VocabSize = %d, want 151936", cfg.VocabSize)
	}
	if !cfg.TieWordEmbeddings {
		t.Error("TieWordEmbeddings = false, want true (qwen2.5-1.5b ties them)")
	}
	if cfg.ModelType != "qwen2" {
		t.Errorf("ModelType = %q, want %q", cfg.ModelType, "qwen2")
	}
}

func TestLCGIDs(t *testing.T) {
	const n, vocab = 64, 48

	got := lcgIDs(n, vocab, 7)
	if len(got) != n {
		t.Fatalf("len = %d, want %d", len(got), n)
	}
	for i, id := range got {
		if id < 0 || id >= vocab {
			t.Fatalf("ids[%d] = %d, out of range [0,%d)", i, id, vocab)
		}
	}

	// Deterministic: the same (n, vocab, seed) must reproduce byte for byte —
	// the witness binary relies on this to make phases comparable.
	again := lcgIDs(n, vocab, 7)
	for i := range got {
		if got[i] != again[i] {
			t.Fatalf("not deterministic: ids[%d] = %d then %d for the same seed", i, got[i], again[i])
		}
	}

	// Seed-sensitive: a different seed must produce a different stream.
	other := lcgIDs(n, vocab, 8)
	same := true
	for i := range got {
		if got[i] != other[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("seeds 7 and 8 produced identical streams; lcgIDs ignores its seed")
	}

	// Not degenerate: the stream must not be one constant value.
	allEq := true
	for _, id := range got[1:] {
		if id != got[0] {
			allEq = false
			break
		}
	}
	if allEq {
		t.Errorf("all %d ids equal %d; generator is degenerate", n, got[0])
	}

	if empty := lcgIDs(0, vocab, 1); len(empty) != 0 {
		t.Errorf("lcgIDs(0, ...) len = %d, want 0", len(empty))
	}
}

func TestCat(t *testing.T) {
	got := cat([]int{1, 2}, nil, []int{3}, []int{}, []int{4, 5, 6})
	want := []int{1, 2, 3, 4, 5, 6}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d (got %v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("cat[%d] = %d, want %d (got %v)", i, got[i], want[i], got)
		}
	}
	if out := cat(); out != nil {
		t.Errorf("cat() = %v, want nil", out)
	}

	// cat must copy, not alias: mutating the result may not write back into
	// an input slice (the witness reuses the same turn slices across phases).
	src := []int{9, 9}
	out := cat(src, []int{1})
	out[0] = 42
	if src[0] != 9 {
		t.Errorf("cat aliased its input: src[0] = %d after mutating the output, want 9", src[0])
	}
}

func TestMaxAbsDiff(t *testing.T) {
	if d := maxAbsDiff([]float32{1, 2, 3}, []float32{1, 2, 3}); d != 0 {
		t.Errorf("identical slices: d = %v, want 0", d)
	}
	if d := maxAbsDiff([]float32{1, -4, 3}, []float32{1, 2, 2.5}); d != 6 {
		t.Errorf("d = %v, want 6 (|-4-2|)", d)
	}
	// Sign-symmetric: |a-b| == |b-a|.
	if d := maxAbsDiff([]float32{1, 2, 2.5}, []float32{1, -4, 3}); d != 6 {
		t.Errorf("swapped args: d = %v, want 6", d)
	}
	// Documented (if blunt) behavior: comparison runs over the SHORTER
	// length, so a tail-only divergence past len(b) is not seen.
	if d := maxAbsDiff([]float32{1, 2, 100}, []float32{1, 2}); d != 0 {
		t.Errorf("short b: d = %v, want 0 (only min length compared)", d)
	}
	if d := maxAbsDiff(nil, []float32{5}); d != 0 {
		t.Errorf("nil a: d = %v, want 0", d)
	}
}

func TestMs(t *testing.T) {
	cases := []struct {
		d    time.Duration
		want float64
	}{
		{0, 0},
		{time.Millisecond, 1},
		{1500 * time.Microsecond, 1.5},
		{2 * time.Second, 2000},
	}
	for _, c := range cases {
		if got := ms(c.d); got != c.want {
			t.Errorf("ms(%v) = %v, want %v", c.d, got, c.want)
		}
	}
}

// TestBoundedRunConstantResidency is the package's core O(1) claim at test
// scale: once the K-turn window has filled, residency must sit at exactly
// sysLen + K*turnLen forever, no matter how many more turns arrive.
func TestBoundedRunConstantResidency(t *testing.T) {
	m := model.NewSynthetic(tinyCfg())
	const (
		sysLen  = 2
		turnLen = 4
		k       = 3
		nTurns  = 9
	)
	vocab := tinyCfg().VocabSize
	sys := lcgIDs(sysLen, vocab, 1)
	turns := make([][]int, nTurns)
	for i := range turns {
		turns[i] = lcgIDs(turnLen, vocab, uint64(100+i))
	}

	durs, lens, c := boundedRun(m, sys, turns, k)
	if len(durs) != nTurns || len(lens) != nTurns {
		t.Fatalf("got %d durs / %d cacheLens, want %d each", len(durs), len(lens), nTurns)
	}

	for i, got := range lens {
		resident := i + 1 // turns in the window after turn i's plan
		if resident > k {
			resident = k
		}
		want := sysLen + resident*turnLen
		if got != want {
			t.Errorf("cacheLens[%d] = %d, want %d", i, got, want)
		}
	}

	// The load-bearing bound: the final residency equals the constant target
	// and matches the live context, i.e. eviction actually happened.
	wantConst := sysLen + k*turnLen
	if lens[nTurns-1] != wantConst {
		t.Errorf("final residency = %d, want constant %d", lens[nTurns-1], wantConst)
	}
	if got := c.CacheLen(); got != wantConst {
		t.Errorf("context CacheLen() = %d, want %d", got, wantConst)
	}
}

// TestNaiveRunGrowsLinearly pins the contrast arm: with no eviction,
// residency after turn i is exactly sysLen + (i+1)*turnLen.
func TestNaiveRunGrowsLinearly(t *testing.T) {
	m := model.NewSynthetic(tinyCfg())
	const (
		sysLen  = 2
		turnLen = 4
		nTurns  = 5
	)
	vocab := tinyCfg().VocabSize
	sys := lcgIDs(sysLen, vocab, 1)
	turns := make([][]int, nTurns)
	for i := range turns {
		turns[i] = lcgIDs(turnLen, vocab, uint64(200+i))
	}

	durs, lens, c := naiveRun(m, sys, turns)
	if len(durs) != nTurns || len(lens) != nTurns {
		t.Fatalf("got %d durs / %d cacheLens, want %d each", len(durs), len(lens), nTurns)
	}
	for i, got := range lens {
		want := sysLen + (i+1)*turnLen
		if got != want {
			t.Errorf("cacheLens[%d] = %d, want %d (naive arm must grow O(M))", i, got, want)
		}
	}
	if got, want := c.CacheLen(), sysLen+nTurns*turnLen; got != want {
		t.Errorf("context CacheLen() = %d, want %d", got, want)
	}
}
